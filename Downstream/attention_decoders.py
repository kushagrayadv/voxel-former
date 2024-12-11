import math
import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
import clip
from functools import partial
import random
import json
from tqdm import tqdm
import utils
from models import PriorNetwork, BrainDiffusionPrior
from diffusers.models.vae import Decoder
import sys
sys.path.append('../')
from backbone import BrainNAT

class CrossSelfAttentionLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 encoder_hidden_size=None,
                 attention_probs_dropout_prob=0.0,
                 position_embedding_type="absolute",
                 max_position_embeddings=256,
                 is_cross_attention=False):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        if is_cross_attention:
            if encoder_hidden_size is None:
                raise ValueError("encoder_hidden_size must be provided for cross attention")
            self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
    ):
        is_cross_attention = encoder_hidden_states is not None

        query_layer = self.query(hidden_states)
        if is_cross_attention:
            key_layer = self.key(encoder_hidden_states)
            value_layer = self.value(encoder_hidden_states)
        else:
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)

        # Reshape for multi-head attention
        batch_size, seq_length, _ = query_layer.size()
        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Scaled dot-product attention
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, dropout_p=self.dropout.p
        )

        # Reshape context to original format
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class ResidualConnectionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.LayerNorm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class CrossSelfAttentionBlock(nn.Module):
    def __init__(self, dim, cross_dim, num_heads, mlp_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossSelfAttentionLayer(
            hidden_size=dim,
            num_attention_heads=num_heads,
            encoder_hidden_size=cross_dim,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_seq_len,
            is_cross_attention=True,
        )
        self.projector_1 = ResidualConnectionLayer(dim, mlp_dim, dropout=dropout)

        self.self_attn = CrossSelfAttentionLayer(
            hidden_size=dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=dropout
        )
        self.projector_2 = ResidualConnectionLayer(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cross_attn_input):
        x = self.projector_1(self.cross_attn(x, cross_attn_input), x)
        x = self.projector_2(self.self_attn(x), x)
        return x
    
class SpatialAwareBrainNetwork(nn.Module):
    def __init__(
        self,
        h=4096,
        out_dim=768,
        seq_len=256,
        n_blocks=4,
        num_heads=4,
        drop=0.15,
        blurry_recon=True,
        clip_scale=1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        
        # Initialize learnable queries
        self.queries = nn.Parameter(torch.randn(1, seq_len, h))
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            CrossSelfAttentionBlock(h, h, num_heads, h, seq_len, drop) 
            for _ in range(n_blocks)
        ])
        self.backbone_head = nn.ModuleList([
            ResidualConnectionLayer(h, h),
            ResidualConnectionLayer(h, h),   
        ])
        self.backbone_proj = nn.Linear(h, out_dim)
        # Optionally remove or adjust the clip projection if avoiding MLPs
        if clip_scale > 0:
            self.clip_head = nn.ModuleList([
                ResidualConnectionLayer(h, h),
                ResidualConnectionLayer(h, h),   
            ])
            self.clip_proj = nn.Linear(h, out_dim)
        else:
            self.clip_proj = None

        if self.blurry_recon:
            self.blin1 = nn.Linear(h * seq_len, 4 * 28 * 28, bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1, 512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1, 512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        # Expand queries to batch size
        cross_attn_output = self.queries.repeat(batch_size, 1, 1)
        
        # Apply attention blocks
        for i, attention_block in enumerate(self.attention_blocks):
            cross_attn_output = attention_block(cross_attn_output, x)
            # Debugging shapes
            # print(f"Backbone shape after attention block {i}: {backbone.shape}")
            
        backbone = cross_attn_output
        # backbone output
        for head in self.backbone_head:
            backbone = head(backbone, backbone)
            
        backbone = self.backbone_proj(backbone)
        # CLIP projection if enabled
        if self.clip_proj is not None:
            c = cross_attn_output
            for head in self.clip_head:
                c = head(c, c)
            c = self.clip_proj(cross_attn_output)
        else:
            c = cross_attn_output  # Or handle accordingly if clip_proj is None

        # Initialize blurry reconstruction
        b = torch.zeros((batch_size, 2, 1), device=x.device)
        
        # Apply blurry reconstruction if enabled
        if self.blurry_recon:
            b = self.blin1(x.view(batch_size, -1))
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1)
            b_aux = b_aux.view(batch_size, 49, 512)
            b = (self.bupsampler(b), b_aux)

        return backbone, c, b
    
    
class NAT_BrainNet(nn.Module):
    def __init__(self, args, clip_emb_dim=1664, clip_seq_dim=256):
        super(NAT_BrainNet, self).__init__()
        
        # NAT backbone feature extractor
        self.brain_nat = BrainNAT(
            in_chans=1,
            embed_dim=args.encoder_hidden_dim,
            depth=args.nat_depth,
            num_heads=args.num_heads,
            num_neighbors=args.nat_num_neighbors,
            tome_r=args.tome_r,
            layer_scale_init_value=1e-6,
            coord_dim=3,
            omega_0=30,
            last_n_features=args.last_n_features,
            full_attention=args.full_attention,
            drop_rate=args.drop,
            progressive_dims=args.progressive_dims,
            initial_tokens=args.initial_tokens,
            dim_scale_factor=args.dim_scale_factor
        )
        
        # Linear layer to map brain_nat output to clip_emb_dim
        self.feature_mapper = nn.Linear(self.brain_nat.blocks.final_dim, args.decoder_hidden_dim)

        # Brain Network backbone
        self.backbone = SpatialAwareBrainNetwork(
            h=args.decoder_hidden_dim,       # Dimension of brain_nat output
            out_dim=clip_emb_dim,    # Desired output dimension
            seq_len=clip_seq_dim,
            n_blocks=args.n_blocks_decoder,
            num_heads=args.num_heads,
            drop=args.drop,
            blurry_recon=args.blurry_recon,
            clip_scale=args.clip_scale,
        )
        
        # Optional Prior Network
        # if args.use_prior:
        #     out_dim = clip_emb_dim
        #     depth = 6
        #     dim_head = 52
        #     heads = clip_emb_dim // 52
        #     timesteps = 100
            
        #     prior_network = PriorNetwork(
        #         dim=out_dim,
        #         depth=depth,
        #         dim_head=dim_head,
        #         heads=heads,
        #         causal=False,
        #         num_tokens=clip_seq_dim,
        #         learned_query_mode="pos_emb"
        #     )
            
        #     self.diffusion_prior = BrainDiffusionPrior(
        #         net=prior_network,
        #         image_embed_dim=out_dim,
        #         condition_on_text_encodings=False,
        #         timesteps=timesteps,
        #         cond_drop_prob=0.2,
        #         image_embed_scale=None,
        #     )
        # else:
        #     self.diffusion_prior = None

    def forward(self, x, coords):
        # NAT backbone processing
        x = self.brain_nat(x, coords)
        x = self.feature_mapper(x)
        # Map features to clip_emb_dim
        # x = self.feature_mapper(x)

        # Brain Network processing
        backbone, clip_voxels, blurry_image_enc = self.backbone(x)
        
        # Add shape assertions for debugging
        batch_size = x.shape[0]
        assert backbone.shape[0] == batch_size, f"Expected backbone batch size {batch_size}, got {backbone.shape[0]}"
        assert clip_voxels.shape[0] == batch_size, f"Expected clip_voxels batch size {batch_size}, got {clip_voxels.shape[0]}"
        
        return backbone, clip_voxels, blurry_image_enc


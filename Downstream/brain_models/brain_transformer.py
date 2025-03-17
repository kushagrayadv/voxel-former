import torch
import torch.nn as nn


# Q-former Decoder
class CrossSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        encoder_hidden_size=None,
        attention_probs_dropout_prob=0.0,
        position_embedding_type="absolute",
        max_position_embeddings=256,
        is_cross_attention=False,
    ):
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
                raise ValueError(
                    "encoder_hidden_size must be provided for cross attention"
                )
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
        query_layer = query_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        key_layer = key_layer.view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        value_layer = value_layer.view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

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

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
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
            attention_probs_dropout_prob=dropout,
        )
        self.projector_2 = ResidualConnectionLayer(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cross_attn_input):
        x = self.projector_1(self.cross_attn(x, cross_attn_input), x)
        x = self.projector_2(self.self_attn(x), x)
        return x


class BrainDecoder(nn.Module):
    def __init__(
        self,
        h=4096,
        out_dim=768,
        seq_len=256,
        n_blocks=4,
        num_heads=4,
        drop=0.15,
        blurry_recon=True,
        clip_scale=1,
    ):
        # This is nothing more than a QFormer
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale

        # Initialize learnable queries
        self.queries = nn.Parameter(torch.randn(1, seq_len, h))

        # Attention blocks
        self.attention_blocks = nn.ModuleList(
            [
                CrossSelfAttentionBlock(h, h, num_heads, h, seq_len, drop)
                for _ in range(n_blocks)
            ]
        )
        self.backbone_head = nn.ModuleList(
            [
                ResidualConnectionLayer(h, h),
                ResidualConnectionLayer(h, h),
            ]
        )
        self.backbone_proj = nn.Linear(h, out_dim)
        # Optionally remove or adjust the clip projection if avoiding MLPs
        if clip_scale > 0:
            self.clip_head = nn.ModuleList(
                [
                    ResidualConnectionLayer(h, h),
                    ResidualConnectionLayer(h, h),
                ]
            )
            self.clip_proj = nn.Linear(h, out_dim)
        else:
            self.clip_proj = None

        # TODO: next step, more tasks
        # if self.blurry_recon:
        #     self.blin1 = nn.Linear(h * seq_len, 4 * 28 * 28, bias=True)
        #     self.bdropout = nn.Dropout(.3)
        #     self.bnorm = nn.GroupNorm(1, 64)
        #     self.bupsampler = Decoder(
        #         in_channels=64,
        #         out_channels=4,
        #         up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        #         block_out_channels=[32, 64, 128],
        #         layers_per_block=1,
        #     )
        #     self.b_maps_projector = nn.Sequential(
        #         nn.Conv2d(64, 512, 1, bias=False),
        #         nn.GroupNorm(1, 512),
        #         nn.ReLU(True),
        #         nn.Conv2d(512, 512, 1, bias=False),
        #         nn.GroupNorm(1, 512),
        #         nn.ReLU(True),
        #         nn.Conv2d(512, 512, 1, bias=True),
        #     )

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

        # TODO: next step, more tasks
        # # Apply blurry reconstruction if enabled
        # if self.blurry_recon:
        #     b = self.blin1(x.view(batch_size, -1))
        #     b = self.bdropout(b)
        #     b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
        #     b = self.bnorm(b)
        #     b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1)
        #     b_aux = b_aux.view(batch_size, 49, 512)
        #     b = (self.bupsampler(b), b_aux)

        return backbone, c, b


# Transformer
class BrainTransformer(nn.Module):
    def __init__(self, args):
        super(BrainTransformer, self).__init__()
        model_args = args.model
        from .tomer import Tomer
        from .perceiver_decoder import PerceiverDecoder
        from .linformer import Linformer

        self.decoder_type = model_args.decoder_type
        self.encoder_type = model_args.encoder_type

        if self.decoder_type == "perceiver":
            # For Perceiver, we don't need the encoder
            self.brain_decoder = PerceiverDecoder(
                h=model_args.decoder_hidden_dim,
                out_dim=model_args.clip_emb_dim,
                num_latents=model_args.clip_seq_dim,
                n_blocks=model_args.n_blocks_decoder,
                num_heads=model_args.num_heads,
                head_dim=model_args.head_dim,
                drop=model_args.drop,
                clip_scale=args.train.clip_scale,
                self_per_cross_attn=model_args.self_per_cross_attn,
                input_dim=1,  # Input dimension is 1 for brain signal
            )
        else:  # 'qformer'
            # Use encoder + Q-former decoder
            if self.encoder_type == "linformer":
                self.brain_encoder = Linformer(
                    dim=model_args.encoder_hidden_dim,
                    seq_len=model_args.clip_seq_dim,
                    depth=model_args.nat_depth,
                    k=model_args.encoder_hidden_dim,
                    heads=model_args.num_heads,
                    dim_head=model_args.head_dim,
                    one_kv_head=False,
                    share_kv=False,
                    dropout=model_args.drop,
                )

                self.feature_mapper = nn.Linear(
                    model_args.encoder_hidden_dim, model_args.decoder_hidden_dim
                )

            else:
                self.brain_encoder = Tomer(
                    in_chans=1,
                    embed_dim=model_args.encoder_hidden_dim,
                    depth=model_args.nat_depth,
                    num_heads=model_args.num_heads,
                    num_neighbors=model_args.nat_num_neighbors,
                    tome_r=model_args.tome_r,
                    layer_scale_init_value=1e-6,
                    coord_dim=3,
                    omega_0=30,
                    last_n_features=model_args.last_n_features,
                    full_attention=model_args.full_attention,
                    drop_rate=model_args.drop,
                    progressive_dims=model_args.progressive_dims,
                    initial_tokens=model_args.initial_tokens,
                    dim_scale_factor=model_args.dim_scale_factor,
                )

                # Linear layer to map encoder output to decoder input
                self.feature_mapper = nn.Linear(
                    self.brain_encoder.blocks.final_dim, model_args.decoder_hidden_dim
                )

            self.brain_decoder = BrainDecoder(
                h=model_args.decoder_hidden_dim,
                out_dim=model_args.clip_emb_dim,
                seq_len=model_args.clip_seq_dim,
                n_blocks=model_args.n_blocks_decoder,
                num_heads=model_args.num_heads,
                drop=model_args.drop,
                blurry_recon=args.train.blurry_recon,
                clip_scale=args.train.clip_scale,
            )

    def forward(self, x, coords):
        if self.decoder_type == "perceiver":
            # For Perceiver, directly process the brain signal
            # Reshape x if needed (assuming x is [batch_size, 1, num_voxels])
            x = x.squeeze(1)  # Remove channel dimension if present
            backbone, clip_voxels, blurry_image_enc = self.brain_decoder(x)
        else:
            # For Q-former, use encoder + decoder
            if self.encoder_type == 'linformer':
                x = x.squeeze(1)
                x = self.brain_encoder(x)
            else:
                x = self.brain_encoder(x, coords)

            x = self.feature_mapper(x)
            backbone, clip_voxels, blurry_image_enc = self.brain_decoder(x)

        # Add shape assertions for debugging
        batch_size = x.shape[0]
        assert (
            backbone.shape[0] == batch_size
        ), f"Expected backbone batch size {batch_size}, got {backbone.shape[0]}"
        assert (
            clip_voxels.shape[0] == batch_size
        ), f"Expected clip_voxels batch size {batch_size}, got {clip_voxels.shape[0]}"

        return backbone, clip_voxels, blurry_image_enc

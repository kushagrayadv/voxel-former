import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import wraps

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        h=1280,                  # Hidden dimension
        out_dim=768,            # Output dimension
        num_latents=256,        # Number of latent vectors
        n_blocks=4,             # Number of cross-attention blocks
        num_heads=8,            # Number of attention heads
        head_dim=64,           # Dimension per head
        drop=0.15,
        clip_scale=1,
        self_per_cross_attn=1,  # Number of self-attention layers per cross-attention
        input_dim=1            # Input dimension (1 for brain signal)
    ):
        super().__init__()
        self.clip_scale = clip_scale
        
        # Initialize learnable latent vectors
        self.latents = nn.Parameter(torch.randn(1, num_latents, h))
        
        # Input projection to map brain signal to hidden dimension
        self.input_proj = nn.Linear(input_dim, h)
        
        # Create layers
        self.layers = nn.ModuleList([])
        for _ in range(n_blocks):
            # Cross attention block
            cross_attn = PreNorm(h, Attention(
                query_dim=h,
                context_dim=h,  # Context is now projected brain signal
                heads=num_heads,
                dim_head=head_dim,
                dropout=drop
            ), context_dim=h)
            
            cross_ff = PreNorm(h, FeedForward(h, dropout=drop))
            
            # Self attention blocks
            self_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    PreNorm(h, Attention(h, heads=num_heads, dim_head=head_dim, dropout=drop)),
                    PreNorm(h, FeedForward(h, dropout=drop))
                ]))
            
            self.layers.append(nn.ModuleList([
                cross_attn,
                cross_ff,
                self_attns
            ]))
        
        # Output projections
        self.backbone_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, out_dim)
        )
        
        if clip_scale > 0:
            self.clip_head = nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, out_dim)
            )
        else:
            self.clip_head = None

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project input brain signal to hidden dimension
        x = self.input_proj(x.unsqueeze(-1))  # Add feature dimension if needed
        
        # Expand latents to batch size
        latents = repeat(self.latents, '1 n d -> b n d', b=batch_size)
        
        # Process through layers
        for cross_attn, cross_ff, self_attns in self.layers:
            # Cross attention
            latents = cross_attn(latents, context=x) + latents
            latents = cross_ff(latents) + latents
            
            # Self attention blocks
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents
        
        # Project to output dimensions
        backbone = self.backbone_head(latents)
        
        if self.clip_head is not None:
            clip_output = self.clip_head(latents)
        else:
            clip_output = latents
            
        # Placeholder for blurry reconstruction
        b = torch.zeros((batch_size, 2, 1), device=x.device)
        
        return backbone, clip_output, b 
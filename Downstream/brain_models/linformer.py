import math
import torch
from torch import nn
from einops import repeat, rearrange

def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()

        activation = default(activation, nn.GELU)

        self.glu = glu
        self.layer1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.layer1(x)
            x = self.act(x)
        else:
            x, v = self.layer1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.layer2(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim, seq_len, num_heads=8):

        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        # Learnable queries to project num_voxels dimension to seq_len
        self.queries = nn.Parameter(torch.randn(1, seq_len, dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]

        queries = repeat(self.queries, "1 n d -> b n d", b=batch_size)

        pooled_output, _ = self.attention(queries, x, x)

        return pooled_output


class SirenPositionalEmbedding(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.in_features = in_features
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
    
    def forward(self, coords):
        x = self.linear(coords)
        x = torch.sin(self.omega_0 * x)
        return x


class LinformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
    ):
        super().__init__()
        assert (dim % heads) == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert (
            kv_len <= self.seq_len
        ), f"the sequence length of the key / values must be {self.seq_len} - {kv_len} given"

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum("b n d, n k->b k d", *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections

        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = (
            lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        )
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (d_h**-0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out), keys


class TokenMerging(nn.Module):
    def __init__(self, r=0, trace_source=False, prop_attn=True):
        super().__init__()
        self.r = r
        self.trace_source = trace_source
        self.prop_attn = prop_attn
        self._tome_info = {
            "r": self.r,
            "size": None,
            "source": None,
            "trace_source": self.trace_source,
            "prop_attn": self.prop_attn,
            "class_token": False,  # Set this based on your specific model's requirements
            "distill_token": False,  # Set this based on your specific model's requirements
        }

    def forward(self, x: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        # Reset tome_info for each batch, if needed
        self._tome_info["source"] = None
        self._tome_info["size"] = None

        r = self._tome_info["r"]
        
        if r > 0:
            try:
                from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
                
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                if self.trace_source:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                try:
                    x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
                except Exception as e:
                    print(f"Error in merge_wavg: {str(e)}")
                    import pdb; pdb.set_trace()
            except ImportError:
                print("ToMe requires installation: pip install tome-torch")
                pass
        return x


class Linformer(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        depth,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
        input_dim=1,
        tome_r=0,
        coord_dim=3,
        omega_0=30
    ):
        super().__init__()
        self.attn_pooling = AttentionPooling(dim=dim, seq_len=seq_len, num_heads=heads)
        self.input_proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([])
        
        # Add positional embedding
        self.pos_embed = SirenPositionalEmbedding(in_features=coord_dim, out_features=dim, omega_0=omega_0)
        
        # Add token merging
        self.token_merging = TokenMerging(r=tome_r)
        
        for _ in range(depth):
            attn = LinformerSelfAttention(
                dim,
                seq_len,
                k=k,
                heads=heads,
                dim_head=dim_head,
                one_kv_head=one_kv_head,
                share_kv=share_kv,
                dropout=dropout,
            )
            ff = FeedForward(dim, dropout=dropout)

            self.layers.append(nn.ModuleList([PreNorm(dim, attn), PreNorm(dim, ff)]))

    def forward(self, x, coords=None):
        x = self.input_proj(x.unsqueeze(-1))
        
        # Add positional embeddings if coordinates are provided
        if coords is not None:
            pos_embeds = self.pos_embed(coords)
            x = x + pos_embeds
            
        x = self.attn_pooling(x)
        
        for self_attn, ff in self.layers:
            attn_out, metric = self_attn(x)
            x = attn_out + x
            
            # Apply token merging
            x = self.token_merging(x, metric)
            
            x = ff(x) + x

        return x

import math
import torch
from torch import nn
from einops import repeat

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
        return self.to_out(out)


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
    ):
        super().__init__()
        self.attn_pooling = AttentionPooling(dim=dim, seq_len=seq_len, num_heads=heads)
        self.input_proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([])
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

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        x = self.attn_pooling(x)
        for self_attn, ff in self.layers:
            x = self_attn(x) + x
            x = ff(x) + x

        return x

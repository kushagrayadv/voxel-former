import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from functools import wraps
import math


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# Logpsaced fourier encoding
def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype = x.device, x.dtype

    scales = torch.logspace(
        start=0.0,
        end=math.log(max_freq / 2, base),
        steps=num_bands,
        base=base,
        device=device,
        dtype=dtype,
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    return x


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


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        """
        This version uses nn.MultiheadAttention internally.  We project
        x -> Q, context -> K,V, then hand those to MHA and finally map
        back to query_dim.  This way, PyTorch handles the actual
        attention ops, including scaling, dropout, etc.
        """

        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else query_dim

        # Linear maps from input to Q/K/V of dimension inner_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        # Final linear map back to query_dim
        self.to_out = nn.Linear(self.inner_dim, query_dim)

    def forward(self, x, context=None):
        if context is None:
            context = x

        B, T, _ = x.shape
        _, S, _ = context.shape

        # Project input to Q, K, V
        q = (
            self.to_q(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        )  # (B, heads, T, dim_head)
        k = (
            self.to_k(context).view(B, S, self.heads, self.dim_head).transpose(1, 2)
        )  # (B, heads, S, dim_head)
        v = (
            self.to_v(context).view(B, S, self.heads, self.dim_head).transpose(1, 2)
        )  # (B, heads, S, dim_head)

        # FlashAttention-2 scaled dot-product attention (set `enable_math` and `enable_mem_efficient` to True for GPUs other than H100/A100)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        # Concatenate heads and map back to query_dim
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.to_out(out)


class CoordinateBasedDownsampling(nn.Module):
    def __init__(self, dim, factor=2, method="grid"):
        super().__init__()
        self.factor = factor
        self.method = method
        self.projection = nn.Linear(dim, dim)  # Feature projection after downsampling

    def forward(self, x, coords):
        batch_size, num_tokens, feature_dim = x.shape
        device = x.device

        # Pre-allocate result tensors
        result_x = []
        result_coords = []
        all_downsample_sizes = []

        # Process each batch item - this loop is difficult to vectorize due to variable output sizes
        for b in range(batch_size):
            batch_x = x[b]
            batch_coords = coords[b]

            if self.method == "grid":
                # Grid-based downsampling
                # 1. Determine coordinate range
                min_coords = batch_coords.min(dim=0)[0]
                max_coords = batch_coords.max(dim=0)[0]
                coord_range = max_coords - min_coords + 1e-6  # Avoid division by zero

                # 2. Calculate grid cells more efficiently
                target_cells = num_tokens / self.factor
                cells_per_dim = max(
                    2, int(target_cells ** (1 / 3))
                )  # Cube root for 3D, minimum 2

                # 3. Compute grid indices using vectorized operations
                normalized_coords = (batch_coords - min_coords) / coord_range
                grid_indices = (
                    (normalized_coords * cells_per_dim)
                    .long()
                    .clamp(0, cells_per_dim - 1)
                )

                # 4. Compute flattened indices more efficiently
                # Use stride multipliers for each dimension
                strides = torch.tensor(
                    [cells_per_dim * cells_per_dim, cells_per_dim, 1], device=device
                )
                flat_indices = (grid_indices * strides).sum(dim=1)

                # 5. Find unique cells and inverse mapping - this is still needed
                unique_cells, inverse, counts = torch.unique(
                    flat_indices, return_inverse=True, return_counts=True
                )
                num_unique = len(unique_cells)

                # 6. Use a more efficient approach to aggregate features for each cell
                # Create a sparse matrix representation for scattering
                indices = torch.stack(
                    [inverse, torch.arange(num_tokens, device=device)]
                )
                values = torch.ones(num_tokens, device=device)
                scatter_matrix = torch.sparse_coo_tensor(
                    indices, values, (num_unique, num_tokens)
                )

                # Compute downsampled features using sparse matrix multiplication
                normalized_matrix = scatter_matrix.to_dense()
                normalized_matrix = normalized_matrix / counts.unsqueeze(1)
                downsampled_x = torch.mm(normalized_matrix, batch_x)

                # 7. Get coordinates more efficiently - use the first point in each cell
                # Create a mapping from cells to the first encountered point index
                mapping = -torch.ones(num_unique, dtype=torch.long, device=device)
                arange_idx = torch.arange(len(inverse), device=device)

                # Use mask to only consider the first occurrence
                for i in range(num_unique):
                    mask = inverse == i
                    if mask.any():
                        first_idx = arange_idx[mask][0]
                        mapping[i] = first_idx

                downsampled_coords = batch_coords[mapping]

            result_x.append(downsampled_x)
            result_coords.append(downsampled_coords)
            all_downsample_sizes.append(downsampled_x.shape[0])

        # Pad sequences to same length more efficiently
        max_len = max(all_downsample_sizes)
        padded_x = torch.zeros((batch_size, max_len, feature_dim), device=device)
        padded_coords = torch.zeros(
            (batch_size, max_len, coords.shape[-1]), device=device
        )
        masks = torch.zeros((batch_size, max_len, 1), device=device)

        # Use vectorized operations for padding
        for b, (b_x, b_coords, size) in enumerate(
            zip(result_x, result_coords, all_downsample_sizes)
        ):
            padded_x[b, :size] = b_x
            padded_coords[b, :size] = b_coords
            masks[b, :size] = 1.0

        # Apply feature projection
        padded_x = self.projection(padded_x)

        # Apply mask to ensure padded tokens don't contribute
        padded_x = padded_x * masks

        return padded_x, padded_coords, masks


class HierarchicalPerceiverDecoder(nn.Module):
    def __init__(
        self,
        h=1280,  # Hidden dimension
        out_dim=768,  # Output dimension
        num_latents=256,  # Number of latent vectors
        n_blocks=4,  # Number of cross-attention blocks
        num_heads=8,  # Number of attention heads
        head_dim=64,  # Dimension per head
        drop=0.15,
        clip_scale=1,
        self_per_cross_attn=1,  # Number of self-attention layers per cross-attention
        input_dim=1,  # Input dimension (1 for brain signal)
        coord_dim=3,  # Dimension of coordinate input
        omega_0=30,  # Frequency for SIREN
        downsample_factors=[2, 2, 2, 2],  # Downsampling factors for each level
        use_residual=True,  # Whether to use U-Net style residual connections
        downsample_method="grid",  # 'grid', 'fps', or 'knn'
        max_freq=10,
        num_freq_bands=6,
        use_siren_embed=True,
        use_avg_pool=False,
    ):
        super().__init__()
        self.clip_scale = clip_scale
        self.n_blocks = n_blocks
        self.use_residual = use_residual
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.use_siren_embed = use_siren_embed
        self.use_avg_pool = use_avg_pool

        # Ensure downsample_factors has right length
        self.downsample_factors = downsample_factors[:n_blocks]
        while len(self.downsample_factors) < n_blocks:
            self.downsample_factors.append(2)

        # Initialize learnable latent vectors
        self.latents = nn.Parameter(torch.randn(1, num_latents, h))

        # Input projection to map brain signal to hidden dimension
        self.input_proj = nn.Linear(input_dim, h)

        if use_siren_embed:
            self.pos_embed = SirenPositionalEmbedding(
                in_features=coord_dim, out_features=h, omega_0=omega_0
            )
            self.pos_emb_proj = None
        else:
            self.fourier_encode = fourier_encode
            pos_emb_dim = coord_dim * num_freq_bands * 2
            self.pos_emb_proj = nn.Linear(pos_emb_dim, h)

        # Create downsampling layers
        self.downsample_layers = nn.ModuleList(
            [
                CoordinateBasedDownsampling(h, factor=factor, method=downsample_method)
                for factor in self.downsample_factors
            ]
        )

        # Create upsampling layers for residual connections (if used)
        if use_residual:
            self.upsample_layers = nn.ModuleList(
                [
                    nn.Linear(h, h)  # Feature transformation for upsampling path
                    for _ in range(n_blocks - 1)
                ]
            )

        # Create attention blocks
        self.encoder_blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            # Cross attention block
            cross_attn = PreNorm(
                h,
                Attention(
                    query_dim=h,
                    context_dim=h,
                    heads=num_heads,
                    dim_head=head_dim,
                    dropout=drop,
                ),
                context_dim=h,
            )

            cross_ff = PreNorm(h, FeedForward(h, dropout=drop))

            # Self attention blocks
            self_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                self_attns.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                h,
                                Attention(
                                    h, heads=num_heads, dim_head=head_dim, dropout=drop
                                ),
                            ),
                            PreNorm(h, FeedForward(h, dropout=drop)),
                        ]
                    )
                )

            self.encoder_blocks.append(
                nn.ModuleList([cross_attn, cross_ff, self_attns])
            )

        # Create decoder blocks for residual connection path
        if use_residual:
            self.decoder_blocks = nn.ModuleList([])
            for i in range(n_blocks - 1):
                # Cross attention for merging residual connection
                merge_attn = PreNorm(
                    h,
                    Attention(
                        query_dim=h,
                        context_dim=h,
                        heads=num_heads,
                        dim_head=head_dim,
                        dropout=drop,
                    ),
                    context_dim=h,
                )

                merge_ff = PreNorm(h, FeedForward(h, dropout=drop))

                self.decoder_blocks.append(nn.ModuleList([merge_attn, merge_ff]))
        
        if use_avg_pool:
            self.avg_pool = Reduce('b n d -> b d', 'mean')

        # Output projections
        self.backbone_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, out_dim))

        if clip_scale > 0:
            self.clip_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, out_dim))
        else:
            self.clip_head = None

    def forward(self, x, coords=None):
        batch_size = x.shape[0]

        # Project input brain signal to hidden dimension
        x = self.input_proj(x.unsqueeze(-1))  # Add feature dimension if needed

        # Apply positional embedding if coordinates are provided
        if coords is not None:
            if self.use_siren_embed:
                x = x + self.pos_embed(coords)
            else:
                pos_embeds = self.fourier_encode(
                    coords, self.max_freq, self.num_freq_bands
                )
                pos_embeds = rearrange(pos_embeds, "... n d -> ... (n d)")
                if self.pos_emb_proj is not None:
                    pos_embeds = self.pos_emb_proj(pos_embeds)
                    x = x + pos_embeds
                else:
                    raise ValueError("No position embedding projection provided")

        # Create hierarchical inputs
        contexts = [x]
        coord_hierarchy = [coords]
        masks = [torch.ones(batch_size, x.shape[1], 1, device=x.device)]

        # Apply downsampling to create hierarchical inputs
        for i, downsample in enumerate(self.downsample_layers):
            down_x, down_coords, down_mask = downsample(
                contexts[-1], coord_hierarchy[-1]
            )
            contexts.append(down_x)
            coord_hierarchy.append(down_coords)
            masks.append(down_mask)

        # Expand latents to batch size
        latents = repeat(self.latents, "1 n d -> b n d", b=batch_size)
        # Process through encoder path (downsampling)
        encoder_features = []

        for i, (cross_attn, cross_ff, self_attns) in enumerate(self.encoder_blocks):
            # Use appropriate context level
            context_idx = min(i, len(contexts) - 1)
            context = contexts[context_idx]

            # Store latents for residual connection if needed
            if self.use_residual and i < self.n_blocks - 1:
                encoder_features.append(latents)

            # Cross attention
            latents = cross_attn(latents, context=context) + latents
            latents = cross_ff(latents) + latents

            # Self attention blocks
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents

        # Process through decoder path (upsampling with residual connections)
        if self.use_residual:
            for i, (merge_attn, merge_ff) in enumerate(self.decoder_blocks):
                # Get features from encoder path (in reverse order)
                skip_connection = encoder_features[-(i + 1)]

                # Transform features for upsampling path
                upsampled = self.upsample_layers[i](latents)

                # Merge with skip connection via cross-attention
                latents = merge_attn(upsampled, context=skip_connection) + upsampled
                latents = merge_ff(latents) + latents

        if self.use_avg_pool:
            latents = self.avg_pool(latents)

        # Project to output dimensions
        backbone = self.backbone_head(latents)

        if self.clip_head is not None:
            clip_output = self.clip_head(latents)
        else:
            clip_output = latents

        # Placeholder for blurry reconstruction
        b = torch.zeros((batch_size, 2, 1), device=x.device)

        return backbone, clip_output, b


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        h=1280,  # Hidden dimension
        out_dim=768,  # Output dimension
        num_latents=256,  # Number of latent vectors
        n_blocks=4,  # Number of cross-attention blocks
        num_heads=8,  # Number of attention heads
        head_dim=64,  # Dimension per head
        drop=0.15,
        clip_scale=1,
        self_per_cross_attn=1,  # Number of self-attention layers per cross-attention
        input_dim=1,  # Input dimension (1 for brain signal)
        coord_dim=3,
        omega_0=30,
        max_freq=10,
        num_freq_bands=6,
        use_siren_embed=False,
        use_avg_pool=False,
        mlp_clip_head=False
    ):
        super().__init__()

        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.clip_scale = clip_scale
        self.use_siren_embed = use_siren_embed
        self.use_avg_pool = use_avg_pool

        if use_siren_embed:
            context_dim = h

            self.pos_embed = SirenPositionalEmbedding(
                in_features=coord_dim, out_features=h, omega_0=omega_0
            )
        else:
            # Hidden dimension will be changed after concatenating with position embedding
            context_dim = h + coord_dim * num_freq_bands * 2

            # self.siren_embed = SirenPositionalEmbedding(in_features=coord_dim, out_features=h, omega_0=omega_0)
            self.fourier_encode = fourier_encode

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
                context_dim=context_dim,  # Context is now projected brain signal
                heads=num_heads,
                dim_head=head_dim,
                dropout=drop
            ), context_dim=context_dim)
            
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
        
        if use_avg_pool:
            self.avg_pool = Reduce('b n d -> b d', 'mean')

        # Output projections
        self.backbone_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, out_dim))

        if clip_scale > 0:
            if mlp_clip_head:
                self.clip_head = nn.Sequential(
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, out_dim),
                )
            else:
                self.clip_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, out_dim))
        else:
            self.clip_head = None

    def forward(self, x, coords=None):
        batch_size = x.shape[0]

        # Project input brain signal to hidden dimension
        x = self.input_proj(x.unsqueeze(-1))  # Add feature dimension if needed

        # Add position embedding if voxel coordinates are provided
        if coords is not None:
            if self.use_siren_embed:
                x = x + self.pos_embed(coords)
            else:
                pos_embeds = self.fourier_encode(
                    coords, self.max_freq, self.num_freq_bands
                )
                pos_embeds = rearrange(pos_embeds, "... n d -> ... (n d)")
                x = torch.cat((x, pos_embeds), dim=-1)

        # Expand latents to batch size
        latents = repeat(self.latents, "1 n d -> b n d", b=batch_size)

        # Process through layers
        for cross_attn, cross_ff, self_attns in self.layers:
            # Cross attention
            latents = cross_attn(latents, context=x) + latents
            latents = cross_ff(latents) + latents

            # Self attention blocks
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents

                latents = self_ff(latents) + latents

        if self.use_avg_pool:
            latents = self.avg_pool(latents)

        # Project to output dimensions
        backbone = self.backbone_head(latents)

        if self.clip_head is not None:
            clip_output = self.clip_head(latents)
        else:
            clip_output = latents

        # Placeholder for blurry reconstruction
        b = torch.zeros((batch_size, 2, 1), device=x.device)

        return backbone, clip_output, b


class VariablePerceiverDecoder(nn.Module):
    def __init__(
        self,
        h_dims=[128, 256, 512, 1024],  # Hidden dimensions for intemediate latents
        out_dim=768,  # Output dimension
        num_latents=256,  # Number of latent vectors
        n_blocks=4,  # Number of cross-attention blocks
        num_heads=8,  # Number of attention heads
        head_dim=64,  # Dimension per head
        drop=0.15,
        clip_scale=1,
        self_per_cross_attn=1,  # Number of self-attention layers per cross-attention
        input_dim=1,  # Input dimension (1 for brain signal)
        coord_dim=3,
        omega_0=30,
        max_freq=10,
        num_freq_bands=6,
        use_siren_embed=False,
        use_avg_pool=False,
        mlp_clip_head=False
    ):
        super().__init__()

        assert len(h_dims) == n_blocks, "Hidden latent dimension for each block mismatch"

        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.clip_scale = clip_scale
        self.use_siren_embed = use_siren_embed
        self.use_avg_pool=use_avg_pool

        self.fourier_encode = fourier_encode

        # Initialize learnable latent vectors for each block
        self.latents = nn.ParameterList([])

        # Create layers
        self.layers = nn.ModuleList([])
        
        for i, h in enumerate(h_dims):

            # Learnable Position Embedding
            if use_siren_embed:
                context_dim = h
                pos_embed = SirenPositionalEmbedding(
                                    in_features=coord_dim, out_features=h, omega_0=omega_0
                                )
            else:   # Fourier embeddings
                # Hidden dimension will be changed after concatenating with position embedding
                context_dim = h + coord_dim * num_freq_bands * 2
                pos_embed = None

            # Initialise latents
            latent = nn.Parameter(torch.randn(1, num_latents, h))
        
            # Input projection to map brain signal to hidden dimension
            input_proj = nn.Linear(input_dim, h)

            # Cross attention block
            cross_attn = PreNorm(h, Attention(
                query_dim=h,
                context_dim=context_dim,  # Context is now projected brain signal
                heads=num_heads,
                dim_head=head_dim,
                dropout=drop
            ), context_dim=context_dim)
            
            cross_ff = PreNorm(h, FeedForward(h, dropout=drop))

            # Self attention blocks
            self_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    PreNorm(h, Attention(h, heads=num_heads, dim_head=head_dim, dropout=drop)),
                    PreNorm(h, FeedForward(h, dropout=drop))
                ]))

            # Upscale embedding dimension except last block
            upscale_latents = None
            if not h == h_dims[-1]:
                block_out_dim = h_dims[i+1]
                upscale_latents = PreNorm(h, nn.Linear(h, block_out_dim))

            self.latents.append(latent)
                
            self.layers.append(nn.ModuleList([
                pos_embed,
                input_proj,
                cross_attn,
                cross_ff,
                self_attns,
                upscale_latents
            ]))
        
        if use_avg_pool:
            self.avg_pool = Reduce('b n d -> b d', 'mean')

        # Output projections
        self.backbone_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, out_dim))

        if clip_scale > 0:
            if mlp_clip_head:
                self.clip_head = nn.Sequential(
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, out_dim),
                )
            else:
                self.clip_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, out_dim))
        else:
            self.clip_head = None

    def forward(self, x, coords=None):
        batch_size = x.shape[0]
        orig_x = x

        prev_latents = None
        for (pos_embed, input_proj, cross_attn, cross_ff, self_attns, upscale_latents), latents in zip(self.layers, self.latents):
            
            # Expand latents to batch size
            latents = repeat(latents, "1 n d -> b n d", b=batch_size)

            if prev_latents is not None:
                latents = latents + prev_latents

            # Project input brain signal to hidden dimension
            x = input_proj(orig_x.unsqueeze(-1))  # Add feature dimension if needed

            # Add position embedding if voxel coordinates are provided
            if coords is not None:
                if self.use_siren_embed and pos_embed is not None:
                    x = x + pos_embed(coords)
                else:
                    pos_embed = self.fourier_encode(
                        coords, self.max_freq, self.num_freq_bands
                    )
                    pos_embed = rearrange(pos_embed, "... n d -> ... (n d)")
                    x = torch.cat((x, pos_embed), dim=-1)


            # Cross attention
            latents = cross_attn(latents, context=x) + latents
            
            # Feed forward
            latents = cross_ff(latents) + latents

            # Self attention blocks
            for self_attn, self_ff in self_attns:
                latents = self_attn(latents) + latents

                latents = self_ff(latents) + latents

            # Upscale
            if upscale_latents:
                latents = upscale_latents(latents)

            prev_latents = latents

        if self.use_avg_pool:
            latents = self.avg_pool(latents)

        # Project to output dimensions
        backbone = self.backbone_head(latents)

        if self.clip_head is not None:
            clip_output = self.clip_head(latents)
        else:
            clip_output = latents

        # Placeholder for blurry reconstruction
        b = torch.zeros((batch_size, 2, 1), device=x.device)

        return backbone, clip_output, b

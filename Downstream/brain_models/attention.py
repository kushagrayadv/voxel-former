import torch
import torch.nn as nn
from functools import lru_cache
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class BaseAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.dtype = torch.bfloat16
        self.scale = self.head_dim ** -0.5

        # Initialize projection layers
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def _project_qkv(self, x):
        batch_size, seq_len, _ = x.shape

        # Project inputs to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape and transpose to get shapes: (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        return query, key, value

    def _reshape_output(self, output, batch_size, seq_len):
        # Reshape output to (batch_size, seq_len, feature_dim)
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)

class FullAttention(BaseAttention):
    def __init__(self, feature_dim, num_heads):
        super().__init__(feature_dim, num_heads)

    def forward(self, x, coords=None, **kwargs):
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
            batch_size, seq_len, _ = x.shape
            query, key, value = self._project_qkv(x)

            # Compute attention using scaled dot product attention
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, scale=self.scale
            )

            # Compute metric for visualization
            metric = key.mean(1)  # Averaging over heads

            # Reshape output
            output = self._reshape_output(output, batch_size, seq_len)

            return output, metric

class NearestNeighborAttention(BaseAttention):
    def __init__(self, feature_dim, num_heads, num_neighbors):
        super().__init__(feature_dim, num_heads)
        self.num_neighbors = num_neighbors

    @lru_cache
    def _compute_nearest_neighbors(self, voxel_coords):
        batch_size, seq_len, coord_dim = voxel_coords.shape
        device = voxel_coords.device
        x = voxel_coords  # (batch_size, seq_len, coord_dim)
        # Compute squared norms
        x_norm = (x ** 2).sum(-1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        # Compute pairwise squared distances
        dists_sq = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        # Exclude self by setting diagonal to large value
        diag_inf = torch.full((seq_len,), float('inf'), device=device)
        dists_sq += torch.diag(diag_inf).unsqueeze(0)
        # Get top k indices
        _, top_k_indices = torch.topk(dists_sq, k=self.num_neighbors, dim=-1, largest=False)  # (batch_size, seq_len, num_neighbors)
        return top_k_indices

    def forward(self, x, coords, **kwargs):
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
            batch_size, seq_len, _ = x.shape
            query, key, value = self._project_qkv(x)

            # Compute nearest neighbors
            top_k_indices = self._compute_nearest_neighbors(coords)
            top_k_indices = top_k_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Create batch and head indices for advanced indexing
            batch_indices = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1, 1).expand_as(top_k_indices)
            head_indices = torch.arange(self.num_heads, device=x.device).view(1, self.num_heads, 1, 1).expand_as(top_k_indices)

            # Gather key and value vectors for nearest neighbors
            key_neighbors = key[batch_indices, head_indices, top_k_indices, :]
            value_neighbors = value[batch_indices, head_indices, top_k_indices, :]

            # Compute attention scores and probabilities
            attn_scores = torch.einsum('bnhd,bnhkd->bnhk', query, key_neighbors) * self.scale
            attn_probs = torch.softmax(attn_scores, dim=-1)

            # Compute attention output
            output = torch.einsum('bnhk,bnhkd->bnhd', attn_probs, value_neighbors)

            # Compute metric for visualization
            metric = key.mean(1)

            # Reshape output
            output = self._reshape_output(output, batch_size, seq_len)

            return output, metric

# Test the NearestNeighborAttention module
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    feature_dim = 8
    num_heads = 2
    num_neighbors = 1
    coord_dim = 3
    
    # Create test inputs
    x = torch.randn(batch_size, seq_len, feature_dim)
    coords = torch.randn(batch_size, seq_len, coord_dim)
    
    # Test both full attention and nearest neighbor attention
    for full_attention in [True, False]:
        print(f"\nTesting with full_attention={full_attention}")
        
        # Initialize model
        model = NearestNeighborAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_neighbors=num_neighbors
        )
        
        # Forward pass
        output, metric = model(x, coords)
        
        # Print shapes
        print(f"Input shape: {x.shape}")
        print(f"Coordinates shape: {coords.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Metric shape: {metric.shape}")
        
        # Basic checks
        assert output.shape == (batch_size, seq_len, feature_dim), "Output shape mismatch"
        assert metric.shape == (batch_size, seq_len, feature_dim // num_heads), "Metric shape mismatch"
        
        print("All shape checks passed!")
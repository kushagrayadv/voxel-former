import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from atten_flex_customize import NearestNeighborAttention
from tome_customize import TokenMerging

class ConvTokenizer1D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1, stride=1, padding=0),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)  # Divide by 2 because of chunking
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def swiglu(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

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

class BrainNATLayer(nn.Module):
    def __init__(self, dim, num_heads, num_neighbors, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, tome_r=0, layer_scale_init_value=1e-6, use_coords=True,
                 last_n_features=16, full_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = NearestNeighborAttention(dim, num_heads, num_neighbors, full_attention=full_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.token_merging = TokenMerging(r=tome_r)
        mlp_hidden_dim = int(dim * mlp_ratio * 2)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        
        self.use_coords = use_coords
        self.last_n_features = last_n_features

    def forward(self, x, coords):
        if self.use_coords:
            x_attn, metric = self.attn(self.norm1(x), coords)
        else:
            last_n_features = x[:, :, -self.last_n_features:]
            x_attn, metric = self.attn(self.norm1(x), last_n_features)
        
        x = x + self.drop_path(self.gamma_1 * x_attn)
        x = self.token_merging(x, metric)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class BrainNATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, num_neighbors, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, tome_r=0, layer_scale_init_value=1e-6,
                 last_n_features=16, full_attention=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            BrainNATLayer(
                dim=dim, num_heads=num_heads, num_neighbors=num_neighbors,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, tome_r=tome_r, layer_scale_init_value=layer_scale_init_value,
                use_coords=(i == 0), last_n_features=last_n_features, full_attention=full_attention)
            for i in range(depth)])

    def forward(self, x, coords):
        for blk in self.blocks:
            x = blk(x, coords)
        return x

class BrainNAT(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, depth=4, num_heads=8, num_neighbors=5,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, tome_r=0, layer_scale_init_value=1e-6, coord_dim=3, 
                 omega_0=30, last_n_features=16,full_attention=False):
        super().__init__()
        self.embed_dim = 2 ** int(torch.log2(torch.tensor(embed_dim)).ceil().item())
        self.pos_embed_dim = embed_dim
        self.total_embed_dim = self.embed_dim 
        self.num_features = self.embed_dim
        self.embed_layer = ConvTokenizer1D(in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.pos_embed = SirenPositionalEmbedding(in_features=coord_dim, out_features=self.pos_embed_dim, omega_0=omega_0)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = BrainNATBlock(
            dim=self.total_embed_dim,
            depth=depth, num_heads=num_heads, num_neighbors=num_neighbors,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, tome_r=tome_r,
            layer_scale_init_value=layer_scale_init_value, last_n_features=last_n_features, full_attention=full_attention)
        self.norm = norm_layer(self.total_embed_dim)
        self.head = nn.Linear(self.total_embed_dim, self.embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, coords):
        x = self.embed_layer(x)  

        x = self.pos_drop(x)
        
        pos_embeds = self.pos_embed(coords)
        x = x + pos_embeds
        
        x = self.blocks(x, coords)
        x = self.norm(x)
        return x

    def forward(self, x, coords):
        x = self.forward_features(x, coords)
        x = self.head(x)
        return x

# Example usage
if __name__ == "__main__":
    import numpy as np
    import nibabel as nib
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
        return trainable

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loading fMRI scans and masks for two subjects
    subj = 1
    fmri_scan_1 = nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj01/func1pt8mm/valid_session01.nii.gz')
    fmri_scan_2 = nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj02/func1pt8mm/valid_session01.nii.gz')
    
    nsdgeneral_roi_mask_1 = nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata() == 1
    nsdgeneral_roi_mask_2 = nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj02/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata() == 1

    # Extracting voxel data for both scans
    input_scan_1 = fmri_scan_1.get_fdata()[nsdgeneral_roi_mask_1]
    input_scan_2 = fmri_scan_2.get_fdata()[nsdgeneral_roi_mask_2]

    # Create visual cortex mask and coordinates tensor
    visual_cortex_mask_1 = torch.tensor(nsdgeneral_roi_mask_1, dtype=torch.bool, device=device).cpu()
    coords_1 = torch.nonzero(visual_cortex_mask_1, as_tuple=False).float().numpy()
    
    visual_cortex_mask_2 = torch.tensor(nsdgeneral_roi_mask_2, dtype=torch.bool, device=device).cpu()
    coords_2 = torch.nonzero(visual_cortex_mask_2, as_tuple=False).float().numpy()

    # Set up batch size and coordinate dimension
    coord_dim = coords_1.shape[-1]

    # Initialize the model with adjusted dimensions
    embed_dim = 64
    # pos_embed_dim = 128  # Ensure that total_embed_dim is divisible by num_heads
    num_heads = 8  # Ensure total_embed_dim % num_heads == 0

    model = BrainNAT(
        in_chans=1,
        embed_dim=embed_dim,
        # pos_embed_dim=pos_embed_dim,
        depth=4,
        num_heads=num_heads,
        num_neighbors=8,
        tome_r=1000,
        layer_scale_init_value=1e-6,
        coord_dim=coord_dim,
        omega_0=30,
        last_n_features=16, # Use the last 16 features for neighborhood attention
        full_attention=True
    ).to(device)

    print("Number of parameters:", count_params(model))
    model.train()

    # Create input tensor for the batch
    batch_size = 32
    coords_1 = torch.tensor(coords_1, dtype=torch.float32, device=device).unsqueeze(0)
    coords = torch.repeat_interleave(coords_1, batch_size, dim=0)
    input_scan_1 = torch.tensor(input_scan_1, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
    # Dummy input tensor for batch of two subjects
    x = torch.repeat_interleave(input_scan_1, batch_size, dim=0).unsqueeze(1)
    print("x shape:", x.shape)
    print("coords shape:", coords.shape)
    # Forward pass
    output = model(x, coords)
    print("Output shape:", output.shape)

    # Create a virtual target tensor of the same shape as the output
    y = torch.randn(output.shape, device=device)

    # Compute the loss
    criterion = nn.MSELoss()
    loss = criterion(output, y)
    print("Loss:", loss.item())
    loss.backward()
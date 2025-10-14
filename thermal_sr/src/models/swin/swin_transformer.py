import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""
    
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x: torch.Tensor, window_size: int):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

class PatchEmbedding(nn.Module):
    """Patch embedding layer."""
    
    def __init__(self, patch_size: int = 4, in_channels: int = 1, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, H*W//patch_size^2, embed_dim
        x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size

class SwinThermalSR(nn.Module):
    """Swin Transformer for thermal super-resolution."""
    
    def __init__(self, 
                 scale: int = 2,
                 patch_size: int = 4,
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 7):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding for thermal
        self.thermal_embed = PatchEmbedding(patch_size, 1, embed_dim)
        
        # Optical guidance encoder
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        
        # Swin Transformer layers
        self.layers = nn.ModuleList()
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim * (2 ** i),
                    num_heads=num_head,
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2
                ) for j in range(depth)
            ])
            self.layers.append(layer)
            
            # Patch merging (except last layer)
            if i < len(depths) - 1:
                self.layers.append(
                    nn.Linear(embed_dim * (2 ** i), embed_dim * (2 ** (i + 1)))
                )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * (2 ** (len(depths) - 1)), embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Upsampling layers
        if scale == 2:
            self.upsampler = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(inplace=True)
            )
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif scale == 4:
            self.upsampler = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 16),
                nn.ReLU(inplace=True)
            )
            self.pixel_shuffle = nn.PixelShuffle(4)
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim // (scale ** 2), 1, 3, padding=1)
        
    def forward(self, thermal_lr: torch.Tensor, optical_hr: torch.Tensor):
        B, C, H, W = thermal_lr.shape
        
        # Upsample thermal to match optical resolution
        thermal_up = F.interpolate(thermal_lr, size=optical_hr.shape[-2:], mode='bicubic')
        
        # Extract optical guidance
        optical_guidance = self.optical_encoder(optical_hr)  # B, embed_dim
        
        # Patch embedding
        x, H_patches, W_patches = self.thermal_embed(thermal_up)
        
        # Add optical guidance as global context
        x = x + optical_guidance.unsqueeze(1)
        
        # Swin Transformer layers
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    x = block(x, H_patches, W_patches)
            else:  # Patch merging
                x = layer(x)
                H_patches, W_patches = H_patches // 2, W_patches // 2
        
        # Decoder
        x = self.decoder(x)
        
        # Upsampling
        if self.scale > 1:
            x = self.upsampler(x)
            
            # Reshape for pixel shuffle
            x = x.view(B, H_patches, W_patches, -1).permute(0, 3, 1, 2)
            x = self.pixel_shuffle(x)
        else:
            x = x.view(B, H_patches, W_patches, -1).permute(0, 3, 1, 2)
        
        # Output projection
        output = self.output_proj(x)
        
        # Residual connection
        if self.scale == 1:
            baseline = thermal_up
        else:
            baseline = F.interpolate(thermal_lr, scale_factor=self.scale, mode='bicubic')
        
        return output + baseline
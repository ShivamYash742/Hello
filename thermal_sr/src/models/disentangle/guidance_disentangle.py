import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureTextureDisentangler(nn.Module):
    """Disentangle optical guidance into structure and texture components."""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Structure branch (edges, boundaries)
        self.structure_branch = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Tanh()  # Normalized structure features
        )
        
        # Texture branch (fine details)
        self.texture_branch = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Tanh()  # Normalized texture features
        )
        
    def forward(self, optical: torch.Tensor):
        shared_feat = self.shared_encoder(optical)
        structure = self.structure_branch(shared_feat)
        texture = self.texture_branch(shared_feat)
        return structure, texture

class ContrastiveGate(nn.Module):
    """Contrastive gating to suppress texture leakage."""
    
    def __init__(self, thermal_channels: int, structure_channels: int, texture_channels: int):
        super().__init__()
        
        self.thermal_proj = nn.Conv2d(thermal_channels, 32, 1)
        self.structure_proj = nn.Conv2d(structure_channels, 32, 1)
        self.texture_proj = nn.Conv2d(texture_channels, 32, 1)
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),  # 32*3 = 96
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),  # Structure vs texture weights
            nn.Softmax(dim=1)
        )
        
    def forward(self, thermal_feat: torch.Tensor, structure: torch.Tensor, texture: torch.Tensor):
        # Resize to match thermal features
        structure = F.interpolate(structure, size=thermal_feat.shape[-2:], mode='bilinear')
        texture = F.interpolate(texture, size=thermal_feat.shape[-2:], mode='bilinear')
        
        # Project to common dimension
        thermal_proj = self.thermal_proj(thermal_feat)
        structure_proj = self.structure_proj(structure)
        texture_proj = self.texture_proj(texture)
        
        # Compute gating weights
        combined = torch.cat([thermal_proj, structure_proj, texture_proj], dim=1)
        gates = self.gate_net(combined)
        
        structure_gate = gates[:, 0:1]
        texture_gate = gates[:, 1:2]
        
        # Apply selective gating (favor structure, suppress texture)
        gated_structure = structure_proj * structure_gate
        gated_texture = texture_proj * texture_gate * 0.1  # Suppress texture
        
        return gated_structure + gated_texture

class GuidanceDisentangleModel(nn.Module):
    """Guidance-Disentanglement model for thermal super-resolution."""
    
    def __init__(self, scale: int = 2, num_channels: int = 64):
        super().__init__()
        self.scale = scale
        
        # Structure-texture disentangler
        self.disentangler = StructureTextureDisentangler()
        
        # Thermal feature extractor
        self.thermal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Contrastive fusion gates
        self.fusion_gate1 = ContrastiveGate(num_channels, 16, 16)
        self.fusion_gate2 = ContrastiveGate(num_channels, 16, 16)
        
        # Feature refinement with skip connections
        self.refinement_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels + 32, num_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels, num_channels, 3, padding=1)
            ) for _ in range(3)
        ])
        
        # Upsampling network
        if scale == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        elif scale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels, num_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        
    def forward(self, thermal_lr: torch.Tensor, optical_hr: torch.Tensor):
        # Disentangle optical guidance
        structure, texture = self.disentangler(optical_hr)
        
        # Upsample thermal to match optical resolution
        thermal_up = F.interpolate(thermal_lr, size=optical_hr.shape[-2:], mode='bicubic')
        
        # Extract thermal features
        thermal_feat = self.thermal_encoder(thermal_up)
        
        # Multi-stage fusion with contrastive gating
        fused_feat1 = self.fusion_gate1(thermal_feat, structure, texture)
        fused_feat2 = self.fusion_gate2(thermal_feat, structure, texture)
        
        # Feature refinement with skip connections
        x = thermal_feat
        for i, block in enumerate(self.refinement_blocks):
            if i == 0:
                x_in = torch.cat([x, fused_feat1], dim=1)
            elif i == 1:
                x_in = torch.cat([x, fused_feat2], dim=1)
            else:
                x_in = torch.cat([x, fused_feat1 + fused_feat2], dim=1)
            
            residual = block(x_in)
            x = x + residual
        
        # Upsampling
        if self.scale > 1:
            upsampled = self.upsampler(x)
        else:
            upsampled = x
        
        # Generate output
        output = self.output_head(upsampled)
        
        # Residual connection with bicubic baseline
        if self.scale == 1:
            baseline = thermal_up
        else:
            baseline = F.interpolate(thermal_lr, scale_factor=self.scale, mode='bicubic')
        
        return output + baseline
    
    def compute_disentanglement_loss(self, structure: torch.Tensor, texture: torch.Tensor):
        """Compute disentanglement regularization loss."""
        # Encourage orthogonality between structure and texture
        structure_flat = structure.view(structure.size(0), -1)
        texture_flat = texture.view(texture.size(0), -1)
        
        # Normalize features
        structure_norm = F.normalize(structure_flat, dim=1)
        texture_norm = F.normalize(texture_flat, dim=1)
        
        # Compute cosine similarity (should be close to 0 for orthogonality)
        similarity = torch.sum(structure_norm * texture_norm, dim=1)
        orthogonality_loss = torch.mean(similarity ** 2)
        
        return orthogonality_loss
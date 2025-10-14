import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Attention gate for feature fusion."""
    
    def __init__(self, thermal_channels: int, optical_channels: int):
        super().__init__()
        self.thermal_conv = nn.Conv2d(thermal_channels, thermal_channels//2, 1)
        self.optical_conv = nn.Conv2d(optical_channels, thermal_channels//2, 1)
        self.attention_conv = nn.Conv2d(thermal_channels//2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, thermal_feat: torch.Tensor, optical_feat: torch.Tensor) -> torch.Tensor:
        # Resize optical to match thermal
        optical_feat = F.interpolate(optical_feat, size=thermal_feat.shape[-2:], mode='bilinear')
        
        # Compute attention weights
        thermal_proj = self.thermal_conv(thermal_feat)
        optical_proj = self.optical_conv(optical_feat)
        
        combined = thermal_proj + optical_proj
        attention = self.sigmoid(self.attention_conv(combined))
        
        # Apply attention to thermal features
        return thermal_feat * attention

class GuidanceExtractor(nn.Module):
    """Extract guidance features from optical image."""
    
    def __init__(self):
        super().__init__()
        # Edge detection branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Low-frequency luminance branch
        self.lum_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, optical: torch.Tensor) -> torch.Tensor:
        # Extract luminance
        luminance = torch.mean(optical, dim=1, keepdim=True)
        
        # Edge features
        edge_feat = self.edge_conv(optical)
        
        # Luminance features
        lum_feat = self.lum_conv(luminance)
        
        # Combine guidance features
        guidance = torch.cat([edge_feat, lum_feat], dim=1)
        return guidance

class AlignmentFusionCNN(nn.Module):
    """Alignment-Fusion CNN for thermal super-resolution."""
    
    def __init__(self, scale: int = 2, num_channels: int = 64):
        super().__init__()
        self.scale = scale
        
        # Guidance extractor
        self.guidance_extractor = GuidanceExtractor()
        
        # Thermal feature extractor
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, num_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels//2, num_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layers with attention gates
        self.fusion_layers = nn.ModuleList([
            AttentionGate(num_channels, 64),  # 64 from guidance (32+32)
            AttentionGate(num_channels, 64),
            AttentionGate(num_channels, 64)
        ])
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
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
        
        # Output layer
        self.output_conv = nn.Conv2d(num_channels, 1, 3, padding=1)
        
    def forward(self, thermal_lr: torch.Tensor, optical_hr: torch.Tensor) -> torch.Tensor:
        # Extract guidance from optical
        guidance = self.guidance_extractor(optical_hr)
        
        # Upsample thermal to match optical resolution
        thermal_up = F.interpolate(thermal_lr, size=optical_hr.shape[-2:], mode='bicubic')
        
        # Extract thermal features
        thermal_feat = self.thermal_conv(thermal_up)
        
        # Multi-scale fusion with attention
        for fusion_layer in self.fusion_layers:
            thermal_feat = fusion_layer(thermal_feat, guidance)
        
        # Feature refinement
        refined_feat = self.refinement(thermal_feat)
        
        # Upsampling (if needed for higher scales)
        if self.scale > 1:
            upsampled = self.upsampler(refined_feat)
        else:
            upsampled = refined_feat
        
        # Generate output
        output = self.output_conv(upsampled)
        
        # Residual connection
        if self.scale == 1:
            output = output + thermal_up
        else:
            thermal_bicubic = F.interpolate(thermal_lr, scale_factor=self.scale, mode='bicubic')
            output = output + thermal_bicubic
        
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class SSIMLoss(nn.Module):
    """SSIM loss for structural similarity."""
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        
    def gaussian(self, window_size: int, sigma: float):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
            for x in range(window_size)
        ])
        return gauss/gauss.sum()
    
    def create_window(self, window_size: int, channel: int):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)
        return 1 - self.ssim(img1, img2)

class EdgeAwareLoss(nn.Module):
    """Edge-aware guidance loss tied to optical gradients."""
    
    def __init__(self):
        super().__init__()
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def compute_gradients(self, img: torch.Tensor):
        """Compute image gradients using Sobel operators."""
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return gradient_magnitude
    
    def forward(self, 
                sr_thermal: torch.Tensor, 
                gt_thermal: torch.Tensor,
                optical_hr: torch.Tensor):
        """Compute edge-aware loss using optical guidance."""
        # Compute gradients
        sr_grad = self.compute_gradients(sr_thermal)
        gt_grad = self.compute_gradients(gt_thermal)
        
        # Optical edge map (luminance gradients)
        optical_lum = torch.mean(optical_hr, dim=1, keepdim=True)
        optical_grad = self.compute_gradients(optical_lum)
        
        # Weight thermal gradient loss by optical edges
        edge_weight = torch.sigmoid(optical_grad * 5.0)  # Enhance edge regions
        
        # Weighted L1 loss on gradients
        gradient_loss = torch.mean(edge_weight * torch.abs(sr_grad - gt_grad))
        
        return gradient_loss

class TotalVariationLoss(nn.Module):
    """Total variation regularization."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, img: torch.Tensor):
        """Compute total variation loss."""
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w

class CombinedLoss(nn.Module):
    """Combined loss function for thermal super-resolution."""
    
    def __init__(self, 
                 loss_weights: Dict[str, float] = None,
                 physics_loss_fn: Optional[nn.Module] = None):
        super().__init__()
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'content': 1.0,
                'ssim': 0.2,
                'edge': 0.1,
                'tv': 1e-4,
                'physics': 0.5
            }
        
        self.loss_weights = loss_weights
        
        # Loss components
        self.content_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeAwareLoss()
        self.tv_loss = TotalVariationLoss()
        self.physics_loss_fn = physics_loss_fn
        
    def forward(self, 
                sr_output: torch.Tensor,
                gt_thermal: torch.Tensor,
                optical_hr: torch.Tensor,
                physics_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        losses = {}
        
        # Content loss (L1)
        losses['content'] = self.content_loss(sr_output, gt_thermal)
        
        # SSIM loss
        losses['ssim'] = self.ssim_loss(sr_output, gt_thermal)
        
        # Edge-aware loss
        losses['edge'] = self.edge_loss(sr_output, gt_thermal, optical_hr)
        
        # Total variation regularization
        losses['tv'] = self.tv_loss(sr_output)
        
        # Physics consistency loss
        if self.physics_loss_fn is not None and physics_data is not None:
            physics_losses = self.physics_loss_fn(
                sr_temperature=sr_output,
                reference_temperature=gt_thermal,
                **physics_data
            )
            losses['physics'] = physics_losses['total']
            
            # Add individual physics losses for monitoring
            for k, v in physics_losses.items():
                if k != 'total':
                    losses[f'physics_{k}'] = v
        else:
            losses['physics'] = torch.tensor(0.0, device=sr_output.device)
        
        # Compute weighted total loss
        total_loss = sum(
            self.loss_weights.get(k, 0.0) * v 
            for k, v in losses.items() 
            if k in self.loss_weights
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update loss weights during training."""
        self.loss_weights.update(new_weights)
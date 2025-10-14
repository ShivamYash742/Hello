import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

class TiledInference:
    """Sliding window tiled inference with overlap and feathering."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 tile_size: int = 512,
                 overlap: int = 64,
                 device: torch.device = None):
        """
        Args:
            model: Trained SR model
            tile_size: Size of processing tiles
            overlap: Overlap between adjacent tiles
            device: Device for computation
        """
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device or torch.device('cpu')
        
        # Feathering weights for blending overlaps
        self.feather_weights = self._create_feather_weights()
        
    def _create_feather_weights(self) -> torch.Tensor:
        """Create feathering weights for smooth blending."""
        weights = torch.ones(1, 1, self.tile_size, self.tile_size)
        
        # Create linear ramp in overlap regions
        ramp = torch.linspace(0, 1, self.overlap)
        
        # Top edge
        weights[:, :, :self.overlap, :] *= ramp.view(-1, 1)
        # Bottom edge  
        weights[:, :, -self.overlap:, :] *= ramp.flip(0).view(-1, 1)
        # Left edge
        weights[:, :, :, :self.overlap] *= ramp.view(1, -1)
        # Right edge
        weights[:, :, :, -self.overlap:] *= ramp.flip(0).view(1, -1)
        
        return weights.to(self.device)
    
    def _extract_patches(self, 
                        thermal_lr: torch.Tensor,
                        optical_hr: torch.Tensor) -> list:
        """Extract overlapping patches from input images."""
        B, C_th, H_th, W_th = thermal_lr.shape
        B, C_opt, H_opt, W_opt = optical_hr.shape
        
        # Calculate stride (tile_size - overlap)
        stride = self.tile_size - self.overlap
        
        # Calculate number of tiles
        n_tiles_h = math.ceil((H_opt - self.overlap) / stride)
        n_tiles_w = math.ceil((W_opt - self.overlap) / stride)
        
        patches = []
        positions = []
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate patch coordinates
                start_h = i * stride
                start_w = j * stride
                end_h = min(start_h + self.tile_size, H_opt)
                end_w = min(start_w + self.tile_size, W_opt)
                
                # Adjust start coordinates if patch extends beyond image
                if end_h - start_h < self.tile_size:
                    start_h = max(0, end_h - self.tile_size)
                if end_w - start_w < self.tile_size:
                    start_w = max(0, end_w - self.tile_size)
                
                # Extract optical patch
                opt_patch = optical_hr[:, :, start_h:start_h+self.tile_size, 
                                      start_w:start_w+self.tile_size]
                
                # Extract corresponding thermal patch (accounting for scale difference)
                scale = self.model.scale if hasattr(self.model, 'scale') else 1
                th_start_h = start_h // scale
                th_start_w = start_w // scale
                th_size = self.tile_size // scale
                
                th_patch = thermal_lr[:, :, th_start_h:th_start_h+th_size,
                                     th_start_w:th_start_w+th_size]
                
                # Pad if necessary
                if opt_patch.shape[-2:] != (self.tile_size, self.tile_size):
                    pad_h = self.tile_size - opt_patch.shape[-2]
                    pad_w = self.tile_size - opt_patch.shape[-1]
                    opt_patch = F.pad(opt_patch, (0, pad_w, 0, pad_h), mode='reflect')
                
                if th_patch.shape[-2:] != (th_size, th_size):
                    pad_h = th_size - th_patch.shape[-2]
                    pad_w = th_size - th_patch.shape[-1]
                    th_patch = F.pad(th_patch, (0, pad_w, 0, pad_h), mode='reflect')
                
                patches.append((th_patch, opt_patch))
                positions.append((start_h, start_w, end_h, end_w))
        
        return patches, positions
    
    def _stitch_patches(self, 
                       sr_patches: list,
                       positions: list,
                       output_shape: Tuple[int, int]) -> torch.Tensor:
        """Stitch SR patches back together with feathering."""
        B, C = sr_patches[0].shape[:2]
        H_out, W_out = output_shape
        
        # Initialize output and weight accumulation tensors
        output = torch.zeros(B, C, H_out, W_out, device=self.device)
        weight_sum = torch.zeros(B, C, H_out, W_out, device=self.device)
        
        for sr_patch, (start_h, start_w, end_h, end_w) in zip(sr_patches, positions):
            # Get actual patch size (may be smaller than tile_size at boundaries)
            patch_h = end_h - start_h
            patch_w = end_w - start_w
            
            # Crop patch and weights if needed
            patch_crop = sr_patch[:, :, :patch_h, :patch_w]
            weight_crop = self.feather_weights[:, :, :patch_h, :patch_w]
            
            # Accumulate weighted patches
            output[:, :, start_h:end_h, start_w:end_w] += patch_crop * weight_crop
            weight_sum[:, :, start_h:end_h, start_w:end_w] += weight_crop
        
        # Normalize by accumulated weights
        output = output / (weight_sum + 1e-8)
        
        return output
    
    def process_large_image(self, 
                           thermal_lr: torch.Tensor,
                           optical_hr: torch.Tensor) -> torch.Tensor:
        """Process large image using tiled inference."""
        self.model.eval()
        
        # Extract patches
        patches, positions = self._extract_patches(thermal_lr, optical_hr)
        
        # Process patches
        sr_patches = []
        
        with torch.no_grad():
            for th_patch, opt_patch in patches:
                th_patch = th_patch.to(self.device)
                opt_patch = opt_patch.to(self.device)
                
                # Run inference
                sr_patch = self.model(th_patch, opt_patch)
                sr_patches.append(sr_patch.cpu())
        
        # Determine output shape
        scale = getattr(self.model, 'scale', 1)
        if hasattr(self.model, 'module'):  # Handle DataParallel
            scale = getattr(self.model.module, 'scale', 1)
        
        H_out = optical_hr.shape[-2]
        W_out = optical_hr.shape[-1]
        
        # Stitch patches together
        sr_result = self._stitch_patches(sr_patches, positions, (H_out, W_out))
        
        return sr_result
    
    def estimate_memory_usage(self, 
                             image_shape: Tuple[int, int],
                             dtype: torch.dtype = torch.float32) -> dict:
        """Estimate memory usage for tiled inference."""
        H, W = image_shape
        
        # Calculate number of tiles
        stride = self.tile_size - self.overlap
        n_tiles_h = math.ceil((H - self.overlap) / stride)
        n_tiles_w = math.ceil((W - self.overlap) / stride)
        total_tiles = n_tiles_h * n_tiles_w
        
        # Memory per tile (input + output + intermediate)
        bytes_per_element = 4 if dtype == torch.float32 else 2
        
        # Thermal LR patch
        scale = getattr(self.model, 'scale', 1)
        th_patch_size = (self.tile_size // scale) ** 2
        th_memory = th_patch_size * bytes_per_element
        
        # Optical HR patch  
        opt_patch_size = self.tile_size ** 2 * 3  # RGB
        opt_memory = opt_patch_size * bytes_per_element
        
        # SR output patch
        sr_patch_size = self.tile_size ** 2
        sr_memory = sr_patch_size * bytes_per_element
        
        # Model parameters (rough estimate)
        model_params = sum(p.numel() for p in self.model.parameters())
        model_memory = model_params * bytes_per_element
        
        # Total memory per tile
        memory_per_tile = th_memory + opt_memory + sr_memory
        
        # Peak memory (model + one tile + output accumulation)
        output_memory = H * W * bytes_per_element * 2  # output + weights
        peak_memory = model_memory + memory_per_tile + output_memory
        
        return {
            'total_tiles': total_tiles,
            'memory_per_tile_mb': memory_per_tile / (1024**2),
            'model_memory_mb': model_memory / (1024**2),
            'output_memory_mb': output_memory / (1024**2),
            'peak_memory_mb': peak_memory / (1024**2),
            'estimated_time_per_tile_ms': 100  # Rough estimate
        }

class AdaptiveTiling:
    """Adaptive tiling that adjusts tile size based on available memory."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 max_memory_mb: float = 4000,
                 device: torch.device = None):
        self.model = model
        self.max_memory_mb = max_memory_mb
        self.device = device or torch.device('cpu')
        
    def find_optimal_tile_size(self, image_shape: Tuple[int, int]) -> int:
        """Find optimal tile size based on memory constraints."""
        H, W = image_shape
        
        # Start with large tile size and reduce until memory fits
        for tile_size in [1024, 768, 512, 384, 256, 128]:
            tiled_inference = TiledInference(
                model=self.model,
                tile_size=tile_size,
                overlap=min(64, tile_size // 8),
                device=self.device
            )
            
            memory_info = tiled_inference.estimate_memory_usage(image_shape)
            
            if memory_info['peak_memory_mb'] <= self.max_memory_mb:
                return tile_size
        
        # Fallback to smallest tile size
        return 128
    
    def process_image(self, 
                     thermal_lr: torch.Tensor,
                     optical_hr: torch.Tensor) -> torch.Tensor:
        """Process image with adaptive tiling."""
        image_shape = optical_hr.shape[-2:]
        optimal_tile_size = self.find_optimal_tile_size(image_shape)
        
        print(f"Using adaptive tile size: {optimal_tile_size}")
        
        tiled_inference = TiledInference(
            model=self.model,
            tile_size=optimal_tile_size,
            overlap=min(64, optimal_tile_size // 8),
            device=self.device
        )
        
        return tiled_inference.process_large_image(thermal_lr, optical_hr)
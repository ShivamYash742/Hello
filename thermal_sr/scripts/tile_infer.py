#!/usr/bin/env python3
"""Tiled inference script for large GeoTIFF scenes."""

import argparse
import yaml
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import sys
import os
from tqdm import tqdm
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn.alignment_fusion import AlignmentFusionCNN
from models.disentangle.guidance_disentangle import GuidanceDisentangleModel
from models.swin.swin_transformer import SwinThermalSR
from utils.tiling import TiledInference
from utils.logger import setup_logger

class ThermalSRInference:
    """Tiled inference for thermal super-resolution."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: str = None,
                 device: str = 'auto'):
        
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Override with external config if provided
        if config_path:
            with open(config_path, 'r') as f:
                external_config = yaml.safe_load(f)
                self.config.update(external_config)
        
        # Initialize model
        self._setup_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup tiled inference
        self.tiled_inference = TiledInference(
            model=self.model,
            tile_size=self.config.get('inference', {}).get('tile_size', 512),
            overlap=self.config.get('inference', {}).get('overlap', 64),
            device=self.device
        )
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def _setup_model(self):
        """Initialize model from config."""
        model_type = self.config['model']['type']
        model_params = self.config['model']['params']
        
        if model_type == 'cnn':
            self.model = AlignmentFusionCNN(**model_params)
        elif model_type == 'disentangle':
            self.model = GuidanceDisentangleModel(**model_params)
        elif model_type == 'swin':
            self.model = SwinThermalSR(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
    
    def process_scene(self, 
                     optical_path: str,
                     thermal_path: str,
                     output_path: str,
                     roi_bounds: tuple = None):
        """Process a full scene with tiled inference."""
        
        optical_path = Path(optical_path)
        thermal_path = Path(thermal_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open input files
        with rasterio.open(optical_path) as opt_src, \
             rasterio.open(thermal_path) as thermal_src:
            
            # Get metadata
            optical_meta = opt_src.meta.copy()
            thermal_meta = thermal_src.meta.copy()
            
            # Determine processing bounds
            if roi_bounds:
                # Convert geographic bounds to pixel coordinates
                opt_window = rasterio.windows.from_bounds(*roi_bounds, opt_src.transform)
                thermal_window = rasterio.windows.from_bounds(*roi_bounds, thermal_src.transform)
            else:
                opt_window = Window(0, 0, opt_src.width, opt_src.height)
                thermal_window = Window(0, 0, thermal_src.width, thermal_src.height)
            
            # Read data
            optical_data = opt_src.read(window=opt_window)
            thermal_data = thermal_src.read(window=thermal_window)
            
            print(f"Optical shape: {optical_data.shape}")
            print(f"Thermal shape: {thermal_data.shape}")
            
            # Resize thermal to match optical resolution
            scale_factor = self.config['model']['params']['scale']
            if optical_data.shape[-2:] != thermal_data.shape[-2:]:
                thermal_resized = np.zeros((thermal_data.shape[0], *optical_data.shape[-2:]))
                for i in range(thermal_data.shape[0]):
                    thermal_resized[i] = cv2.resize(
                        thermal_data[i], 
                        (optical_data.shape[-1], optical_data.shape[-2]),
                        interpolation=cv2.INTER_CUBIC
                    )
                thermal_data = thermal_resized
            
            # Convert to tensors
            optical_tensor = torch.from_numpy(optical_data).float() / 255.0
            thermal_tensor = torch.from_numpy(thermal_data).float()
            
            # Add batch dimension
            optical_tensor = optical_tensor.unsqueeze(0)
            thermal_tensor = thermal_tensor.unsqueeze(0)
            
            print("Running tiled inference...")
            
            # Run tiled inference
            with torch.no_grad():
                sr_result = self.tiled_inference.process_large_image(
                    thermal_lr=thermal_tensor,
                    optical_hr=optical_tensor
                )
            
            # Convert back to numpy
            sr_result = sr_result.squeeze(0).cpu().numpy()
            
            # Update metadata for output
            output_meta = thermal_meta.copy()
            output_meta.update({
                'height': sr_result.shape[-2],
                'width': sr_result.shape[-1],
                'count': sr_result.shape[0] if len(sr_result.shape) == 3 else 1,
                'dtype': 'float32'
            })
            
            # Adjust transform for super-resolution
            if scale_factor > 1:
                transform = output_meta['transform']
                output_meta['transform'] = rasterio.Affine(
                    transform.a / scale_factor,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e / scale_factor,
                    transform.f
                )
            
            # Write output
            with rasterio.open(output_path, 'w', **output_meta) as dst:
                if len(sr_result.shape) == 2:
                    dst.write(sr_result, 1)
                else:
                    dst.write(sr_result)
            
            print(f"Super-resolution result saved to: {output_path}")
            
            # Generate quicklook PNG
            self._generate_quicklook(sr_result, output_path.with_suffix('.png'))
    
    def _generate_quicklook(self, sr_result: np.ndarray, png_path: Path):
        """Generate quicklook PNG for visualization."""
        if len(sr_result.shape) == 3:
            sr_vis = sr_result[0]  # Take first channel
        else:
            sr_vis = sr_result
        
        # Normalize to 0-255
        sr_vis = (sr_vis - sr_vis.min()) / (sr_vis.max() - sr_vis.min() + 1e-8)
        sr_vis = (sr_vis * 255).astype(np.uint8)
        
        # Apply colormap for thermal visualization
        import matplotlib.pyplot as plt
        plt.imsave(png_path, sr_vis, cmap='hot')
        print(f"Quicklook saved to: {png_path}")
    
    def process_roi(self, 
                   optical_path: str,
                   thermal_path: str,
                   output_dir: str,
                   roi_polygon: list):
        """Process region of interest defined by polygon coordinates."""
        from shapely.geometry import Polygon
        import geopandas as gpd
        
        # Create polygon
        polygon = Polygon(roi_polygon)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        
        output_path = Path(output_dir) / f"sr_roi_{hash(str(roi_polygon))}.tif"
        
        self.process_scene(
            optical_path=optical_path,
            thermal_path=thermal_path,
            output_path=output_path,
            roi_bounds=bounds
        )
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Tiled inference for thermal super-resolution')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--optical', type=str, required=True, help='Path to optical GeoTIFF')
    parser.add_argument('--thermal', type=str, required=True, help='Path to thermal GeoTIFF')
    parser.add_argument('--output', type=str, required=True, help='Output path for SR result')
    parser.add_argument('--config', type=str, help='Optional config override')
    parser.add_argument('--roi-bounds', type=float, nargs=4, 
                       help='ROI bounds as minx miny maxx maxy')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ThermalSRInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Process scene
    inference.process_scene(
        optical_path=args.optical,
        thermal_path=args.thermal,
        output_path=args.output,
        roi_bounds=tuple(args.roi_bounds) if args.roi_bounds else None
    )

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Generate synthetic thermal and optical data for testing."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import cv2
from pathlib import Path
import argparse

def generate_synthetic_optical(height: int, width: int) -> np.ndarray:
    """Generate synthetic RGB optical image."""
    
    # Create base landscape with different regions
    optical = np.zeros((3, height, width), dtype=np.float32)
    
    # Generate different land cover regions
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Water bodies (blue)
    water_mask = ((x - 0.3)**2 + (y - 0.7)**2) < 0.1
    optical[0, water_mask] = 0.1  # Low red
    optical[1, water_mask] = 0.3  # Medium green  
    optical[2, water_mask] = 0.8  # High blue
    
    # Vegetation (green)
    veg_mask = (x > 0.6) & (y < 0.4)
    optical[0, veg_mask] = 0.2
    optical[1, veg_mask] = 0.7
    optical[2, veg_mask] = 0.2
    
    # Urban areas (gray)
    urban_mask = (x < 0.4) & (y > 0.6)
    optical[0, urban_mask] = 0.5
    optical[1, urban_mask] = 0.5
    optical[2, urban_mask] = 0.5
    
    # Soil/bare ground (brown)
    soil_mask = ~(water_mask | veg_mask | urban_mask)
    optical[0, soil_mask] = 0.6
    optical[1, soil_mask] = 0.4
    optical[2, soil_mask] = 0.2
    
    # Add some texture and noise
    for c in range(3):
        # Add Perlin-like noise
        noise = np.random.randn(height//4, width//4)
        noise_resized = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        optical[c] += noise_resized * 0.1
        
        # Add fine texture
        fine_noise = np.random.randn(height, width) * 0.05
        optical[c] += fine_noise
    
    # Clip to valid range
    optical = np.clip(optical, 0, 1)
    
    return optical

def generate_synthetic_thermal(height: int, width: int, 
                              optical: np.ndarray = None) -> np.ndarray:
    """Generate synthetic thermal image correlated with optical."""
    
    # Base temperature map (in Kelvin)
    base_temp = 295.0  # ~22°C
    thermal = np.full((height, width), base_temp, dtype=np.float32)
    
    if optical is not None:
        # Use optical to guide thermal generation
        luminance = np.mean(optical, axis=0)
        
        # Water is cooler
        water_mask = optical[2] > 0.6  # High blue channel
        thermal[water_mask] -= 10.0
        
        # Vegetation is moderate
        veg_mask = optical[1] > 0.6  # High green channel
        thermal[veg_mask] -= 3.0
        
        # Urban areas are warmer (heat island effect)
        urban_mask = (optical[0] > 0.4) & (optical[1] > 0.4) & (optical[2] > 0.4)
        thermal[urban_mask] += 8.0
        
        # Bare soil varies with illumination
        thermal += (luminance - 0.5) * 5.0
    
    # Add spatial correlation patterns
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Add some hot spots (buildings, etc.)
    hot_spots = np.sin(x * 10) * np.cos(y * 8) * 3.0
    thermal += hot_spots
    
    # Add smooth temperature gradients
    gradient = (x + y) * 2.0
    thermal += gradient
    
    # Add noise
    noise = np.random.randn(height, width) * 1.5
    thermal += noise
    
    # Ensure reasonable temperature range
    thermal = np.clip(thermal, 280.0, 320.0)  # -7°C to 47°C
    
    return thermal[np.newaxis, ...]  # Add channel dimension

def create_land_cover_mask(height: int, width: int, optical: np.ndarray) -> np.ndarray:
    """Create land cover classification mask."""
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Water = 1
    water_mask = optical[2] > 0.6
    mask[water_mask] = 1
    
    # Vegetation = 2  
    veg_mask = optical[1] > 0.6
    mask[veg_mask] = 2
    
    # Urban = 3
    urban_mask = (optical[0] > 0.4) & (optical[1] > 0.4) & (optical[2] > 0.4)
    mask[urban_mask] = 3
    
    # Soil = 4 (default)
    soil_mask = ~(water_mask | veg_mask | urban_mask)
    mask[soil_mask] = 4
    
    return mask

def save_geotiff(data: np.ndarray, 
                output_path: str,
                bounds: tuple = (-180, -90, 180, 90),
                crs: str = 'EPSG:4326'):
    """Save data as GeoTIFF."""
    
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    count, height, width = data.shape
    
    # Create transform
    transform = from_bounds(*bounds, width, height)
    
    # Metadata
    meta = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'nodata': None,
        'width': width,
        'height': height,
        'count': count,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    # Write file
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data)

def generate_dataset(output_dir: str, 
                    num_scenes: int = 10,
                    hr_size: tuple = (512, 512),
                    lr_scale: int = 2):
    """Generate complete synthetic dataset."""
    
    output_dir = Path(output_dir)
    
    # Create directories
    (output_dir / 'optical').mkdir(parents=True, exist_ok=True)
    (output_dir / 'thermal').mkdir(parents=True, exist_ok=True)
    (output_dir / 'thermal_lr').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    
    hr_height, hr_width = hr_size
    lr_height, lr_width = hr_height // lr_scale, hr_width // lr_scale
    
    print(f"Generating {num_scenes} synthetic scenes...")
    print(f"HR size: {hr_size}, LR size: ({lr_height}, {lr_width})")
    
    for i in range(num_scenes):
        print(f"Generating scene {i+1}/{num_scenes}")
        
        # Generate HR optical
        optical_hr = generate_synthetic_optical(hr_height, hr_width)
        
        # Generate HR thermal (correlated with optical)
        thermal_hr = generate_synthetic_thermal(hr_height, hr_width, optical_hr)
        
        # Generate LR thermal (downsampled)
        thermal_lr = cv2.resize(
            thermal_hr[0], (lr_width, lr_height), 
            interpolation=cv2.INTER_CUBIC
        )[np.newaxis, ...]
        
        # Generate land cover mask
        land_cover = create_land_cover_mask(hr_height, hr_width, optical_hr)
        
        # Define geographic bounds (random locations)
        lon_center = np.random.uniform(-120, 120)
        lat_center = np.random.uniform(-60, 60)
        extent = 0.1  # degrees
        
        bounds = (
            lon_center - extent/2, lat_center - extent/2,
            lon_center + extent/2, lat_center + extent/2
        )
        
        # Save files
        scene_id = f"scene_{i:03d}"
        
        # Optical (convert to 0-255 range)
        optical_uint8 = (optical_hr * 255).astype(np.uint8)
        save_geotiff(
            optical_uint8,
            output_dir / 'optical' / f'{scene_id}.tif',
            bounds=bounds
        )
        
        # Thermal HR
        save_geotiff(
            thermal_hr.astype(np.float32),
            output_dir / 'thermal' / f'{scene_id}.tif',
            bounds=bounds
        )
        
        # Thermal LR
        lr_bounds = bounds  # Same geographic bounds, different resolution
        save_geotiff(
            thermal_lr.astype(np.float32),
            output_dir / 'thermal_lr' / f'{scene_id}.tif',
            bounds=lr_bounds
        )
        
        # Land cover mask
        save_geotiff(
            land_cover.astype(np.uint8),
            output_dir / 'masks' / f'{scene_id}.tif',
            bounds=bounds
        )
    
    print(f"Dataset generated successfully in: {output_dir}")
    
    # Create dataset info file
    info = {
        'num_scenes': num_scenes,
        'hr_size': hr_size,
        'lr_size': (lr_height, lr_width),
        'scale_factor': lr_scale,
        'land_cover_classes': {
            0: 'background',
            1: 'water',
            2: 'vegetation', 
            3: 'urban',
            4: 'soil'
        },
        'temperature_range_kelvin': [280.0, 320.0],
        'optical_range': [0, 255]
    }
    
    import yaml
    with open(output_dir / 'dataset_info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic thermal SR dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--num-scenes', type=int, default=20,
                       help='Number of scenes to generate')
    parser.add_argument('--hr-size', type=int, nargs=2, default=[512, 512],
                       help='HR image size (height width)')
    parser.add_argument('--scale', type=int, default=2,
                       help='Downsampling scale factor')
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        hr_size=tuple(args.hr_size),
        lr_scale=args.scale
    )

if __name__ == '__main__':
    main()
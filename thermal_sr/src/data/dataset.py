import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, Optional, Dict, Any

class ThermalSRDataset(Dataset):
    """Dataset for paired optical-thermal super-resolution."""
    
    def __init__(self, 
                 data_dir: str,
                 scale: int = 2,
                 patch_size: int = 128,
                 augment: bool = True,
                 split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.split = split
        
        # Load file pairs
        self.pairs = self._load_pairs()
        
    def _load_pairs(self):
        """Load optical-thermal file pairs."""
        pairs = []
        optical_dir = self.data_dir / 'optical'
        thermal_dir = self.data_dir / 'thermal'
        
        for opt_file in optical_dir.glob('*.tif'):
            thermal_file = thermal_dir / opt_file.name
            if thermal_file.exists():
                pairs.append((opt_file, thermal_file))
        return pairs
    
    def _read_geotiff(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Read GeoTIFF with metadata."""
        with rasterio.open(path) as src:
            data = src.read()
            meta = {
                'crs': src.crs,
                'transform': src.transform,
                'nodata': src.nodata
            }
        return data, meta
    
    def _synthesize_lr(self, hr_thermal: np.ndarray) -> np.ndarray:
        """Synthesize LR thermal from HR via blur-downsample."""
        # Gaussian blur then bicubic downsample
        blurred = cv2.GaussianBlur(hr_thermal, (5, 5), 1.0)
        h, w = blurred.shape
        lr = cv2.resize(blurred, (w//self.scale, h//self.scale), 
                       interpolation=cv2.INTER_CUBIC)
        return lr
    
    def _augment_pair(self, optical: np.ndarray, thermal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        if not self.augment:
            return optical, thermal
            
        # Random flip
        if np.random.rand() > 0.5:
            optical = np.flip(optical, axis=-1)
            thermal = np.flip(thermal, axis=-1)
            
        # Random rotation (90 degree multiples)
        k = np.random.randint(0, 4)
        optical = np.rot90(optical, k, axes=(-2, -1))
        thermal = np.rot90(thermal, k, axes=(-2, -1))
        
        return optical, thermal
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        opt_path, thermal_path = self.pairs[idx]
        
        # Read data
        optical, opt_meta = self._read_geotiff(opt_path)
        thermal_hr, thermal_meta = self._read_geotiff(thermal_path)
        
        # Synthesize LR thermal
        thermal_lr = self._synthesize_lr(thermal_hr[0])
        
        # Random crop to patch size
        if self.split == 'train':
            h, w = thermal_hr.shape[-2:]
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            
            optical = optical[:, top:top+self.patch_size, left:left+self.patch_size]
            thermal_hr = thermal_hr[:, top:top+self.patch_size, left:left+self.patch_size]
            
            # Corresponding LR patch
            lr_top, lr_left = top//self.scale, left//self.scale
            lr_size = self.patch_size // self.scale
            thermal_lr = thermal_lr[lr_top:lr_top+lr_size, lr_left:lr_left+lr_size]
        
        # Augmentation
        optical, thermal_hr = self._augment_pair(optical, thermal_hr[0])
        
        # Convert to tensors
        optical = torch.from_numpy(optical).float() / 255.0
        thermal_lr = torch.from_numpy(thermal_lr).float().unsqueeze(0)
        thermal_hr = torch.from_numpy(thermal_hr).float().unsqueeze(0)
        
        return {
            'optical': optical,
            'thermal_lr': thermal_lr,
            'thermal_hr': thermal_hr,
            'scale': self.scale
        }

class ThermalSRDataModule:
    """Data module for thermal SR training."""
    
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 scale: int = 2,
                 patch_size: int = 128):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scale = scale
        self.patch_size = patch_size
        
    def train_dataloader(self):
        dataset = ThermalSRDataset(
            self.data_dir, 
            scale=self.scale,
            patch_size=self.patch_size,
            augment=True,
            split='train'
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        dataset = ThermalSRDataset(
            self.data_dir,
            scale=self.scale, 
            patch_size=self.patch_size,
            augment=False,
            split='val'
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers
        )
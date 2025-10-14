import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Optional, Tuple
import csv
from pathlib import Path

class ThermalSRMetrics:
    """Comprehensive evaluation metrics for thermal super-resolution."""
    
    def __init__(self, temperature_range: Tuple[float, float] = (273.15, 373.15)):
        """
        Args:
            temperature_range: (min_temp, max_temp) in Kelvin for PSNR calculation
        """
        self.temp_min, self.temp_max = temperature_range
        
    def psnr_kelvin(self, sr_temp: np.ndarray, gt_temp: np.ndarray) -> float:
        """Compute PSNR in Kelvin temperature domain."""
        mse = np.mean((sr_temp - gt_temp) ** 2)
        if mse == 0:
            return float('inf')
        
        max_temp = self.temp_max
        psnr = 20 * np.log10(max_temp) - 10 * np.log10(mse)
        return psnr
    
    def rmse_kelvin(self, sr_temp: np.ndarray, gt_temp: np.ndarray) -> float:
        """Compute RMSE in Kelvin."""
        mse = np.mean((sr_temp - gt_temp) ** 2)
        return np.sqrt(mse)
    
    def ssim_metric(self, 
                   sr_temp: np.ndarray, 
                   gt_temp: np.ndarray,
                   window_size: int = 11) -> float:
        """Compute SSIM with configurable window size."""
        # Normalize to [0, 1] range for SSIM
        sr_norm = (sr_temp - self.temp_min) / (self.temp_max - self.temp_min)
        gt_norm = (gt_temp - self.temp_min) / (self.temp_max - self.temp_min)
        
        # Ensure single channel
        if len(sr_norm.shape) == 3:
            sr_norm = sr_norm[0]
        if len(gt_norm.shape) == 3:
            gt_norm = gt_norm[0]
        
        return ssim(gt_norm, sr_norm, win_size=window_size, data_range=1.0)
    
    def compute_per_class_metrics(self, 
                                 sr_temp: np.ndarray,
                                 gt_temp: np.ndarray,
                                 class_mask: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Compute metrics stratified by land cover class."""
        unique_classes = np.unique(class_mask)
        per_class_metrics = {}
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background/no-data
                continue
                
            mask = (class_mask == class_id)
            if np.sum(mask) < 100:  # Skip classes with too few pixels
                continue
            
            sr_class = sr_temp[mask]
            gt_class = gt_temp[mask]
            
            per_class_metrics[int(class_id)] = {
                'psnr': self.psnr_kelvin(sr_class, gt_class),
                'rmse': self.rmse_kelvin(sr_class, gt_class),
                'mae': np.mean(np.abs(sr_class - gt_class)),
                'pixel_count': np.sum(mask)
            }
        
        return per_class_metrics
    
    def evaluate_full_image(self, 
                           sr_temp: np.ndarray,
                           gt_temp: np.ndarray,
                           class_mask: Optional[np.ndarray] = None) -> Dict:
        """Compute comprehensive evaluation metrics."""
        # Ensure numpy arrays
        if torch.is_tensor(sr_temp):
            sr_temp = sr_temp.detach().cpu().numpy()
        if torch.is_tensor(gt_temp):
            gt_temp = gt_temp.detach().cpu().numpy()
        
        # Remove batch dimension if present
        if len(sr_temp.shape) == 4:
            sr_temp = sr_temp[0, 0]
            gt_temp = gt_temp[0, 0]
        elif len(sr_temp.shape) == 3:
            sr_temp = sr_temp[0]
            gt_temp = gt_temp[0]
        
        # Full image metrics
        metrics = {
            'psnr': self.psnr_kelvin(sr_temp, gt_temp),
            'ssim': self.ssim_metric(sr_temp, gt_temp),
            'rmse': self.rmse_kelvin(sr_temp, gt_temp),
            'mae': np.mean(np.abs(sr_temp - gt_temp)),
            'bias': np.mean(sr_temp - gt_temp),
            'std_error': np.std(sr_temp - gt_temp)
        }
        
        # Per-class metrics if mask provided
        if class_mask is not None:
            if torch.is_tensor(class_mask):
                class_mask = class_mask.detach().cpu().numpy()
            
            if len(class_mask.shape) == 4:
                class_mask = class_mask[0, 0]
            elif len(class_mask.shape) == 3:
                class_mask = class_mask[0]
            
            per_class = self.compute_per_class_metrics(sr_temp, gt_temp, class_mask)
            metrics['per_class'] = per_class
        
        return metrics
    
    def save_metrics_csv(self, 
                        metrics_list: list,
                        save_path: str,
                        include_per_class: bool = True):
        """Save metrics to CSV file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare CSV data
        fieldnames = ['image_id', 'psnr', 'ssim', 'rmse', 'mae', 'bias', 'std_error']
        
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, metrics in enumerate(metrics_list):
                row = {
                    'image_id': i,
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'bias': metrics['bias'],
                    'std_error': metrics['std_error']
                }
                writer.writerow(row)
        
        # Save per-class metrics separately if available
        if include_per_class and any('per_class' in m for m in metrics_list):
            per_class_path = save_path.parent / f"{save_path.stem}_per_class.csv"
            
            with open(per_class_path, 'w', newline='') as csvfile:
                fieldnames = ['image_id', 'class_id', 'psnr', 'rmse', 'mae', 'pixel_count']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, metrics in enumerate(metrics_list):
                    if 'per_class' in metrics:
                        for class_id, class_metrics in metrics['per_class'].items():
                            row = {
                                'image_id': i,
                                'class_id': class_id,
                                **class_metrics
                            }
                            writer.writerow(row)
    
    def compute_summary_statistics(self, metrics_list: list) -> Dict:
        """Compute summary statistics across multiple images."""
        if not metrics_list:
            return {}
        
        # Aggregate metrics
        psnr_values = [m['psnr'] for m in metrics_list if np.isfinite(m['psnr'])]
        ssim_values = [m['ssim'] for m in metrics_list]
        rmse_values = [m['rmse'] for m in metrics_list]
        mae_values = [m['mae'] for m in metrics_list]
        
        summary = {
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
            'num_images': len(metrics_list)
        }
        
        return summary

class LeakageDetector:
    """Detect optical texture leakage in thermal SR results."""
    
    def __init__(self):
        pass
    
    def compute_edge_correlation(self, 
                               sr_thermal: np.ndarray,
                               optical_hr: np.ndarray) -> float:
        """Compute correlation between thermal and optical edges."""
        from scipy import ndimage
        
        # Compute edge maps
        thermal_edges = ndimage.sobel(sr_thermal)
        optical_lum = np.mean(optical_hr, axis=0) if len(optical_hr.shape) == 3 else optical_hr
        optical_edges = ndimage.sobel(optical_lum)
        
        # Normalize edge maps
        thermal_edges = (thermal_edges - np.mean(thermal_edges)) / (np.std(thermal_edges) + 1e-8)
        optical_edges = (optical_edges - np.mean(optical_edges)) / (np.std(optical_edges) + 1e-8)
        
        # Compute correlation
        correlation = np.corrcoef(thermal_edges.flatten(), optical_edges.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def texture_leakage_score(self, 
                             sr_thermal: np.ndarray,
                             lr_thermal: np.ndarray,
                             optical_hr: np.ndarray) -> float:
        """Compute texture leakage score."""
        # Upsample LR thermal for comparison
        from scipy.ndimage import zoom
        scale_factor = sr_thermal.shape[0] / lr_thermal.shape[0]
        lr_upsampled = zoom(lr_thermal, scale_factor, order=1)
        
        # Compute high-frequency components
        sr_hf = sr_thermal - ndimage.gaussian_filter(sr_thermal, sigma=2.0)
        lr_hf = lr_upsampled - ndimage.gaussian_filter(lr_upsampled, sigma=2.0)
        
        # Additional high-frequency content in SR
        added_hf = np.abs(sr_hf) - np.abs(lr_hf)
        added_hf = np.maximum(added_hf, 0)
        
        # Optical high-frequency content
        optical_lum = np.mean(optical_hr, axis=0) if len(optical_hr.shape) == 3 else optical_hr
        optical_hf = optical_lum - ndimage.gaussian_filter(optical_lum, sigma=2.0)
        
        # Correlation between added thermal HF and optical HF
        if np.std(added_hf) > 1e-8 and np.std(optical_hf) > 1e-8:
            leakage_score = np.corrcoef(added_hf.flatten(), optical_hf.flatten())[0, 1]
            return leakage_score if not np.isnan(leakage_score) else 0.0
        else:
            return 0.0
#!/usr/bin/env python3
"""Evaluation script for thermal super-resolution models."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import ThermalSRDataset
from models.cnn.alignment_fusion import AlignmentFusionCNN
from models.disentangle.guidance_disentangle import GuidanceDisentangleModel
from models.swin.swin_transformer import SwinThermalSR
from metrics.evaluation import ThermalSRMetrics, LeakageDetector
from utils.logger import setup_logger

class ThermalSREvaluator:
    """Evaluator for thermal super-resolution models."""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Override with external config if provided
        if config_path:
            with open(config_path, 'r') as f:
                external_config = yaml.safe_load(f)
                self.config.update(external_config)
        
        # Setup model
        self._setup_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup metrics
        self.metrics = ThermalSRMetrics()
        self.leakage_detector = LeakageDetector()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model type: {self.config['model']['type']}")
    
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
    
    def evaluate_dataset(self, 
                        data_dir: str,
                        output_dir: str,
                        split: str = 'test',
                        save_visualizations: bool = True):
        """Evaluate model on a dataset."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        logger = setup_logger(output_dir / 'evaluation.log')
        
        # Create dataset
        dataset = ThermalSRDataset(
            data_dir=data_dir,
            scale=self.config['model']['params']['scale'],
            patch_size=None,  # Use full images for evaluation
            augment=False,
            split=split
        )
        
        logger.info(f"Evaluating on {len(dataset)} images")
        
        # Evaluation loop
        all_metrics = []
        leakage_scores = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataset, desc="Evaluating")):
                # Move to device
                optical = batch['optical'].unsqueeze(0).to(self.device)
                thermal_lr = batch['thermal_lr'].unsqueeze(0).to(self.device)
                thermal_hr = batch['thermal_hr'].unsqueeze(0).to(self.device)
                
                # Run inference
                sr_output = self.model(thermal_lr, optical)
                
                # Compute metrics
                metrics = self.metrics.evaluate_full_image(sr_output, thermal_hr)
                all_metrics.append(metrics)
                
                # Compute leakage scores
                sr_np = sr_output.squeeze().cpu().numpy()
                lr_np = thermal_lr.squeeze().cpu().numpy()
                opt_np = optical.squeeze().cpu().numpy()
                
                edge_corr = self.leakage_detector.compute_edge_correlation(sr_np, opt_np)
                texture_leak = self.leakage_detector.texture_leakage_score(sr_np, lr_np, opt_np)
                
                leakage_scores.append({
                    'edge_correlation': edge_corr,
                    'texture_leakage': texture_leak
                })
                
                # Save visualizations
                if save_visualizations and i < 10:  # Save first 10 images
                    self._save_visualization(
                        optical.cpu(), thermal_lr.cpu(), thermal_hr.cpu(), sr_output.cpu(),
                        output_dir / f'vis_{i:03d}.png'
                    )
        
        # Compute summary statistics
        summary_metrics = self.metrics.compute_summary_statistics(all_metrics)
        
        # Compute leakage statistics
        avg_edge_corr = np.mean([s['edge_correlation'] for s in leakage_scores])
        avg_texture_leak = np.mean([s['texture_leakage'] for s in leakage_scores])
        
        # Log results
        logger.info("=== Evaluation Results ===")
        logger.info(f"PSNR: {summary_metrics['psnr_mean']:.2f} ± {summary_metrics['psnr_std']:.2f} dB")
        logger.info(f"SSIM: {summary_metrics['ssim_mean']:.4f} ± {summary_metrics['ssim_std']:.4f}")
        logger.info(f"RMSE: {summary_metrics['rmse_mean']:.2f} ± {summary_metrics['rmse_std']:.2f} K")
        logger.info(f"MAE: {summary_metrics['mae_mean']:.2f} ± {summary_metrics['mae_std']:.2f} K")
        logger.info(f"Edge Correlation: {avg_edge_corr:.4f}")
        logger.info(f"Texture Leakage: {avg_texture_leak:.4f}")
        
        # Save detailed metrics
        self.metrics.save_metrics_csv(all_metrics, output_dir / 'detailed_metrics.csv')
        
        # Save summary
        summary_results = {
            **summary_metrics,
            'avg_edge_correlation': avg_edge_corr,
            'avg_texture_leakage': avg_texture_leak
        }
        
        with open(output_dir / 'summary_metrics.yaml', 'w') as f:
            yaml.dump(summary_results, f, default_flow_style=False)
        
        return summary_results
    
    def _save_visualization(self, 
                           optical: torch.Tensor,
                           thermal_lr: torch.Tensor,
                           thermal_hr: torch.Tensor,
                           sr_output: torch.Tensor,
                           save_path: Path):
        """Save visualization comparing inputs and outputs."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Convert to numpy and normalize
        opt_vis = optical.squeeze().permute(1, 2, 0).numpy()
        opt_vis = (opt_vis - opt_vis.min()) / (opt_vis.max() - opt_vis.min() + 1e-8)
        
        th_lr_vis = thermal_lr.squeeze().numpy()
        th_hr_vis = thermal_hr.squeeze().numpy()
        sr_vis = sr_output.squeeze().numpy()
        
        # Plot images
        axes[0, 0].imshow(opt_vis)
        axes[0, 0].set_title('Optical HR')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(th_lr_vis, cmap='hot')
        axes[0, 1].set_title('Thermal LR')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(th_hr_vis, cmap='hot')
        axes[0, 2].set_title('Thermal HR (GT)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(sr_vis, cmap='hot')
        axes[1, 0].set_title('Thermal SR')
        axes[1, 0].axis('off')
        
        # Error map
        error_map = np.abs(sr_vis - th_hr_vis)
        im = axes[1, 1].imshow(error_map, cmap='viridis')
        axes[1, 1].set_title('Error Map')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        
        # Difference histogram
        diff = (sr_vis - th_hr_vis).flatten()
        axes[1, 2].hist(diff, bins=50, alpha=0.7)
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].set_xlabel('Temperature Difference (K)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate thermal super-resolution model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--config', type=str, help='Optional config override')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ThermalSREvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        data_dir=args.data,
        output_dir=args.output,
        split=args.split,
        save_visualizations=not args.no_vis
    )
    
    print("\n=== Final Results ===")
    print(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"RMSE: {results['rmse_mean']:.2f} ± {results['rmse_std']:.2f} K")

if __name__ == '__main__':
    main()
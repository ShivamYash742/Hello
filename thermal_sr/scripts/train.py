#!/usr/bin/env python3
"""Training script for thermal super-resolution models."""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import ThermalSRDataModule
from models.cnn.alignment_fusion import AlignmentFusionCNN
from models.disentangle.guidance_disentangle import GuidanceDisentangleModel
from models.swin.swin_transformer import SwinThermalSR
from losses.combined_loss import CombinedLoss
from physics.atmospheric import PhysicsConsistencyLoss, AtmosphericCorrection, EmissivityHandler
from metrics.evaluation import ThermalSRMetrics
from utils.checkpoint import CheckpointManager
from utils.logger import setup_logger

class ThermalSRTrainer:
    """Trainer for thermal super-resolution models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.run_dir = Path(config['output_dir']) / f"run_{config['experiment_name']}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(self.run_dir / 'train.log')
        self.writer = SummaryWriter(self.run_dir / 'tensorboard')
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_metrics()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.run_dir / 'checkpoints',
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        
        self.global_step = 0
        self.best_val_rmse = float('inf')
        
    def _setup_model(self):
        """Initialize the model based on config."""
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
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {model_type}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_data(self):
        """Setup data loaders."""
        data_config = self.config['data']
        self.data_module = ThermalSRDataModule(**data_config)
        
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _setup_loss(self):
        """Setup loss functions."""
        loss_config = self.config['loss']
        
        # Physics loss if enabled
        physics_loss = None
        if loss_config.get('use_physics', False):
            # Setup atmospheric correction and emissivity handler
            sensor_params = {
                'wavelength': 10.8,
                'bandwidth': 1.0,
                'calibration_constants': [774.89, 1321.08]  # Example for Landsat 8 TIRS
            }
            
            emissivity_table = {
                1: 0.92,  # Soil
                2: 0.98,  # Vegetation
                3: 0.99,  # Water
                4: 0.94   # Urban
            }
            
            atmospheric_corrector = AtmosphericCorrection(sensor_params)
            emissivity_handler = EmissivityHandler(emissivity_table)
            
            physics_loss = PhysicsConsistencyLoss(atmospheric_corrector, emissivity_handler)
        
        self.criterion = CombinedLoss(
            loss_weights=loss_config['weights'],
            physics_loss_fn=physics_loss
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
        
        # Learning rate scheduler
        sched_config = self.config.get('scheduler', {})
        if sched_config.get('type') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_config.get('type') == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        self.metrics = ThermalSRMetrics()
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            optical = batch['optical'].to(self.device)
            thermal_lr = batch['thermal_lr'].to(self.device)
            thermal_hr = batch['thermal_hr'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.config['training'].get('mixed_precision', False)):
                sr_output = self.model(thermal_lr, optical)
                
                # Compute loss
                loss_dict = self.criterion(sr_output, thermal_hr, optical)
                loss = loss_dict['total']
            
            # Backward pass
            if self.config['training'].get('mixed_precision', False):
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                
                # TensorBoard logging
                for loss_name, loss_value in loss_dict.items():
                    self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), self.global_step)
                
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        metrics_list = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                optical = batch['optical'].to(self.device)
                thermal_lr = batch['thermal_lr'].to(self.device)
                thermal_hr = batch['thermal_hr'].to(self.device)
                
                # Forward pass
                sr_output = self.model(thermal_lr, optical)
                
                # Compute loss
                loss_dict = self.criterion(sr_output, thermal_hr, optical)
                total_loss += loss_dict['total'].item()
                
                # Compute metrics
                metrics = self.metrics.evaluate_full_image(sr_output, thermal_hr)
                metrics_list.append(metrics)
        
        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        summary_metrics = self.metrics.compute_summary_statistics(metrics_list)
        
        # Logging
        self.logger.info(f"Validation - Epoch {epoch}")
        self.logger.info(f"Loss: {avg_loss:.6f}")
        self.logger.info(f"PSNR: {summary_metrics['psnr_mean']:.2f} ± {summary_metrics['psnr_std']:.2f}")
        self.logger.info(f"SSIM: {summary_metrics['ssim_mean']:.4f} ± {summary_metrics['ssim_std']:.4f}")
        self.logger.info(f"RMSE: {summary_metrics['rmse_mean']:.2f} ± {summary_metrics['rmse_std']:.2f}")
        
        # TensorBoard logging
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        for metric_name, metric_value in summary_metrics.items():
            if 'mean' in metric_name:
                self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
        
        return avg_loss, summary_metrics['rmse_mean']
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Mixed precision scaler
        if self.config['training'].get('mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_rmse = self.validate(epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_rmse < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_rmse
            
            self.checkpoint_manager.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'config': self.config
            }, is_best=is_best)
            
            # Early stopping
            if self.config['training'].get('early_stopping', 0) > 0:
                if epoch - self.checkpoint_manager.best_epoch > self.config['training']['early_stopping']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train thermal super-resolution model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    torch.manual_seed(config.get('seed', 42))
    torch.cuda.manual_seed_all(config.get('seed', 42))
    
    # Create trainer and train
    trainer = ThermalSRTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.checkpoint_manager.load_checkpoint(args.resume, trainer.model, trainer.optimizer)
    
    trainer.train()

if __name__ == '__main__':
    main()
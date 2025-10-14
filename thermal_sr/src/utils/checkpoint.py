import torch
from pathlib import Path
import shutil
from typing import Dict, Optional

class CheckpointManager:
    """Manage model checkpoints with best model tracking."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_checkpoints: int = 5):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('inf')
        self.best_epoch = 0
        
    def save_checkpoint(self, 
                       state_dict: Dict,
                       is_best: bool = False,
                       filename: str = None) -> Path:
        """Save checkpoint and manage checkpoint history."""
        
        if filename is None:
            epoch = state_dict.get('epoch', 0)
            filename = f'checkpoint_epoch_{epoch:03d}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(state_dict, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            shutil.copy2(checkpoint_path, best_path)
            
            self.best_metric = state_dict.get('val_rmse', float('inf'))
            self.best_epoch = state_dict.get('epoch', 0)
            
            print(f"New best model saved with RMSE: {self.best_metric:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer = None,
                       scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict:
        """Load checkpoint and restore model/optimizer state."""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if hasattr(model, 'module'):  # DataParallel model
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number and return latest
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]
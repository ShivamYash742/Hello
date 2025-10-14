import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn.alignment_fusion import AlignmentFusionCNN
from models.disentangle.guidance_disentangle import GuidanceDisentangleModel
from models.swin.swin_transformer import SwinThermalSR

class TestThermalSRModels:
    """Test suite for thermal super-resolution models."""
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors."""
        batch_size = 2
        thermal_lr = torch.randn(batch_size, 1, 64, 64)
        optical_hr = torch.randn(batch_size, 3, 128, 128)
        return thermal_lr, optical_hr
    
    def test_alignment_fusion_cnn(self, sample_inputs):
        """Test Alignment-Fusion CNN model."""
        thermal_lr, optical_hr = sample_inputs
        
        # Test scale 2
        model = AlignmentFusionCNN(scale=2, num_channels=32)
        model.eval()
        
        with torch.no_grad():
            output = model(thermal_lr, optical_hr)
        
        # Check output shape
        expected_shape = (2, 1, 128, 128)  # Same as optical HR
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        # Test scale 4
        model_4x = AlignmentFusionCNN(scale=4, num_channels=32)
        model_4x.eval()
        
        with torch.no_grad():
            output_4x = model_4x(thermal_lr, optical_hr)
        
        assert output_4x.shape == expected_shape, "Scale 4 output shape incorrect"
    
    def test_guidance_disentangle(self, sample_inputs):
        """Test Guidance-Disentanglement model."""
        thermal_lr, optical_hr = sample_inputs
        
        model = GuidanceDisentangleModel(scale=2, num_channels=32)
        model.eval()
        
        with torch.no_grad():
            output = model(thermal_lr, optical_hr)
        
        # Check output shape
        expected_shape = (2, 1, 128, 128)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        # Test disentanglement loss
        structure, texture = model.disentangler(optical_hr)
        disentangle_loss = model.compute_disentanglement_loss(structure, texture)
        
        assert torch.isfinite(disentangle_loss), "Disentanglement loss is not finite"
        assert disentangle_loss >= 0, "Disentanglement loss should be non-negative"
    
    def test_swin_transformer(self, sample_inputs):
        """Test Swin Transformer model."""
        thermal_lr, optical_hr = sample_inputs
        
        # Use smaller model for testing
        model = SwinThermalSR(
            scale=2,
            patch_size=4,
            embed_dim=48,
            depths=[1, 1, 1, 1],
            num_heads=[2, 4, 8, 16],
            window_size=4
        )
        model.eval()
        
        with torch.no_grad():
            output = model(thermal_lr, optical_hr)
        
        # Check output shape
        expected_shape = (2, 1, 128, 128)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_model_parameters(self):
        """Test model parameter counts are reasonable."""
        
        # CNN model
        cnn_model = AlignmentFusionCNN(scale=2, num_channels=64)
        cnn_params = sum(p.numel() for p in cnn_model.parameters())
        assert 100000 < cnn_params < 10000000, f"CNN params {cnn_params} outside expected range"
        
        # Disentanglement model
        disentangle_model = GuidanceDisentangleModel(scale=2, num_channels=64)
        disentangle_params = sum(p.numel() for p in disentangle_model.parameters())
        assert 100000 < disentangle_params < 20000000, f"Disentangle params {disentangle_params} outside expected range"
        
        # Swin model (smaller version)
        swin_model = SwinThermalSR(
            scale=2, embed_dim=48, depths=[1, 1, 1, 1], num_heads=[2, 4, 8, 16]
        )
        swin_params = sum(p.numel() for p in swin_model.parameters())
        assert 50000 < swin_params < 50000000, f"Swin params {swin_params} outside expected range"
    
    def test_gradient_flow(self, sample_inputs):
        """Test that gradients flow properly through models."""
        thermal_lr, optical_hr = sample_inputs
        
        models = [
            AlignmentFusionCNN(scale=2, num_channels=32),
            GuidanceDisentangleModel(scale=2, num_channels=32),
            SwinThermalSR(scale=2, embed_dim=48, depths=[1, 1], num_heads=[2, 4])
        ]
        
        for model in models:
            model.train()
            
            # Forward pass
            output = model(thermal_lr, optical_hr)
            
            # Dummy loss
            loss = torch.mean(output)
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            assert has_gradients, f"No gradients found for {model.__class__.__name__}"
            
            # Check gradients are finite
            grad_finite = all(
                torch.isfinite(p.grad).all() 
                for p in model.parameters() 
                if p.grad is not None
            )
            assert grad_finite, f"Non-finite gradients in {model.__class__.__name__}"
    
    def test_different_input_sizes(self):
        """Test models with different input sizes."""
        model = AlignmentFusionCNN(scale=2, num_channels=32)
        model.eval()
        
        # Test different sizes
        sizes = [(32, 32), (64, 64), (96, 96)]
        
        for h, w in sizes:
            thermal_lr = torch.randn(1, 1, h, w)
            optical_hr = torch.randn(1, 3, h*2, w*2)
            
            with torch.no_grad():
                output = model(thermal_lr, optical_hr)
            
            expected_shape = (1, 1, h*2, w*2)
            assert output.shape == expected_shape, f"Size {(h,w)}: expected {expected_shape}, got {output.shape}"

if __name__ == "__main__":
    pytest.main([__file__])
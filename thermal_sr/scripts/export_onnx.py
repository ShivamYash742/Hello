#!/usr/bin/env python3
"""Export trained models to ONNX format."""

import argparse
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn.alignment_fusion import AlignmentFusionCNN
from models.disentangle.guidance_disentangle import GuidanceDisentangleModel
from models.swin.swin_transformer import SwinThermalSR

def export_to_onnx(checkpoint_path: str, 
                   output_path: str,
                   input_shape: tuple = (1, 1, 64, 64),
                   optical_shape: tuple = (1, 3, 128, 128),
                   opset_version: int = 11):
    """Export PyTorch model to ONNX format."""
    
    device = torch.device('cpu')  # Export on CPU for compatibility
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Initialize model
    model_type = config['model']['type']
    model_params = config['model']['params']
    
    if model_type == 'cnn':
        model = AlignmentFusionCNN(**model_params)
    elif model_type == 'disentangle':
        model = GuidanceDisentangleModel(**model_params)
    elif model_type == 'swin':
        model = SwinThermalSR(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy inputs
    thermal_input = torch.randn(input_shape)
    optical_input = torch.randn(optical_shape)
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {model_type} model to ONNX...")
    print(f"Thermal input shape: {input_shape}")
    print(f"Optical input shape: {optical_shape}")
    
    torch.onnx.export(
        model,
        (thermal_input, optical_input),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['thermal_lr', 'optical_hr'],
        output_names=['thermal_sr'],
        dynamic_axes={
            'thermal_lr': {0: 'batch_size', 2: 'height', 3: 'width'},
            'optical_hr': {0: 'batch_size', 2: 'height', 3: 'width'},
            'thermal_sr': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"Model exported to: {output_path}")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(str(output_path))
        
        # Run inference test
        ort_inputs = {
            'thermal_lr': thermal_input.numpy(),
            'optical_hr': optical_input.numpy()
        }
        
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"ONNX Runtime test passed! Output shape: {ort_outputs[0].shape}")
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(thermal_input, optical_input)
        
        max_diff = abs(torch_output.numpy() - ort_outputs[0]).max()
        print(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✓ ONNX export successful with high precision!")
        else:
            print("⚠ ONNX export completed but with some precision loss")
            
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False
    
    return True

def create_onnx_inference_wrapper(onnx_path: str, output_path: str):
    """Create a Python wrapper for ONNX inference."""
    
    wrapper_code = f'''
import onnxruntime as ort
import numpy as np
from typing import Tuple

class ThermalSRONNXInference:
    """ONNX inference wrapper for thermal super-resolution."""
    
    def __init__(self, model_path: str = "{onnx_path}"):
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Loaded ONNX model: {{model_path}}")
        print(f"Input names: {{self.input_names}}")
        print(f"Output names: {{self.output_names}}")
    
    def predict(self, 
                thermal_lr: np.ndarray, 
                optical_hr: np.ndarray) -> np.ndarray:
        """Run inference on thermal and optical inputs."""
        
        # Ensure correct input format
        if thermal_lr.ndim == 3:
            thermal_lr = thermal_lr[np.newaxis, ...]
        if optical_hr.ndim == 3:
            optical_hr = optical_hr[np.newaxis, ...]
        
        # Prepare inputs
        inputs = {{
            self.input_names[0]: thermal_lr.astype(np.float32),
            self.input_names[1]: optical_hr.astype(np.float32)
        }}
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        return outputs[0]
    
    def get_input_shapes(self) -> Tuple[tuple, tuple]:
        """Get expected input shapes."""
        thermal_shape = self.session.get_inputs()[0].shape
        optical_shape = self.session.get_inputs()[1].shape
        return thermal_shape, optical_shape

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize inference
    inference = ThermalSRONNXInference()
    
    # Get input shapes
    thermal_shape, optical_shape = inference.get_input_shapes()
    print(f"Expected thermal shape: {{thermal_shape}}")
    print(f"Expected optical shape: {{optical_shape}}")
    
    # Create dummy inputs for testing
    thermal_dummy = np.random.randn(1, 1, 64, 64).astype(np.float32)
    optical_dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
    
    # Run inference
    result = inference.predict(thermal_dummy, optical_dummy)
    print(f"Output shape: {{result.shape}}")
    print("ONNX inference test completed successfully!")
'''
    
    with open(output_path, 'w') as f:
        f.write(wrapper_code)
    
    print(f"ONNX inference wrapper saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Export thermal SR model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for ONNX model')
    parser.add_argument('--thermal-size', type=int, nargs=2, default=[64, 64],
                       help='Thermal input size (height width)')
    parser.add_argument('--optical-size', type=int, nargs=2, default=[128, 128],
                       help='Optical input size (height width)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--create-wrapper', action='store_true',
                       help='Create Python inference wrapper')
    
    args = parser.parse_args()
    
    # Prepare input shapes
    thermal_shape = (args.batch_size, 1, args.thermal_size[0], args.thermal_size[1])
    optical_shape = (args.batch_size, 3, args.optical_size[0], args.optical_size[1])
    
    # Export to ONNX
    success = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=thermal_shape,
        optical_shape=optical_shape,
        opset_version=args.opset_version
    )
    
    if success and args.create_wrapper:
        wrapper_path = Path(args.output).with_suffix('.py')
        create_onnx_inference_wrapper(args.output, wrapper_path)

if __name__ == '__main__':
    main()
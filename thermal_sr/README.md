# ISRO Thermal SR Lab - PyTorch Implementation

**Optics-Guided, Physics-Grounded Thermal Super-Resolution Pipeline**

Production-ready PyTorch implementation with three interchangeable backends for thermal imagery super-resolution using optical guidance and physics-based constraints.

## Features

- **Three SR Backends**: Alignment-Fusion CNN, Guidance-Disentanglement, Swin Transformer
- **Physics Constraints**: Atmospheric correction, emissivity handling, energy-balance regularization
- **Scalable Inference**: Tiled processing for large GeoTIFF scenes
- **Comprehensive Evaluation**: PSNR, SSIM, RMSE(K) with per-class analysis
- **Production Ready**: ONNX export, CLI/API, deterministic seeding

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python scripts/train.py --config configs/train_cnn.yaml

# Run inference on tiles
python scripts/tile_infer.py --checkpoint runs/best.pt --scene data/sample_scene

# Evaluate results
python scripts/eval.py --checkpoint runs/best.pt --data data/test
```

## Model Backends

### 1. Alignment-Fusion CNN
Multi-branch network with optical guidance fusion via attention gates.
```bash
python scripts/train.py --config configs/train_cnn.yaml
```

### 2. Guidance-Disentanglement
Explicit structure/texture disentanglement with contrastive regularization.
```bash
python scripts/train.py --config configs/train_disentangle.yaml
```

### 3. Swin Transformer
Hierarchical vision transformer with shifted window attention.
```bash
python scripts/train.py --config configs/train_swin.yaml
```

## Physics Modules

- **Atmospheric Correction**: Radiance to surface temperature conversion
- **Emissivity Handling**: Class-wise emissivity tables and maps
- **Energy Balance**: Thermal consistency constraints

## Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **RMSE(K)**: Root Mean Square Error in Kelvin
- **Per-class**: Land cover stratified metrics

## Repository Structure

```
thermal_sr/
├── src/
│   ├── align/          # Feature-based registration
│   ├── data/           # Dataset and datamodule classes
│   ├── models/         # Three SR backends
│   ├── physics/        # Atmospheric/emissivity modules
│   ├── losses/         # Physics-aware loss functions
│   ├── metrics/        # Evaluation metrics
│   └── utils/          # Utilities
├── scripts/            # Training and inference scripts
├── configs/            # YAML configurations
├── tests/              # Unit tests
└── sample_data/        # Synthetic test data
```

## License

MIT License - ISRO Thermal SR Lab
# Quick Start Guide - ISRO Thermal SR Lab

## Installation

```bash
# Clone repository
git clone https://github.com/isro/thermal-sr.git
cd thermal-sr

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Or use make
make install
```

## Generate Synthetic Data

```bash
# Generate test dataset
python sample_data/generate_synthetic.py \
    --output-dir sample_data/synthetic \
    --num-scenes 20 \
    --hr-size 512 512 \
    --scale 2

# Or use make
make data
```

## Train Models

### CNN Model (Fastest)
```bash
python scripts/train.py --config configs/train_cnn.yaml

# Or use make
make train-cnn
```

### Disentanglement Model (Best Fidelity)
```bash
python scripts/train.py --config configs/train_disentangle.yaml

# Or use make
make train-dis
```

### Swin Transformer (Highest Quality)
```bash
python scripts/train.py --config configs/train_swin.yaml

# Or use make
make train-swin
```

## Run Inference

### Single Scene
```bash
python scripts/tile_infer.py \
    --checkpoint runs/cnn_thermal_sr/checkpoints/best_model.pt \
    --optical data/optical.tif \
    --thermal data/thermal.tif \
    --output results/sr_result.tif
```

### With ROI
```bash
python scripts/tile_infer.py \
    --checkpoint runs/best.pt \
    --optical data/optical.tif \
    --thermal data/thermal.tif \
    --output results/sr_result.tif \
    --roi-bounds -120.5 35.2 -120.3 35.4
```

## Evaluate Results

```bash
python scripts/eval.py \
    --checkpoint runs/best.pt \
    --data sample_data/synthetic \
    --output results/evaluation

# Or use make
make eval MODEL=runs/best.pt DATA=sample_data/synthetic
```

## Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint runs/best.pt \
    --output models/thermal_sr.onnx \
    --create-wrapper

# Or use make
make export MODEL=runs/best.pt OUTPUT=models/thermal_sr.onnx
```

## Complete Workflow

```bash
# One-command setup and training
make quickstart

# This runs:
# 1. make install
# 2. make data  
# 3. make train-cnn
```

## Expected Results

After training on synthetic data:

**CNN Model:**
- PSNR: ~26-30 dB
- SSIM: ~0.80-0.90
- RMSE: ~1.5-2.5 K
- Training time: ~2-4 hours (GPU)

**Disentanglement Model:**
- PSNR: ~28-32 dB
- SSIM: ~0.85-0.92
- RMSE: ~1.2-2.0 K
- Training time: ~4-6 hours (GPU)

**Swin Transformer:**
- PSNR: ~30-35 dB
- SSIM: ~0.88-0.95
- RMSE: ~1.0-1.8 K
- Training time: ~6-10 hours (GPU)

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
batch_size: 8  # Instead of 16

# Or use smaller tile size for inference
--tile-size 256  # Instead of 512
```

### Poor Results
```bash
# Check data quality
python -c "
import rasterio
with rasterio.open('data/thermal.tif') as src:
    print(f'Shape: {src.shape}')
    print(f'CRS: {src.crs}')
    print(f'Bounds: {src.bounds}')
"

# Verify alignment
python src/align/registration.py \
    --optical data/optical.tif \
    --thermal data/thermal.tif \
    --output aligned.tif
```

### Slow Training
```bash
# Enable mixed precision
mixed_precision: true

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
python scripts/train.py --config configs/train_cnn.yaml
```

## Next Steps

1. **Real Data**: Replace synthetic data with real optical-thermal pairs
2. **Fine-tuning**: Use pre-trained weights for domain adaptation
3. **Production**: Deploy ONNX models with gRPC/REST API
4. **Optimization**: Quantize models for edge deployment

## Configuration Examples

### High Quality (Research)
```yaml
model:
  type: "swin"
  params:
    scale: 2
    embed_dim: 128
    depths: [2, 2, 18, 2]

training:
  epochs: 200
  batch_size: 4
  
loss:
  weights:
    physics: 1.0  # High physics weight
```

### Fast Inference (Production)
```yaml
model:
  type: "cnn"
  params:
    scale: 2
    num_channels: 32  # Smaller model

inference:
  tile_size: 256
  overlap: 32
```

### Memory Efficient
```yaml
training:
  batch_size: 4
  mixed_precision: true
  grad_clip: 0.5

inference:
  tile_size: 128
  overlap: 16
```
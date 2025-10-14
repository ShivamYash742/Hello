# Model Cards - ISRO Thermal SR Lab

## Alignment-Fusion CNN

### Architecture
Multi-branch convolutional network with optical guidance fusion via attention gates.

**Key Components:**
- Guidance Extractor: Extracts edge and luminance features from optical RGB
- Thermal Feature Extractor: Processes LR thermal input
- Attention Gates: Fuse thermal and optical features selectively
- Upsampling: Sub-pixel convolution for 2×/4× super-resolution

**Parameters:** ~2.1M (scale=2, channels=64)

### Training Configuration
```yaml
model:
  type: "cnn"
  params:
    scale: 2
    num_channels: 64

loss:
  weights:
    content: 1.0
    ssim: 0.2
    edge: 0.1
    tv: 1e-4
    physics: 0.5

optimizer:
  type: "adamw"
  lr: 2e-4
```

### Performance Metrics
- **PSNR:** 28.5 ± 2.1 dB
- **SSIM:** 0.85 ± 0.08
- **RMSE:** 1.8 ± 0.4 K
- **Inference Speed:** ~15ms per 512×512 tile (GPU)

### Use Cases
- Real-time thermal enhancement
- Edge-preserving super-resolution
- Balanced performance/speed trade-off

---

## Guidance-Disentanglement Model

### Architecture
Explicit structure/texture disentanglement with contrastive regularization to prevent optical texture leakage.

**Key Components:**
- Structure-Texture Disentangler: Separates optical guidance components
- Contrastive Gates: Selectively inject structure while suppressing texture
- Residual Refinement: Multi-stage feature enhancement
- Physics Consistency: Enhanced thermal fidelity constraints

**Parameters:** ~3.8M (scale=2, channels=64)

### Training Configuration
```yaml
model:
  type: "disentangle"
  params:
    scale: 2
    num_channels: 64

loss:
  weights:
    content: 1.0
    ssim: 0.2
    edge: 0.15
    tv: 1e-4
    physics: 0.7  # Higher physics weight

optimizer:
  type: "adamw"
  lr: 1.5e-4  # Lower LR for stability
```

### Performance Metrics
- **PSNR:** 29.8 ± 1.9 dB
- **SSIM:** 0.88 ± 0.06
- **RMSE:** 1.5 ± 0.3 K
- **Texture Leakage Score:** 0.12 ± 0.05 (lower is better)
- **Inference Speed:** ~25ms per 512×512 tile (GPU)

### Use Cases
- High-fidelity thermal reconstruction
- Applications requiring minimal texture leakage
- Scientific/quantitative thermal analysis

---

## Swin Transformer

### Architecture
Hierarchical vision transformer with shifted window attention and optical guidance integration.

**Key Components:**
- Patch Embedding: Converts thermal patches to token sequences
- Optical Guidance Encoder: Global context from optical imagery
- Swin Transformer Blocks: Hierarchical attention with window shifting
- Pixel Shuffle Upsampling: Learnable upsampling for SR

**Parameters:** ~12.5M (embed_dim=96, depths=[2,2,6,2])

### Training Configuration
```yaml
model:
  type: "swin"
  params:
    scale: 2
    patch_size: 4
    embed_dim: 96
    depths: [2, 2, 6, 2]
    num_heads: [3, 6, 12, 24]
    window_size: 7

loss:
  weights:
    content: 1.0
    ssim: 0.3  # Higher SSIM weight
    edge: 0.1
    tv: 5e-5
    physics: 0.4

optimizer:
  type: "adamw"
  lr: 1e-4
  weight_decay: 5e-2  # Higher weight decay
```

### Performance Metrics
- **PSNR:** 31.2 ± 1.6 dB
- **SSIM:** 0.91 ± 0.04
- **RMSE:** 1.2 ± 0.2 K
- **Long-range Consistency:** Excellent
- **Inference Speed:** ~45ms per 384×384 tile (GPU)

### Use Cases
- Highest quality thermal super-resolution
- Large-scale scene processing
- Research applications requiring state-of-the-art performance

---

## Model Selection Guidelines

### Choose CNN when:
- Real-time processing required
- Balanced performance/efficiency needed
- Limited computational resources

### Choose Disentanglement when:
- Minimal texture leakage critical
- Quantitative thermal analysis
- Scientific accuracy paramount

### Choose Swin Transformer when:
- Highest quality results needed
- Computational resources available
- Large-scale scene processing
- Research/benchmark applications
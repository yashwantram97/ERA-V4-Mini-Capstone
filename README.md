# ImageNet Training Pipeline - Medium Recipe 🌶️🌶️

A highly optimized PyTorch Lightning-based ImageNet training pipeline that achieves **78-79.5% accuracy** with advanced training techniques and efficient resource utilization.

## 🎯 Overview

This project implements state-of-the-art training techniques for ImageNet classification using ResNet-50, combining multiple optimization methods to achieve competitive accuracy while minimizing training time and cost.

## 📊 Datasets

This project supports two dataset variants:

### ImageNet-Mini (Local Training)
- **Dataset:** [ImageNet-Mini-1000](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
- **Size:** ~35,000 images (1000 classes)
- **Use Case:** Local development and experimentation on consumer hardware
- **Training Time:** 2-3 hours on MacBook M4 Pro
- **Download:** Available on Kaggle

### ImageNet-1K (Full Training)
- **Dataset:** [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
- **Size:** 1.2M training images, 50K validation images (1000 classes)
- **Use Case:** Production training on cloud infrastructure
- **Training Time:** 10-11 hours on AWS p3.16xlarge (8x V100 GPUs)
- **Download:** Available through Kaggle competition

## ✨ Key Features

### Speed-Up Techniques
1. **BlurPool** - Antialiased downsampling for shift-invariance (+0.5-1% accuracy)
2. **FixRes** - Fixed resolution fine-tuning at higher resolution (+1-2% accuracy)
3. **Label Smoothing** - Prevents overconfident predictions (+0.5-1% accuracy)
4. **Progressive Resizing** - 128px → 224px → 288px (30-40% faster training)
5. **MixUp** - Data augmentation through linear interpolation (+1-2% accuracy)
6. **SAM** - Sharpness Aware Minimization optimizer (+1-2% accuracy)

### Additional Optimizations
- **Channels Last Memory Format** - 10-30% faster on modern GPUs
- **Dynamic Batch Sizing** - Optimizes GPU utilization across resolutions
- **Multi-GPU Support** - Distributed training with PyTorch Lightning

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 78.1% - 79.5% |
| **Training Time (p3.16xlarge)** | 10-11 hours |
| **Training Time (M4 Pro - Mini)** | 2-3 hours |
| **Model** | ResNet-50 with BlurPool |
| **Parameters** | ~25M |
| **Training Cost (AWS)** | ~$50-70 per run |

### Training Timeline

| Epoch | Resolution | Expected Val Accuracy | Phase |
|-------|-----------|---------------------|-------|
| 0-5   | 128px     | 35% → 68%          | Rapid learning |
| 6-10  | 128px     | 68% → 72%          | Low-res convergence |
| 10-20 | 224px     | 70% → 74%          | Resolution jump recovery |
| 40-60 | 224px     | 75% → 77%          | Steady improvement |
| 70-85 | 224px     | 77% → 78%          | Near convergence |
| 85-90 | 288px     | 78% → 79%+         | FixRes boost |

## ⚙️ Hardware-Specific Configurations

This project includes **three optimized hardware profiles** that automatically configure batch sizes, workers, and training parameters for your hardware:

### Available Profiles

| Profile | Hardware | GPUs | Batch Size | Workers | Use Case |
|---------|----------|------|------------|---------|----------|
| `local` | MacBook M4 Pro | 1 (MPS) | 32-64 | 4 | Development & Testing |
| `g5d` | AWS g5d.12xlarge | 4x A10G | 256-512 | 12 | Cost-Effective Training |
| `p3` | AWS p3.16xlarge | 8x V100 | 256-768 | 16 | Production Training |

### Quick Start

```bash
# Train on your local MacBook M4 Pro
python train.py --config local

# Train on AWS g5d.12xlarge (4x A10G)
python train.py --config g5d

# Train on AWS p3.16xlarge (8x V100)
python train.py --config p3

# List all available configurations
python train.py --list-configs
```

Each profile includes:
- ✅ Optimized batch sizes for progressive resizing
- ✅ Appropriate worker counts for data loading
- ✅ Hardware-specific precision settings
- ✅ Multi-GPU strategy configuration
- ✅ Memory-optimized settings

For detailed configuration options, see [`configs/README.md`](configs/README.md).

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- `torch`
- `torchvision`
- `pytorch_lightning`
- `pytorch_optimizer`
- `torchmetrics`
- `antialiased-cnns`

### Dataset Setup

1. **Download the dataset:**
   - For local training: Download [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
   - For full training: Download [ImageNet-1K](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)

2. **Organize the dataset:**
   ```
   imagenet-mini/  (or imagenet/)
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ...
   ```

### Training

#### Using Configuration Profiles (Recommended)

The easiest way to train is using the hardware-specific configuration system:

```bash
# Basic training
python train.py --config local   # For MacBook M4 Pro
python train.py --config g5d     # For AWS g5d.12xlarge
python train.py --config p3      # For AWS p3.16xlarge

# Advanced options
python train.py --config g5d --use-sam                    # Use SAM optimizer
python train.py --config g5d --lr 0.001                   # Custom learning rate
python train.py --config p3 --resume path/to/last.ckpt   # Resume training
python train.py --config p3 --lr 0.005 --use-sam         # Combine options

# Find optimal learning rate
python find_lr.py --config local
python find_lr.py --config g5d --lr 0.0001  # Custom starting LR
```

#### Using Jupyter Notebooks (Alternative)

```bash
# Local training (ImageNet-Mini on M4 Pro)
jupyter notebook notebook-local.ipynb

# Production training (Full ImageNet on AWS p3.16xlarge)
jupyter notebook notebook-p3.16xlarge.ipynb
```

## 📁 Project Structure

```
.
├── train.py                    # Main training script with config support
├── find_lr.py                  # Learning rate finder with config support
├── config.py                   # Configuration loader (backward compatible)
├── configs/                    # Hardware-specific configurations
│   ├── README.md              # Detailed configuration documentation
│   ├── local_config.py        # MacBook M4 Pro settings
│   ├── g5d_config.py          # AWS g5d.12xlarge settings
│   ├── p3_config.py           # AWS p3.16xlarge settings
│   └── config_manager.py      # Configuration management system
├── src/                        # Source code
│   ├── models/                # Model architectures
│   ├── data_modules/          # Data loading and preprocessing
│   ├── callbacks/             # Training callbacks
│   └── utils/                 # Utility functions
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── notebook-local.ipynb   # Local training (M4 Pro + ImageNet-Mini)
│   └── notebook-p3.16xlarge.ipynb  # Production (8x V100 + ImageNet-1K)
├── docs/
│   └── recipes.md             # Detailed explanation of techniques
├── logs/                      # Training logs and checkpoints
└── dataset/                   # Dataset directory
    └── imagenet-mini/
        ├── train/
        └── val/
```

## 📚 Documentation

- **[configs/README.md](configs/README.md)** - Complete guide to hardware configurations
- **[docs/recipes.md](docs/recipes.md)** - In-depth explanation of all training techniques with papers and code examples

## 🛠️ Hardware Configurations

### AWS p3.16xlarge (Production)
- **GPUs:** 8x NVIDIA V100 (16GB VRAM each)
- **vCPUs:** 64
- **Batch Size:** 4,096 → 2,560 → 2,048 (total across GPUs)
- **Training Time:** ~10-11 hours for full ImageNet-1K
- **Best For:** Full ImageNet training

### MacBook M4 Pro (Local/Development)
- **GPU:** 1x Apple Silicon (MPS)
- **Workers:** 4 data loading workers
- **Batch Size:** 128 → 64 → 32
- **Training Time:** ~2-3 hours for ImageNet-Mini
- **Best For:** Development, experimentation, ImageNet-Mini

## 📖 References

1. Zhang, R. (2019). [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486). ICML.
2. Touvron, H., et al. (2019). [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423). NeurIPS.
3. Szegedy, C., et al. (2016). [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567). CVPR.
4. Zhang, H., et al. (2017). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR.
5. Foret, P., et al. (2020). [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412). ICLR.
6. Howard, J., & Gugger, S. (2020). Deep Learning for Coders with fastai and PyTorch. O'Reilly.

## 🎓 Learning Resources

This project was developed as part of the ERA (Extensive Research and Applications) program, demonstrating practical implementation of cutting-edge deep learning optimization techniques.

## 📝 License

This project is for educational purposes.

---

**Note:** Adjust batch sizes and worker counts based on your hardware capabilities. The provided configurations are optimized for AWS p3.16xlarge and MacBook M4 Pro respectively.


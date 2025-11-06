# ImageNet Training Pipeline - Medium Recipe 🌶️🌶️

A highly optimized PyTorch Lightning-based ImageNet training pipeline that achieves **77.45% top-1 accuracy** on ImageNet-1K with advanced training techniques and efficient resource utilization, exceeding the 75% target.

## 🎯 Overview

This project implements state-of-the-art training techniques for ImageNet classification using ResNet-50, combining multiple optimization methods to achieve competitive accuracy while minimizing training time and cost.

**🎉 Training Results: Achieved 77.45% top-1 accuracy on ImageNet-1K, exceeding the 75% target by 2.45%!**

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
- **Final Accuracy:** 77.45% top-1 (Target: 75% ✅)
- **Training Time:** 356 minutes (~6 hours)
- **Download:** Available through Kaggle competition

## ✨ Key Features

### Speed-Up Techniques
1. **BlurPool** - Antialiased downsampling for shift-invariance (+0.5-1% accuracy)
2. **FixRes** - Fixed resolution fine-tuning at higher resolution (+1-2% accuracy)
3. **Label Smoothing** - Prevents overconfident predictions (+0.5-1% accuracy)
4. **Progressive Resizing** - 128px → 224px → 288px (30-40% faster training) (optional)
5. **MixUp** - Data augmentation through linear interpolation (+1-2% accuracy) (optional)
6. **RandomErase** - Randomly selects a rectangle region in a torch.Tensor image and erases its pixels (+1-2% accuracy)

### Additional Optimizations
- **Channels Last Memory Format** - 10-30% faster on modern GPUs
- **Multi-GPU Support** - Distributed training with PyTorch Lightning

## 📈 Performance Metrics

### Final Training Results ✅

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **77.45%** (top-1) |
| **Target Accuracy** | 75% ✅ **EXCEEDED by 2.45%** |
| **Training Time (p3.16xlarge)** | 356 minutes (~6 hours) |
| **Total Epochs** | 90 |
| **Best Model Checkpoint** | Epoch 90 |
| **Model** | ResNet-50 with BlurPool |
| **Parameters** | 25,576,264 (~25M) |
| **Training Cost (AWS p4.24xlarge)** | ~$15-25 per run |

### Training Timeline (Actual Results)

| Epoch Range | Resolution | Val Accuracy Progression | Phase |
|-------------|-----------|------------------------|-------|
| 1-10   | 128px     | 4.10% → 21.46%          | Initial learning |
| 11-30  | 224px     | 26.00% → 56.04%         | Rapid improvement |
| 31-60  | 224px     | 54.17% → 64.12%         | Steady convergence |
| 61-80  | 224px     | 72.80% → 74.04%         | Near target |
| 81-90  | 288px     | 75.07% → **77.45%**     | FixRes fine-tuning ⭐ |

**Key Milestones:**
- 🎯 **Epoch 81:** Achieved **75.07%** - Target reached!
- ⭐ **Epoch 82:** **76.15%** - Continued improvement
- 🏆 **Epochs 89-90:** **77.45%** - Final best accuracy

### Detailed Training Logs

📊 **Complete training logs are available in [README-logs.md](README-logs.md)** with epoch-by-epoch metrics from epochs 1-90.

## ⚙️ Hardware-Specific Configurations

This project includes **three optimized hardware profiles** that automatically configure batch sizes, workers, and training parameters for your hardware:

### Available Profiles

| Profile | Hardware | GPUs | Batch Size | Workers | Use Case |
|---------|----------|------|------------|---------|----------|
| `local` | MacBook M4 Pro | 1 (MPS) | 32-64 | 4 | Development & Testing |
| `g5` | AWS g5.12xlarge | 4x A10G | 256-512 | 12 | Cost-Effective Training |
| `p4` | AWS p4.24xlarge | 8x V100 | 256-768 | 16 | Production Training |

### Quick Start

```bash
# Train on your local MacBook M4 Pro
python train.py --config local

# Train on AWS g5.12xlarge (4x A10G)
python train.py --config g5

# Train on AWS p4d.24xlarge (8x V100)
python train.py --config p4

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
uv sync
```

Required packages:
- `torch`
- `torchvision`
- `pytorch_lightning`
- `pytorch_optimizer`
- `torchmetrics`
- `timm`

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
python train.py --config g5     # For AWS g5.12xlarge
python train.py --config p3      # For AWS p3.16xlarge

# Advanced options
python train.py --config g5 --lr 0.001                   # Custom learning rate
python train.py --config p3 --resume path/to/last.ckpt   # Resume training
python train.py --config p3 --lr 0.005                   # Combine options

# Find optimal learning rate
python find_lr.py --config local
python find_lr.py --config g5 --lr 0.0001  # Custom starting LR
```

#### Using Jupyter Notebooks (Used for experimentation)

```bash
# Local training (ImageNet-Mini on M4 Pro)
jupyter notebook notebook-local.ipynb

# Production training (Full ImageNet on AWS p3.16xlarge)
jupyter notebook notebook-p3.16xlarge.ipynb
```

## 📁 Project Structure

```
.
├── README.md                    # Main project documentation
├── README-logs.md               # Complete training logs (epochs 1-90)
├── train.py                     # Main training script with config support
├── find_lr.py                   # Learning rate finder with config support
├── config.py                    # Configuration loader (backward compatible)
├── configs/                    # Hardware-specific configurations
│   ├── README.md              # Detailed configuration documentation
│   ├── local_config.py        # MacBook M4 Pro settings
│   ├── g5d_config.py          # AWS g5.12xlarge settings
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

- **[README-logs.md](README-logs.md)** - Complete training logs from epochs 1-90 with detailed metrics
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

1. mosaic resnet experiment - [github link](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/resnet_imagenet)

## 🎓 Learning Resources

This project was developed as part of the ERA (Extensive Research and Applications) program, demonstrating practical implementation of cutting-edge deep learning optimization techniques.

## 📝 License

This project is for educational purposes.

---

**Note:** Adjust batch sizes and worker counts based on your hardware capabilities. The provided configurations are optimized for AWS p4d.24xlarge and MacBook M4 Pro respectively.


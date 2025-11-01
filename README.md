# ImageNet Training Pipeline - Medium Recipe 🌶️🌶️

[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.5+-792ee5?logo=pytorchlightning)](https://lightning.ai/)
[![Python](https://img.shields.io/badge/Python-3.12+-3776ab?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-blue)](LICENSE)

A highly optimized PyTorch Lightning-based ImageNet training pipeline that achieves **78-79.5% top-1 accuracy** using state-of-the-art training techniques and efficient resource utilization. This project demonstrates practical implementation of advanced deep learning optimization methods for large-scale image classification.

---

## 🎯 Overview

This project implements a comprehensive ImageNet training pipeline using ResNet-50, combining multiple cutting-edge optimization techniques to achieve competitive accuracy while minimizing training time and computational cost. The pipeline is designed with hardware-specific configurations for seamless deployment across different environments, from local development to cloud production.

**Key Achievements:**
- ✅ **78-79.5% top-1 accuracy** on ImageNet-1K validation set
- ✅ **30-40% faster training** through progressive resizing
- ✅ **Multi-GPU distributed training** with PyTorch Lightning
- ✅ **Hardware-optimized configurations** for AWS and local machines
- ✅ **Comprehensive documentation** and reproducible results

---

## 👥 Contributors

This project was developed as a collaborative effort by:

1. **Yashwant Ram** - yash97r@gmail.com
2. **Sualeh Qureshi** - sualeh77@gmail.com
3. **Aasim Kureshi** - aasimq776@gmail.com
4. **Aman** - coursesxyz403@gmail.com

This project was developed as part of the **ERA (Extensive & Reimagined AI Program) V4 Mini Capstone Program**.

---

## 📊 Datasets

This project supports two dataset variants for different use cases:

### ImageNet-Mini (Local Training & Development)
- **Dataset:** [ImageNet-Mini-1000](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
- **Size:** ~35,000 training images (1000 classes)
- **Use Case:** Local development, experimentation, and rapid prototyping on consumer hardware
- **Training Time:** 2-3 hours on MacBook M4 Pro
- **Download:** Available on Kaggle

### ImageNet-1K (Full Production Training)
- **Dataset:** [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
- **Size:** 1.2M training images, 50K validation images (1000 classes)
- **Use Case:** Production training on cloud infrastructure (AWS)
- **Training Time:** 10-11 hours on AWS p3.16xlarge (8x V100 GPUs)
- **Download:** Available through Kaggle competition

---

## ✨ Key Features & Techniques

### 🚀 Speed-Up & Optimization Techniques

1. **BlurPool** - Antialiased downsampling for shift-invariance
   - Prevents aliasing artifacts in downsampling layers
   - Improves accuracy by +0.5-1%
   - Reference: [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)

2. **FixRes (Fixed Resolution Fine-tuning)** - Higher resolution fine-tuning
   - Fine-tunes model at higher resolution than training
   - Improves accuracy by +1-2%
   - Reference: [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423)

3. **Progressive Resizing** - Curriculum learning with resolution scheduling
   - Trains at 128px → 224px → 288px progressively
   - Reduces training time by 30-40%
   - Faster initial convergence with smaller images
   - Implemented via callback system with epoch-based scheduling

4. **Label Smoothing** - Regularization technique
   - Prevents overconfident predictions
   - Improves generalization (+0.5-1% accuracy)
   - Smoothing factor: 0.1

5. **MixUp & CutMix** - Advanced data augmentation
   - MixUp: Linear interpolation between sample pairs
   - CutMix: Patch-based augmentation
   - Improves accuracy by +1-2%
   - Reference: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

### ⚡ Additional Optimizations

- **Channels Last Memory Format** - 10-30% faster on modern GPUs
- **Mixed Precision Training (FP16)** - Faster training with reduced memory
- **Multi-GPU Distributed Training** - DDP strategy with PyTorch Lightning
- **OneCycleLR Scheduler** - Optimal learning rate scheduling
- **Synchronized Batch Normalization** - Consistent batch stats across GPUs
- **S3 Remote Storage Support** - Cloud-based logging and checkpointing
- **Advanced Data Augmentation Pipeline** - Color jitter, blur, CoarseDropout

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 78.1% - 79.5% (top-1) |
| **Training Time (p3.16xlarge)** | 10-11 hours |
| **Training Time (M4 Pro - Mini)** | 2-3 hours |
| **Model Architecture** | ResNet-50 with BlurPool |
| **Model Parameters** | ~25M |
| **Training Cost (AWS p3.16xlarge)** | ~$50-70 per full run |
| **Training Cost (AWS g5.12xlarge)** | ~$15-25 per full run |

### 📊 Training Timeline

| Epoch Range | Resolution | Expected Val Accuracy | Training Phase |
|-------------|-----------|---------------------|----------------|
| 0-5 | 128px | 35% → 68% | Rapid learning at low resolution |
| 6-10 | 128px | 68% → 72% | Low-res convergence |
| 10-20 | 224px | 70% → 74% | Resolution jump recovery |
| 20-40 | 224px | 74% → 75% | Steady improvement |
| 40-60 | 224px | 75% → 77% | Main training phase |
| 70-85 | 224px | 77% → 78% | Near convergence |
| 85-90 | 288px | 78% → 79%+ | FixRes fine-tuning boost |

---

## ⚙️ Hardware-Specific Configurations

This project includes **three optimized hardware profiles** that automatically configure batch sizes, workers, and training parameters for your hardware:

### Available Profiles

| Profile | Hardware | GPUs | Batch Size | Workers | Use Case |
|---------|----------|------|------------|---------|----------|
| `local` | MacBook M4 Pro | 1 (MPS) | 32-64 | 4 | Development & Testing |
| `g5` | AWS g5.12xlarge | 4x A10G | 256-512 | 12 | Cost-Effective Training |
| `p3` | AWS p3.16xlarge | 8x V100 | 256-768 | 16 | Production Training |

### Quick Start

```bash
# Train on your local MacBook M4 Pro
python train.py --config local

# Train on AWS g5.12xlarge (4x A10G)
python train.py --config g5

# Train on AWS p3.16xlarge (8x V100)
python train.py --config p3

# List all available configurations
python train.py --list-configs

# Find optimal learning rate
python find_lr.py --config local --runs 3
```

Each profile includes:
- ✅ Optimized batch sizes for progressive resizing stages
- ✅ Appropriate worker counts for efficient data loading
- ✅ Hardware-specific precision settings (FP16/FP32)
- ✅ Multi-GPU strategy configuration (DDP)
- ✅ Memory-optimized settings
- ✅ Pre-configured learning rate schedules

For detailed configuration options, see [`configs/README.md`](configs/README.md).

---

## 🚀 Getting Started

### Prerequisites

Install dependencies using pip or uv:

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv pip install -r requirements.txt
```

**Required Packages:**
- `torch` >= 2.9.0
- `torchvision` >= 0.24.0
- `lightning` >= 2.5.5 (PyTorch Lightning)
- `pytorch-optimizer` >= 3.8.1
- `torchmetrics` >= 1.0.0
- `timm` >= 1.0.20 (for MixUp/CutMix)
- `tensorboard` >= 2.20.0
- `matplotlib` >= 3.10.7
- `numpy` >= 2.3.4
- `pytest` >= 8.0.0 (for testing)

### Dataset Setup

1. **Download the dataset:**
   - For local training: Download [ImageNet-Mini-1000](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) from Kaggle
   - For full training: Download [ImageNet-1K](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) from Kaggle

2. **Organize the dataset:**
   ```
   dataset/
   ├── imagenet-mini/  (or imagenet/)
   │   ├── train/
   │   │   ├── n01440764/
   │   │   │   ├── n01440764_10026.JPEG
   │   │   │   └── ...
   │   │   ├── n01443537/
   │   │   └── ... (1000 classes)
   │   └── val/
   │       ├── n01440764/
   │       ├── n01443537/
   │       └── ... (1000 classes)
   ```

3. **Update dataset paths in config** (if using custom paths):
   - Local: Edit `configs/local_config.py`
   - AWS: Edit `configs/g5_config.py` or `configs/p3_config.py`

### Training

#### Using Configuration Profiles (Recommended)

The easiest way to train is using the hardware-specific configuration system:

```bash
# Basic training
python train.py --config local   # For MacBook M4 Pro
python train.py --config g5      # For AWS g5.12xlarge
python train.py --config p3      # For AWS p3.16xlarge

# Advanced options
python train.py --config g5 --lr 0.001                    # Custom learning rate
python train.py --config p3 --resume path/to/last.ckpt    # Resume from checkpoint
python train.py --config p3 --lr 0.005                    # Combine options

# Find optimal learning rate (runs multiple times for robust results)
python find_lr.py --config local --runs 3
python find_lr.py --config g5 --runs 3 --batch-size 64    # Override batch size
```

#### Using Jupyter Notebooks (Alternative)

For interactive development and experimentation:

```bash
# Local training (ImageNet-Mini on M4 Pro)
jupyter notebook notebooks/notebook-local.ipynb

# Production training (Full ImageNet on AWS p3.16xlarge)
jupyter notebook notebooks/notebook-p3.16xlarge.ipynb
```

---

## 📁 Project Structure

```
ERA-V4-Mini-Capstone/
├── train.py                      # Main training script with config support
├── find_lr.py                    # Learning rate finder (multiple runs for robustness)
├── main.py                       # Main entry point
├── config.py                     # Configuration loader (backward compatible)
│
├── configs/                      # Hardware-specific configurations
│   ├── README.md                # Detailed configuration documentation
│   ├── __init__.py              # Package initialization
│   ├── config_manager.py        # Configuration management system
│   ├── local_config.py          # MacBook M4 Pro settings
│   ├── g5_config.py             # AWS g5.12xlarge settings (4x A10G)
│   └── p3_config.py             # AWS p3.16xlarge settings (8x V100)
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── resnet_module.py     # ResNet-50 Lightning module with BlurPool
│   │
│   ├── data_modules/             # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── imagenet_datamodule.py  # PyTorch Lightning DataModule
│   │   └── imagenet_dataset.py     # Custom ImageNet dataset class
│   │
│   ├── callbacks/                # Training callbacks
│   │   ├── __init__.py
│   │   ├── resolution_schedule_callback.py  # Progressive resizing + FixRes
│   │   └── text_logging_callback.py        # Detailed text logging
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── utils.py             # Transform utilities, path helpers
│       └── lr_finder_utils.py   # Learning rate finder implementation
│
├── notebooks/                     # Jupyter notebooks for experimentation
│   ├── notebook-local.ipynb     # Local training (M4 Pro + ImageNet-Mini)
│   └── notebook-p3.16xlarge.ipynb  # Production (8x V100 + ImageNet-1K)
│
├── docs/                         # Comprehensive documentation
│   ├── BATCH_SIZE_LR_RELATIONSHIP.md
│   ├── BEFORE_AFTER_COMPARISON.md
│   ├── CHANGES_SUMMARY.md
│   ├── CUTMIX_COSINE_IMPLEMENTATION.md
│   ├── CUTMIX_COSINE_TESTS_SUMMARY.md
│   ├── DDP_FIXES_SUMMARY.md
│   ├── FIXRES_IMPLEMENTATION.md
│   ├── FIXRES_IMPLEMENTATION_SUMMARY.md
│   ├── LR_FINDER_GUIDE.md
│   ├── LR_FINDER_SINGLE_GPU.md
│   ├── LR_FINDER_VARIATION_EXPLAINED.md
│   ├── MIXUP_LR_FINDER_GUIDE.md
│   ├── MIXUP_TESTS_README.md
│   ├── MIXUP_TESTS_SUMMARY.md
│   ├── ONECYCLE_FIX.md
│   ├── PROGRESSIVE_RESIZING_COMPOSER.md
│   ├── PROGRESSIVE_RESIZING_UPDATE.md
│   ├── README_FIXES.md
│   ├── S3_IMPLEMENTATION_SUMMARY.md
│   ├── S3_REMOTE_STORAGE_GUIDE.md
│   ├── TRAINING_ANALYSIS_AND_FIXES.md
│   ├── VERIFICATION_GUIDE.md
│   ├── VERIFICATION_SUMMARY.md
│   └── VISUAL_COMPARISON.md
│
├── tests/                         # Test suite
│   ├── README_TESTS.md
│   ├── test_advanced_features.py
│   ├── test_progressive_resume.py
│   ├── test_progressive_schedule.py
│   ├── test_s3_callback.py
│   ├── verify_training_components.py
│   └── verify_visual.py
│
├── logs/                          # Training logs and checkpoints
│   └── [experiment_name]/
│       ├── lightning_logs/        # TensorBoard logs
│       └── checkpoints/          # Model checkpoints
│
├── dataset/                       # Dataset directory (not in repo)
│   └── imagenet-mini/
│       ├── train/
│       └── val/
│
├── pyproject.toml                 # Project dependencies (uv)
├── uv.lock                        # Lock file for dependencies
└── README.md                      # This file
```

---

## 📚 Documentation

### Configuration Guide
- **[configs/README.md](configs/README.md)** - Complete guide to hardware configurations, adding new profiles, and troubleshooting

### Training Techniques
- **[docs/FIXRES_IMPLEMENTATION.md](docs/FIXRES_IMPLEMENTATION.md)** - FixRes implementation details
- **[docs/PROGRESSIVE_RESIZING_COMPOSER.md](docs/PROGRESSIVE_RESIZING_COMPOSER.md)** - Progressive resizing approach
- **[docs/CUTMIX_COSINE_IMPLEMENTATION.md](docs/CUTMIX_COSINE_IMPLEMENTATION.md)** - CutMix and cosine scheduling

### Learning Rate Finding
- **[docs/LR_FINDER_GUIDE.md](docs/LR_FINDER_GUIDE.md)** - Complete guide to learning rate finding
- **[docs/LR_FINDER_SINGLE_GPU.md](docs/LR_FINDER_SINGLE_GPU.md)** - Single GPU LR finder setup
- **[docs/MIXUP_LR_FINDER_GUIDE.md](docs/MIXUP_LR_FINDER_GUIDE.md)** - LR finding with MixUp

### Training Analysis
- **[docs/TRAINING_ANALYSIS_AND_FIXES.md](docs/TRAINING_ANALYSIS_AND_FIXES.md)** - Analysis of training issues and fixes
- **[docs/CHANGES_SUMMARY.md](docs/CHANGES_SUMMARY.md)** - Summary of configuration changes
- **[docs/BEFORE_AFTER_COMPARISON.md](docs/BEFORE_AFTER_COMPARISON.md)** - Performance comparisons

### Cloud & Storage
- **[docs/S3_REMOTE_STORAGE_GUIDE.md](docs/S3_REMOTE_STORAGE_GUIDE.md)** - S3 integration for cloud storage
- **[docs/DDP_FIXES_SUMMARY.md](docs/DDP_FIXES_SUMMARY.md)** - Distributed training fixes

### Testing & Verification
- **[tests/README_TESTS.md](tests/README_TESTS.md)** - Test suite documentation
- **[docs/VERIFICATION_GUIDE.md](docs/VERIFICATION_GUIDE.md)** - Verification procedures

---

## 🛠️ Hardware Configurations

### AWS p3.16xlarge (Production)
- **GPUs:** 8x NVIDIA V100 (16GB VRAM each, Tensor Cores)
- **vCPUs:** 64 (Intel Xeon E5-2686 v4)
- **Memory:** 488GB RAM
- **Network:** 25 Gbps
- **Batch Size:** 256-768 per GPU (2,048-6,144 total)
- **Workers:** 16 (2 per GPU)
- **Training Time:** ~10-11 hours for full ImageNet-1K
- **Best For:** Production training, fastest convergence
- **Cost:** ~$15-25 per full training run

### AWS g5.12xlarge (Cost-Effective)
- **GPUs:** 4x NVIDIA A10G (24GB VRAM each)
- **vCPUs:** 48 (Intel Xeon)
- **Memory:** 192GB RAM
- **Storage:** 1x 3800GB NVMe SSD
- **Batch Size:** 256-512 per GPU (1,024-2,048 total)
- **Workers:** 12 (3 per GPU)
- **Training Time:** ~4-6 hours for full ImageNet-1K
- **Best For:** Balanced cost/performance training
- **Cost:** ~$5-10 per full training run

### MacBook M4 Pro (Local/Development)
- **GPU:** 1x Apple Silicon (MPS - Metal Performance Shaders)
- **CPU:** 12-14 cores (Apple M4 Pro)
- **Memory:** 16-64GB unified memory
- **Batch Size:** 32-64
- **Workers:** 4
- **Training Time:** ~2-3 hours for ImageNet-Mini
- **Best For:** Development, debugging, rapid experimentation
- **Precision:** FP32 (MPS compatibility)

---

## 🔧 Scripts & Tools

### `train.py` - Main Training Script
The primary script for training models with hardware-specific configurations.

**Usage:**
```bash
python train.py --config <profile> [options]
```

**Options:**
- `--config {local,g5,p3}` - Hardware configuration profile (required)
- `--lr FLOAT` - Override learning rate
- `--resume PATH` - Resume from checkpoint
- `--list-configs` - List all available configurations

**Examples:**
```bash
# Basic training
python train.py --config local

# Custom learning rate
python train.py --config g5 --lr 0.001

# Resume training
python train.py --config p3 --resume logs/experiment/checkpoints/last.ckpt
```

### `find_lr.py` - Learning Rate Finder
Robust learning rate finder that runs multiple times and provides statistical analysis.

**Usage:**
```bash
python find_lr.py --config <profile> [options]
```

**Options:**
- `--config {local,g5,p3}` - Hardware configuration profile
- `--runs INT` - Number of LR finder runs (default: 3)
- `--batch-size INT` - Override batch size for LR finding

**Features:**
- Runs multiple times for robust results
- Statistical analysis (mean, median, geometric mean)
- Automatic batch size scaling for multi-GPU configs
- Generates summary plots and recommendations

**Examples:**
```bash
# Default (3 runs)
python find_lr.py --config local

# More runs for higher confidence
python find_lr.py --config g5 --runs 5

# Custom batch size
python find_lr.py --config p3 --batch-size 32
```

---

## 🧪 Testing

Run the test suite to verify all components:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_advanced_features.py

# Run with verbose output
pytest tests/ -v
```

Test coverage includes:
- Progressive resizing schedule
- Resume functionality
- S3 callback integration
- Visual verification
- Training component verification

See [tests/README_TESTS.md](tests/README_TESTS.md) for detailed test documentation.

---

## 📖 Research Papers & References

1. **BlurPool (Antialiasing)**
   - Zhang, R. (2019). [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486). ICML.

2. **FixRes (Fixed Resolution)**
   - Touvron, H., et al. (2019). [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423). NeurIPS.

3. **Label Smoothing**
   - Szegedy, C., et al. (2016). [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567). CVPR.

4. **MixUp**
   - Zhang, H., et al. (2017). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR.

5. **CutMix**
   - Yun, S., et al. (2019). [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899). ICCV.

6. **Progressive Resizing**
   - MosaicML Composer: [Progressive Resizing Method Card](https://docs.mosaicml.com/projects/composer/en/stable/method_cards/progressive_resizing.html)
   - Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with fastai and PyTorch*. O'Reilly.

7. **OneCycleLR**
   - Smith, L. N. (2017). [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186). WACV.
   - Smith, L. N., & Topin, N. (2019). [Super-Convergence: Very Fast Training of Neural Networks](https://arxiv.org/abs/1708.07120).

---

## 🎓 Learning Resources

This project was developed as part of the **ERA (Extensive & Reimagined AI Program) V4 Mini Capstone Program**, demonstrating practical implementation of cutting-edge deep learning optimization techniques in a production-ready codebase.

**Key Learning Outcomes:**
- ✅ Implementation of state-of-the-art training techniques
- ✅ Multi-GPU distributed training with PyTorch Lightning
- ✅ Hardware-optimized configuration management
- ✅ Comprehensive testing and documentation
- ✅ Cloud infrastructure integration (AWS S3)
- ✅ Reproducible research practices

---

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size in config file
- Reduce number of workers
- Enable gradient accumulation
- Use gradient checkpointing

**Slow Data Loading**
- Increase `num_workers` in config
- Use SSD for dataset storage
- Enable `pin_memory` (already enabled)
- Check disk I/O bandwidth

**Poor GPU Utilization**
- Increase batch size (if memory allows)
- Reduce workers if CPU is bottleneck
- Check data loading pipeline
- Ensure mixed precision is enabled

**Training Instability**
- Reduce learning rate
- Increase weight decay
- Enable gradient clipping
- Check for NaN values in logs

For more detailed troubleshooting, see:
- [configs/README.md](configs/README.md) - Configuration troubleshooting
- [docs/TRAINING_ANALYSIS_AND_FIXES.md](docs/TRAINING_ANALYSIS_AND_FIXES.md) - Common training issues

---

## 📝 License

This project is for **educational purposes** only. Please respect the licenses of:
- ImageNet dataset (use for research/education)
- PyTorch and PyTorch Lightning (BSD-style)
- Third-party libraries (see their respective licenses)

---

## 🙏 Acknowledgments

- **PyTorch Lightning** team for the excellent framework
- **TIMM (PyTorch Image Models)** for MixUp/CutMix implementation
- **ERA Program** for providing the learning platform
- **Research community** for the foundational papers

---

## 📧 Contact

For questions, issues, or contributions, please contact:

- **Yashwant Ram** - yash97r@gmail.com
- **Sualeh Qureshi** - sualeh77@gmail.com
- **Aasim Kureshi** - aasimq776@gmail.com
- **Aman** - coursesxyz403@gmail.com

---

**Note:** Adjust batch sizes and worker counts based on your hardware capabilities. The provided configurations are optimized for AWS p3.16xlarge, g5.12xlarge, and MacBook M4 Pro respectively. Always verify your hardware specifications before training.

---

*Last Updated: 2025*

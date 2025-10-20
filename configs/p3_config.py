"""
Configuration for AWS p3.16xlarge Instance

Hardware Specs:
- CPUs: 64 vCPUs (32 cores, Intel Xeon E5-2686 v4)
- GPUs: 8x NVIDIA V100 (16GB HBM2 each)
- Memory: 488GB RAM
- Network: 25 Gbps
- Storage: EBS optimized

Optimizations:
- High batch sizes across 8 GPUs
- Maximum workers for data loading
- Aggressive resolution schedule
- Full mixed precision training
- Optimal for large-scale training
"""

from pathlib import Path
from .base_config import *

# Hardware profile name
PROFILE_NAME = "p3"
PROFILE_DESCRIPTION = "AWS p3.16xlarge - 8x NVIDIA V100 GPUs"

# Dataset paths (AWS EC2 environment)
TRAIN_IMG_DIR = Path("/home/ec2-user/imagenet1k/train")
VAL_IMG_DIR = Path("/home/ec2-user/imagenet1k/val")

# Dataset settings (full ImageNet)
DATASET_SIZE = 1281167  # Full ImageNet-1K train size

# Experiment naming
EXPERIMENT_NAME = "imagenet_p3_training"

# Training settings
EPOCHS = 90
BATCH_SIZE = 256  # Per GPU: 256, Total effective: 256 * 8 = 2048

# DataLoader settings - maximize data throughput
NUM_WORKERS = 16  # 2 workers per GPU (8 GPUs) = 16 total
# V100 is compute-bound, not data-bound, so 2 workers per GPU is sufficient

# Precision settings
PRECISION = "16-mixed"  # V100 has excellent Tensor Core support

# Progressive Resizing + FixRes Schedule
# Aggressive schedule for fast training on 8 V100s
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True),    # Epochs 0-9: 128px, train augs
    10: (224, True),   # Epochs 10-84: 224px, train augs
    85: (288, False),  # Epochs 85-89: 288px, test augs (FixRes)
}

# Early stopping - more patience for full training
EARLY_STOPPING_PATIENCE = 10

# Checkpointing
SAVE_TOP_K = 3  # Save top 3 models
SAVE_LAST = True

# Logging
LOG_EVERY_N_STEPS = 50

# Validation
CHECK_VAL_EVERY_N_EPOCH = 1

# Gradient settings
GRADIENT_CLIP_VAL = 1.0

# Multi-GPU settings
NUM_DEVICES = 8  # Use all 8 V100 GPUs
STRATEGY = "ddp"  # Distributed Data Parallel

# Additional optimizations for p3.16xlarge
# Set these in your training script:
# - Use pin_memory=True in dataloaders (plenty of RAM)
# - Consider using NCCL backend for multi-GPU communication
# - Monitor GPU utilization to ensure you're compute-bound

# Note: Most powerful option, best for production training
# Expected training time: ~2-3 hours for 90 epochs
# Cost: ~$15-25 per full training run
# Recommended for: Final model training, hyperparameter sweeps


"""
Configuration for AWS g5d.12xlarge Instance

Hardware Specs:
- CPUs: 48 vCPUs (24 cores, Intel Xeon)
- GPUs: 4x NVIDIA A10G (24GB GDDR6 each)
- Memory: 192GB RAM
- Storage: 1x 3800GB NVMe SSD

Optimizations:
- Moderate to high batch sizes (A10G has 24GB VRAM)
- Good number of workers (plenty of CPU cores)
- Full resolution schedule for best quality
- Mixed precision for optimal A10G utilization
"""

from pathlib import Path
from .base_config import *

# Hardware profile name
PROFILE_NAME = "g5d"
PROFILE_DESCRIPTION = "AWS g5d.12xlarge - 4x NVIDIA A10G GPUs"

# Dataset paths (AWS EC2 environment)
TRAIN_IMG_DIR = Path("/home/ec2-user/imagenet1k/train")
VAL_IMG_DIR = Path("/home/ec2-user/imagenet1k/val")

# Dataset settings (full ImageNet)
DATASET_SIZE = 1281167  # Full ImageNet-1K train size

# Experiment naming
EXPERIMENT_NAME = "imagenet_g5d_training"

# Training settings
EPOCHS = 90
BATCH_SIZE = 256  # Good starting point for A10G with 24GB
DYNAMIC_BATCH_SIZE = True

# DataLoader settings - plenty of CPU cores available
NUM_WORKERS = 12  # 3 workers per GPU (4 GPUs) = 12 total

# Precision settings
PRECISION = "16-mixed"  # A10G benefits from mixed precision

# Progressive Resizing + FixRes Schedule
# Optimized for 4x A10G GPUs
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True, 512),    # Epochs 0-9: 128px, train augs, BS=512
    10: (224, True, 320),   # Epochs 10-84: 224px, train augs, BS=320
    85: (288, False, 256),  # Epochs 85-89: 288px, test augs (FixRes), BS=256
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
NUM_DEVICES = 4  # Use all 4 A10G GPUs
STRATEGY = "ddp"  # Distributed Data Parallel

# Note: Excellent balance of cost and performance
# Expected training time: ~4-6 hours for 90 epochs
# Cost: ~$5-10 per full training run


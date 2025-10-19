"""
Configuration for MacBook M4 Pro (Local Development)

Hardware Specs:
- CPU: Apple M4 Pro (12-14 cores)
- GPU: Integrated GPU with unified memory
- Memory: 16-64GB unified memory
- Storage: Fast SSD

Optimizations:
- Lower batch sizes to fit in memory
- Fewer workers (CPU cores shared between data loading and training)
- Smaller resolutions in progressive resizing
- Mixed precision for faster training
"""

from .base_config import *

# Hardware profile name
PROFILE_NAME = "local"
PROFILE_DESCRIPTION = "MacBook M4 Pro - Local Development"

# Dataset paths (local environment)
TRAIN_IMG_DIR = PROJECT_ROOT / "dataset" / "imagenet-mini" / "train"
VAL_IMG_DIR = PROJECT_ROOT / "dataset" / "imagenet-mini" / "val"

# Dataset settings
DATASET_SIZE = 35746  # ImageNet-mini train size

# Experiment naming
EXPERIMENT_NAME = "imagenet_local_dev"

# Training settings
EPOCHS = 10
BATCH_SIZE = 32  # Conservative for unified memory
DYNAMIC_BATCH_SIZE = True

# DataLoader settings - fewer workers for M4 Pro
NUM_WORKERS = 4  # M4 Pro has good CPU but shared with training

# Precision settings
PRECISION = "16-mixed"  # Use mixed precision for speed

# Progressive Resizing + FixRes Schedule
# Conservative batch sizes for local development
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True, 64),   # Epochs 0-3: 128px, train augs, BS=64
    4: (224, True, 32),   # Epochs 4-7: 224px, train augs, BS=32
    8: (288, False, 16),  # Epochs 8-9: 288px, test augs (FixRes), BS=16
}

# Early stopping for faster iteration during development
EARLY_STOPPING_PATIENCE = 3

# Checkpointing
SAVE_TOP_K = 1  # Save only best model to save disk space
SAVE_LAST = True

# Logging
LOG_EVERY_N_STEPS = 10  # More frequent logging for debugging

# Validation
CHECK_VAL_EVERY_N_EPOCH = 1

# Gradient settings
GRADIENT_CLIP_VAL = 0.5

# Note: Good for quick experiments and debugging
# Expected training time: ~20-30 minutes for 10 epochs


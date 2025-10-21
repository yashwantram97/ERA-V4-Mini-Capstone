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

from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Hardware profile name
PROFILE_NAME = "local"
PROFILE_DESCRIPTION = "MacBook M4 Pro - Local Development"

# Dataset paths (local environment)
TRAIN_IMG_DIR = PROJECT_ROOT / "dataset" / "imagenet-mini" / "train"
VAL_IMG_DIR = PROJECT_ROOT / "dataset" / "imagenet-mini" / "val"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset settings
DATASET_SIZE = 130000  # ImageNet-mini train size
NUM_CLASSES = 100
INPUT_SIZE = (1, 3, 224, 224)

# Normalization constants (ImageNet standard)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Experiment naming
EXPERIMENT_NAME = "imagenet_local_dev"

# Training settings
EPOCHS = 60
BATCH_SIZE = 64  # Optimized for M4 Pro MPS with mixed precision
LEARNING_RATE = 0.022  # Found with LR finder
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = 'one_cycle_policy'

# DataLoader settings - fewer workers for M4 Pro
NUM_WORKERS = 4  # M4 Pro has good CPU but shared with training

# Precision settings
PRECISION = "16-mixed"  # Use mixed precision for speed

# Progressive Resizing + FixRes Schedule
# Optimized for 60 epochs total
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True),    # Epochs 0-9: 128px, train augs (17% - fast initial learning)
    10: (224, True),   # Epochs 10-49: 224px, train augs (67% - main training phase)
    50: (288, False),  # Epochs 50-59: 288px, test augs (17% - FixRes fine-tuning)
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

# LR Finder settings
LR_FINDER_KWARGS = {
    'start_lr': 1e-7,
    'end_lr': 10,
    'num_iter': 100,
    'step_mode': 'exp'
}

# OneCycle scheduler settings
ONECYCLE_KWARGS = {
    'lr_strategy': 'manual',  # 'conservative', 'manual'
    'pct_start': 0.2,
    'anneal_strategy': 'cos',
    'div_factor': 100.0,
    'final_div_factor': 1000.0
}

# MixUp/CutMix settings (timm implementation)
MIXUP_KWARGS = {
    'mixup_alpha': 0.2,      # MixUp alpha (0.0 = disabled, 0.2-1.0 recommended)
    'cutmix_alpha': 0.0,     # CutMix alpha (0.0 = disabled)
    'cutmix_minmax': None,   # CutMix min/max ratio
    'prob': 1.0,             # Probability of applying mixup/cutmix
    'switch_prob': 0.5,      # Probability of switching to cutmix when both enabled
    'mode': 'batch',         # How to apply mixup/cutmix ('batch', 'pair', 'elem')
    'label_smoothing': 0.1,  # Label smoothing (matches training_step)
}

# Note: Good for quick experiments and debugging
# Expected training time: ~20-30 minutes for 10 epochs


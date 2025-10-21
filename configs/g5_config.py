"""
Configuration for AWS g5.12xlarge Instance

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

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Hardware profile name
PROFILE_NAME = "g5"
PROFILE_DESCRIPTION = "AWS g5.12xlarge - 4x NVIDIA A10G GPUs"

# Dataset paths (AWS EC2 environment)
TRAIN_IMG_DIR = Path("/mnt/nvme-instance/ImageNet100/train")
VAL_IMG_DIR = Path("/mnt/nvme-instance/ImageNet100/val")

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset settings (full ImageNet)
DATASET_SIZE = 130000  # Full ImageNet-1K train size
NUM_CLASSES = 100
INPUT_SIZE = (1, 3, 224, 224)

# Normalization constants (ImageNet standard)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Experiment naming
EXPERIMENT_NAME = "imagenet_g5d_training"

# Training settings
EPOCHS = 60
BATCH_SIZE = 128  # Good starting point for A10G with 24GB
LEARNING_RATE = 0.022  # Found with LR finder
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = 'one_cycle_policy'

# DataLoader settings - plenty of CPU cores available
# In DDP mode, each GPU spawns its own workers
# So 4 workers per GPU process × 4 GPUs = 16 total workers
NUM_WORKERS = 4  # 4 workers per GPU process (reasonable for DDP)

# Precision settings
PRECISION = "16-mixed"  # A10G benefits from mixed precision

# Progressive Resizing + FixRes Schedule
# Optimized for 60 epochs on 4x A10G GPUs with batch size 128
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True),    # Epochs 0-9: 128px, train augs (17% - fast initial learning)
    10: (224, True),   # Epochs 10-49: 224px, train augs (67% - main training phase)
    50: (256, False),  # Epochs 50-59: 256px, test augs (17% - FixRes fine-tuning)
}

# Early stopping - more patience for full training
EARLY_STOPPING_PATIENCE = 10

# Checkpointing
SAVE_TOP_K = 1  # Save top 3 models
SAVE_LAST = False

# Logging
LOG_EVERY_N_STEPS = 50

# Validation
CHECK_VAL_EVERY_N_EPOCH = 1

# Gradient settings
GRADIENT_CLIP_VAL = 1.0

# Multi-GPU settings
NUM_DEVICES = 4  # Use all 4 A10G GPUs
STRATEGY = "ddp"  # Distributed Data Parallel

# LR Finder settings
LR_FINDER_KWARGS = {
    'start_lr': 1e-7,
    'end_lr': 10,
    'num_iter': 1000,
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

# Note: Excellent balance of cost and performance
# Expected training time: ~3-4 hours for 60 epochs
# Cost: ~$3-7 per training run


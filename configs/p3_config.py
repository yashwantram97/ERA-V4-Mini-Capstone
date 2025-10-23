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

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Hardware profile name
PROFILE_NAME = "p3"
PROFILE_DESCRIPTION = "AWS p3.16xlarge - 8x NVIDIA V100 GPUs"

# Dataset paths (AWS EC2 environment)
TRAIN_IMG_DIR = Path("/home/ec2-user/imagenet1k/train")
VAL_IMG_DIR = Path("/home/ec2-user/imagenet1k/val")

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset settings (full ImageNet)
DATASET_SIZE = 1281167  # Full ImageNet-1K train size
NUM_CLASSES = 1000
INPUT_SIZE = (1, 3, 224, 224)

# Normalization constants (ImageNet standard)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Experiment naming
EXPERIMENT_NAME = "imagenet_p3_training"

# Training settings
EPOCHS = 60
BATCH_SIZE = 256  # Per GPU: 256, Total effective: 256 * 8 = 2048
LEARNING_RATE = 2.11e-3  # Found with LR finder
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = 'one_cycle_policy'
ACCUMULATE_GRAD_BATCHES = 1
# DataLoader settings - maximize data throughput
NUM_WORKERS = 16  # 2 workers per GPU (8 GPUs) = 16 total
# V100 is compute-bound, not data-bound, so 2 workers per GPU is sufficient

# Precision settings
PRECISION = "16-mixed"  # V100 has excellent Tensor Core support

# Progressive Resizing + FixRes Schedule
# Optimized for 60 epochs on 8x V100 GPUs
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True),    # Epochs 0-9: 128px, train augs (17% - fast initial learning)
    10: (224, True),   # Epochs 10-49: 224px, train augs (67% - main training phase)
    50: (288, False),  # Epochs 50-59: 288px, test augs (17% - FixRes fine-tuning)
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

# Additional optimizations for p3.16xlarge
# Set these in your training script:
# - Use pin_memory=True in dataloaders (plenty of RAM)
# - Consider using NCCL backend for multi-GPU communication
# - Monitor GPU utilization to ensure you're compute-bound

# Note: Most powerful option, best for production training
# Expected training time: ~7-8 hours for 60 epochs
# Cost: ~$35-45 per training run
# Recommended for: Final model training, hyperparameter sweeps


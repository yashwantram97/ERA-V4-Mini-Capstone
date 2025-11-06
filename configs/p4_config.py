"""
Configuration for AWS p4d.24xlarge Instance

Hardware Specs:
- CPUs: 96 vCPUs (48 cores, Intel Xeon Platinum 8275CL)
- GPUs: 8x NVIDIA A100 (40GB HBM2 each)
- Memory: 1152GB RAM
- Network: 400 Gbps (4x 100 Gbps EFA)
- Storage: 8x 1TB NVMe SSD
- NVLink: 600GB/s GPU-GPU bandwidth

Optimizations:
- High batch sizes across 8 A100 GPUs
- Maximum workers for data loading
- Progressive resizing following MosaicML Composer's proven approach
- Full mixed precision training (bf16 support)
- Optimal for large-scale training with superior A100 performance
"""

from pathlib import Path
from src.callbacks import create_progressive_resize_schedule

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Hardware profile name
PROFILE_NAME = "p4"
PROFILE_DESCRIPTION = "AWS p4d.24xlarge - 8x NVIDIA A100 GPUs"

# Dataset paths (AWS EC2 environment)
TRAIN_IMG_DIR = Path("/home/ec2-user/imagenet1k/train")
VAL_IMG_DIR = Path("/home/ec2-user/imagenet1k/val")

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset settings (full ImageNet)
DATASET_SIZE = 1281167  # Full ImageNet-1K train size
NUM_CLASSES = 1000
INPUT_SIZE = (1, 3, 256, 256)

# Normalization constants (ImageNet standard)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Experiment naming
EXPERIMENT_NAME = "imagenet_p4_training"

# Training settings
EPOCHS = 90
BATCH_SIZE = 1024  # Per GPU: 128, Total effective: 128 * 8 = 1024
LEARNING_RATE = 1.024  # Found with LR finder
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = 'one_cycle_policy'
ACCUMULATE_GRAD_BATCHES = 1
# DataLoader settings - maximize data throughput
NUM_WORKERS = 10
# A100 is compute-bound, not data-bound, so 10 workers per GPU is sufficient
S3_DIR="s3://imagenet-resnet-50-erav4/data/"

# Precision settings
PRECISION = "16-mixed"  # A100 has excellent Tensor Core support (can also use bf16-mixed)

# Progressive Resizing + FixRes Schedule
# Combines progressive resizing with FixRes for optimal training efficiency and accuracy
#
# Progressive Resizing Benefits:
# • Faster training at lower resolutions (smaller images = faster computation)
# • Curriculum learning: coarse features first, then fine details
# • ~30% training speedup in early epochs
#
# FixRes Benefits:
# • Addresses train-test distribution mismatch
# • Fine-tunes at higher resolution with minimal augmentation
# • Expected: +1-2% validation accuracy
#
# Schedule breakdown for 90 epochs:
# Phase 1 (Epochs 0-17, 20%): 224px, train mode
#   - Fast training at standard resolution
#   - Strong augmentation for robustness
#
# Phase 2 (Epochs 18-70, 60%): 224px, train mode
#   - Standard training at full resolution
#   - Convergence to optimal weights
#
# Phase 3 (Epochs 71-80, 10%): 256px, train mode
#   - Higher resolution training
#   - Preparing for FixRes fine-tuning
#
# Phase 4 (Epochs 81-89, 10%): 288px, fixres mode
#   - Even higher resolution (288px) for finest details
#   - Minimal augmentation bridges train-test gap
#   - Final accuracy boost
PROG_RESIZING_FIXRES_SCHEDULE = create_progressive_resize_schedule(
    total_epochs=EPOCHS,
    target_size=224,          # Standard ImageNet resolution
    initial_scale=1.0,        # Start at 224px 
    delay_fraction=0.0,       # Start immediately
    finetune_fraction=1.0,    # Progressive throughout
    size_increment=4,         # Round to multiples of 4
    use_fixres=True,          # Enable FixRes for +1-2% accuracy boost
    fixres_size=288,          # Higher resolution for FixRes phase (288px)
    fixres_epochs=10          # Last 10 epochs for FixRes fine-tuning
)

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
NUM_DEVICES = 8  # Use all 8 A100 GPUs
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
# MIXUP_KWARGS = {
#     'mixup_alpha': 0.2,      # MixUp alpha (0.0 = disabled, 0.2-1.0 recommended)
#     'cutmix_alpha': 0.0,     # CutMix alpha (0.0 = disabled)
#     'cutmix_minmax': None,   # CutMix min/max ratio
#     'prob': 1.0,             # Probability of applying mixup/cutmix
#     'switch_prob': 0.5,      # Probability of switching to cutmix when both enabled
#     'mode': 'batch',         # How to apply mixup/cutmix ('batch', 'pair', 'elem')
#     'label_smoothing': 0.1,  # Label smoothing (matches training_step)
# }
MIXUP_KWARGS = None

# Additional optimizations for p4d.24xlarge
# Set these in your training script:
# - Use pin_memory=True in dataloaders (plenty of RAM)
# - Consider using NCCL backend for multi-GPU communication
# - Monitor GPU utilization to ensure you're compute-bound
# - A100s support bfloat16 for better numerical stability if needed

# Note: Most powerful option, best for production training
# Expected training time: ~6 hours for 90 epochs on A100s
# Cost: ~$25-35 per training run (spot pricing)
# Recommended for: Final model training, hyperparameter sweeps



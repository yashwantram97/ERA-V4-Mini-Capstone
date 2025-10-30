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
- Progressive resizing following MosaicML Composer's proven approach
- Mixed precision for optimal A10G utilization
"""

from pathlib import Path
from src.callbacks import create_progressive_resize_schedule

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Hardware profile name
PROFILE_NAME = "g5"
PROFILE_DESCRIPTION = "AWS g5.12xlarge - 4x NVIDIA A10G GPUs"

# Dataset paths (AWS EC2 environment)
TRAIN_IMG_DIR = Path("/mnt/nvme1/imagenet1k/train")
VAL_IMG_DIR = Path("/mnt/nvme1/imagenet1k/val")

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
EXPERIMENT_NAME = "imagenet_g5d_training"

# Training settings
EPOCHS = 90  # Extended for better convergence on ImageNet-1K
BATCH_SIZE = 128  # Good starting point for A10G with 24GB
ACCUMULATE_GRAD_BATCHES = 2
LEARNING_RATE = 0.2  # Reduced for better stability on ImageNet-1K (was 0.2)
WEIGHT_DECAY =  5e-4
SCHEDULER_TYPE = 'one_cycle_policy'
S3_DIR = "s3://imagenet-resnet-50-erav4/data/"

# DataLoader settings - plenty of CPU cores available
# In DDP mode, each GPU spawns its own workers
# So 4 workers per GPU process × 4 GPUs = 16 total workers
NUM_WORKERS = 4  # 4 workers per GPU process (reasonable for DDP)

# Precision settings
PRECISION = "16-mixed"  # A10G benefits from mixed precision

# Progressive Resizing + FixRes Schedule
# This schedule combines progressive resizing with FixRes fine-tuning for optimal accuracy
#
# Progressive Resizing Benefits:
# • Faster training at lower resolutions (smaller images = faster computation)
# • Curriculum learning: model learns coarse features first, then fine details
# • Better convergence compared to training at full resolution throughout
#
# FixRes (Fixed Resolution) Benefits:
# • Addresses train-test distribution mismatch
# • Training uses RandomResizedCrop (random crops), testing uses CenterCrop
# • Fine-tuning at higher resolution with minimal augmentation bridges this gap
# • Expected improvement: +1-2% validation accuracy
#
# Schedule breakdown for 90 epochs:
# Phase 1 (Epochs 0-89): 224px with full training augmentations
#   - Standard training with RandomResizedCrop, ColorJitter, RandomErasing
#   - Learn features with strong data augmentation
#
# Phase 2 (Epochs 81-89, last 10%): 256px with FixRes augmentations
#   - Higher resolution (256px vs 224px) captures finer details
#   - Minimal augmentation (Resize + RandomCrop + Flip only)
#   - Adapts model to test-time distribution
#   - Bridges the train (RandomResizedCrop) vs test (CenterCrop) gap
#
# Note: We removed progressive resizing (144px → 224px) and go straight to 224px
# because the dataset is already optimized and full resolution training is fast enough
PROG_RESIZING_FIXRES_SCHEDULE = create_progressive_resize_schedule(
    total_epochs=EPOCHS,
    target_size=224,          # Standard ImageNet resolution
    initial_scale=1.0,        # Start at full resolution (224px)
    delay_fraction=0.0,       # No delay, start at target size immediately
    finetune_fraction=1.0,    # Train at 224px for most of training
    size_increment=4,         # Round to multiples of 4
    use_fixres=True,          # Enable FixRes for +1-2% accuracy boost
    fixres_size=256,          # Higher resolution for FixRes phase (256px)
    fixres_epochs=20           # Last 9 epochs (10% of 90) for FixRes fine-tuning
)

# Early stopping - more patience for full training
EARLY_STOPPING_PATIENCE = 20

# Checkpointing
SAVE_TOP_K = 1  # Save top 3 models
SAVE_LAST = True

# Logging
LOG_EVERY_N_STEPS = 50

# Validation
CHECK_VAL_EVERY_N_EPOCH = 1

# Gradient settings
GRADIENT_CLIP_VAL = 1.0

# Multi-GPU settings
NUM_DEVICES = 8  # Use all 4 A10G GPUs
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
    'pct_start': 0.3,
    'anneal_strategy': 'cos',
    'div_factor': 50.0,
    'final_div_factor': 1000.0
}

# MixUp/CutMix settings (timm implementation)
# Using balanced augmentation: ColorJitter + RandomErasing + MixUp(0.2)
# This combo is proven for 75%+ targets in 90 epochs
MIXUP_KWARGS = None

# Note: Excellent balance of cost and performance
# Expected training time: ~3-4 hours for 60 epochs
# Cost: ~$3-7 per training run


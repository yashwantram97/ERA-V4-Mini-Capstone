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
LEARNING_RATE = 0.15  # Reduced for better stability on ImageNet-1K (was 0.2)
WEIGHT_DECAY =  5e-4
SCHEDULER_TYPE = 'one_cycle_policy'
S3_DIR = "s3://imagenet-resnet-50-erav4/data/"

# DataLoader settings - plenty of CPU cores available
# In DDP mode, each GPU spawns its own workers
# So 4 workers per GPU process × 4 GPUs = 16 total workers
NUM_WORKERS = 4  # 4 workers per GPU process (reasonable for DDP)

# Precision settings
PRECISION = "16-mixed"  # A10G benefits from mixed precision

# Progressive Resizing Schedule (Improved Approach)
# Updated from MosaicML's original 112px start to better initial resolution:
# - initial_scale = 0.64: Start at 64% resolution (144px for target 224px) - BETTER than 112px
# - delay_fraction = 0.3: Stay at initial scale for first 30% of training (shorter delay)
# - finetune_fraction = 0.3: Train at full resolution for last 30% (longer fine-tune)
# - size_increment = 4: Round sizes to multiples of 4 for alignment
#
# Why 144px instead of 112px?
# • 112px loses too much visual detail for ImageNet classification
# • 144px provides better feature learning from the start
# • Shorter delay phase allows more time at full resolution
# • Longer fine-tune phase improves final accuracy
#
# Schedule breakdown for 60 epochs:
# - Epochs 0-17 (30%): 144px - Better feature learning from start
# - Epochs 18-41 (40%): 144→224px - Progressive curriculum learning
# - Epochs 42-59 (30%): 224px - Extended fine-tune at full resolution
PROG_RESIZING_FIXRES_SCHEDULE = create_progressive_resize_schedule(
    total_epochs=EPOCHS,
    target_size=224,          # Standard ImageNet resolution
    initial_scale=0.65,       # Start at 64% (144px) - IMPROVED from 0.5
    delay_fraction=0.1,       # First 30% at initial scale - IMPROVED from 0.5
    finetune_fraction=0.4,    # Last 30% at full size - IMPROVED from 0.2
    size_increment=4,         # Round to multiples of 4
    use_fixres=True,         # Enable FixRes for +1-2% accuracy boost
    fixres_size=256           # Higher resolution for FixRes phase
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


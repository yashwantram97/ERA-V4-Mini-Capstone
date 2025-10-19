"""
Base configuration shared across all hardware profiles.
"""

from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Note: Dataset paths are defined in each hardware config
# since they differ between local and cloud environments

# Logs directory (same for all configs)
LOGS_DIR = PROJECT_ROOT / "logs"
NUM_CLASSES = 1000
INPUT_SIZE = (1, 3, 224, 224)

# Normalization constants (ImageNet standard)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Training hyperparameters (shared)
LEARNING_RATE = 2.11e-3  # Found with LR finder
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = 'one_cycle_policy'

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


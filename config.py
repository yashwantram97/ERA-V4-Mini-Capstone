"""
MODERN CONFIGURATION SYSTEM

This repository now uses a modern, hardware-specific configuration system.
Choose the appropriate configuration for your hardware:

1. MacBook M4 Pro (local development):
   python train.py --config local

2. AWS g5.12xlarge (4x A10G GPUs):
   python train.py --config g5

3. AWS p3.16xlarge (8x V100 GPUs):
   python train.py --config p3

For more details, see: configs/README.md
"""

# Import the new configuration system for easy access
from configs import get_config, list_configs

# Default to local config for backward compatibility
_default_config = get_config('local')

# Export commonly used values for backward compatibility
PROJECT_ROOT = _default_config.project_root
train_img_dir = _default_config.train_img_dir
val_img_dir = _default_config.val_img_dir
logs_dir = _default_config.logs_dir
mean = _default_config.mean
std = _default_config.std
num_classes = _default_config.num_classes
input_size = _default_config.input_size
learning_rate = _default_config.learning_rate
weight_decay = _default_config.weight_decay
scheduler_type = _default_config.scheduler_type
lr_finder_kwargs = _default_config.lr_finder_kwargs
onecycle_kwargs = _default_config.onecycle_kwargs
epochs = _default_config.epochs
batch_size = _default_config.batch_size
dynamic_batch_size = _default_config.dynamic_batch_size
prog_resizing_fixres_schedule = _default_config.prog_resizing_fixres_schedule
dataset_size = _default_config.dataset_size
num_workers = _default_config.num_workers
experiment_name = _default_config.experiment_name

# Device detection
import torch
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Note: This file provides backward compatibility with the old config system.
# For new code, use: from configs import get_config
# See the docstring at the top of this file for more information.

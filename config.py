from pathlib import Path
import torch

# Picked Widely used values
mean = tuple([0.485, 0.485, 0.406])
std = tuple([0.229, 0.224, 0.225])

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Dataset directories
train_img_dir = PROJECT_ROOT / "dataset" / "train" # Update this to the correct path
val_img_dir = PROJECT_ROOT / "dataset" / "val" # Update this to the correct path
logs_dir = PROJECT_ROOT / "logs"

# Device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

input_size = (1, 3, 224, 224)
num_classes = 1000

num_workers: int = -1,

learning_rate = 2.11E-03 # Found with LR finder
weight_decay = 1e-4

experiment_name = "imagenet_training_code_test"

scheduler_type = 'one_cycle_policy'

lr_finder_kwargs = {
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': 1000,
            'step_mode': 'exp'
        }

onecycle_kwargs = {
            'lr_strategy': 'manual',  # 'conservative', 'manual'
            'pct_start': 0.2,
            'anneal_strategy': 'cos',
            'div_factor': 100.0,
            'final_div_factor': 1000.0
        }

epochs = 10
batch_size = 128
dynamic_batch_size = True

# Define resolution schedule for Progressive Resizing + FixRes
# Format: {epoch: (resolution, use_train_augs, batch_size)}
# Set to None for No schedule, uses default 224px throughout the training
prog_resizing_fixres_schedule = {
    # 0: (128, True, 512),    # Epochs 0-9: 128px, train augs, BS=512
    # 10: (224, True, 320),   # Epochs 10-84: 224px, train augs, BS=320
    # 85: (288, False, 256),  # Epochs 85-90: 288px, test augs (FixRes), BS=256
    0: (128, True, 128),  # 280 * 4 = 1120  
    4: (224, True, 64),   # 559 * 4 = 2236
    8: (288, False, 32),  # 1118 * 2 = 2236
}

dataset_size = 35746 # Update this to the correct size

### Example
# Assumes batch_size=128 for all 90 epochs
# total_steps = 90 * (1281167 // 128) = 90 * 10009 = 900,810 steps
# With dynamic batch sizing and set resolution schedule):
# Epochs 0-9: BS=512  → 10 * (1281167 // 512) = 10 * 2502 = 25,020 steps
# Epochs 10-84: BS=320 → 75 * (1281167 // 320) = 75 * 4004 = 300,300 steps  
# Epochs 85-89: BS=256 → 5 * (1281167 // 256) = 5 * 5005 = 25,025 steps
# Total: 25,020 + 300,300 + 25,025 = 350,345 steps
# This is a significant reduction in the number of steps required for training, which is a major benefit of using dynamic batch sizing and resolution schedule.

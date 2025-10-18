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

learning_rate = 1.13e-01 # Found with LR finder
weight_decay = 1e-4

experiment_name = "experiment1"

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

epochs = 50
batch_size = 128
dynamic_batch_size = True

dataset_size = 1281167 # Update this to the correct size
batch_size_schedule = [] # Update this to the correct batch size schedule
batch_size_schedule[:5] = [128] * 5 # 5 epochs at 128
batch_size_schedule[5:10] = [64] * 5 # 5 epochs at 64
batch_size_schedule[-1] = 32 # Rest epoch at 32

from utils import get_total_num_steps
total_steps = get_total_num_steps(dataset_size, batch_size, batch_size_schedule, epochs, dynamic_batch_size)
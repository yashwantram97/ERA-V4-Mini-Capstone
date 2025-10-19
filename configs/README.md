# Hardware Configuration Profiles

This directory contains hardware-specific configuration profiles optimized for different training environments.

## Available Profiles

### 1. Local (`local`) - MacBook M4 Pro
**Profile Name:** `local`  
**Best for:** Development, debugging, quick experiments

**Hardware:**
- CPU: Apple M4 Pro (12-14 cores)
- GPU: Integrated GPU with unified memory
- Memory: 16-64GB unified memory

**Dataset:**
- Path: `PROJECT_ROOT/dataset/imagenet-mini/`
- Size: 35,746 images (ImageNet-Mini)
- Use Case: Local development and testing

**Optimizations:**
- Conservative batch sizes (32-64)
- Fewer workers (4)
- Mixed precision training
- Smaller progressive resizing schedule
- Faster iterations for debugging

**Expected Training Time:** ~20-30 minutes for 10 epochs

### 2. AWS g5d.12xlarge (`g5d`)
**Profile Name:** `g5d`  
**Best for:** Production training, good cost/performance balance

**Hardware:**
- CPUs: 48 vCPUs (Intel Xeon)
- GPUs: 4x NVIDIA A10G (24GB GDDR6 each)
- Memory: 192GB RAM
- Storage: 1x 3800GB NVMe SSD

**Dataset:**
- Path: `/home/ec2-user/imagenet1k/`
- Size: 1,281,167 images (Full ImageNet-1K)
- Use Case: Production training

**Optimizations:**
- Moderate to high batch sizes (256-512)
- Good number of workers (12)
- DDP strategy for multi-GPU
- Full progressive resizing schedule
- Mixed precision for A10G optimization

**Expected Training Time:** ~4-6 hours for 90 epochs  
**Cost:** ~$5-10 per full training run

### 3. AWS p3.16xlarge (`p3`)
**Profile Name:** `p3`  
**Best for:** Large-scale training, fastest training time

**Hardware:**
- CPUs: 64 vCPUs (Intel Xeon E5-2686 v4)
- GPUs: 8x NVIDIA V100 (16GB HBM2 each)
- Memory: 488GB RAM
- Network: 25 Gbps

**Dataset:**
- Path: `/home/ec2-user/imagenet1k/`
- Size: 1,281,167 images (Full ImageNet-1K)
- Use Case: Production training

**Optimizations:**
- High batch sizes across 8 GPUs (256-768)
- Maximum workers (16)
- DDP strategy for 8-GPU training
- Aggressive progressive resizing
- Tensor Core optimization

**Expected Training Time:** ~2-3 hours for 90 epochs  
**Cost:** ~$15-25 per full training run

## Usage

### Command Line
```bash
# Train on local MacBook M4 Pro
python train.py --config local

# Train on AWS g5d.12xlarge
python train.py --config g5d

# Train on AWS p3.16xlarge with SAM optimizer
python train.py --config p3 --use-sam

# Override learning rate (uses config value if not specified)
python train.py --config g5d --lr 0.001

# Combine multiple options
python train.py --config p3 --lr 0.005 --use-sam

# Resume from checkpoint
python train.py --config g5d --resume logs/experiment/checkpoints/last.ckpt

# List available configs
python train.py --list-configs
```

### Python API
```python
from configs import get_config

# Get a specific configuration
config = get_config('local')
# or
config = get_config('g5d')
# or
config = get_config('p3')

# Access config values
print(f"Batch size: {config.batch_size}")
print(f"Workers: {config.num_workers}")
print(f"Epochs: {config.epochs}")

# Use in training
train_with_lightning(config=config)
```

## Configuration Structure

Each configuration file contains:

1. **Profile Information**
   - `PROFILE_NAME`: Short name for the profile
   - `PROFILE_DESCRIPTION`: Human-readable description

2. **Dataset Paths** (Environment-specific)
   - `TRAIN_IMG_DIR`: Path to training images
   - `VAL_IMG_DIR`: Path to validation images
   - `DATASET_SIZE`: Number of training images
   - Note: Paths differ between local and cloud environments

3. **Training Settings**
   - `EPOCHS`: Number of training epochs
   - `BATCH_SIZE`: Base batch size
   - `DYNAMIC_BATCH_SIZE`: Whether to use dynamic batch sizing
   - `LEARNING_RATE`: Initial learning rate
   - `WEIGHT_DECAY`: Weight decay for optimizer

3. **Hardware Settings**
   - `NUM_WORKERS`: Number of data loading workers
   - `PRECISION`: Training precision (16-mixed, 32-true, etc.)
   - `NUM_DEVICES`: Number of GPUs to use
   - `STRATEGY`: Multi-GPU strategy (DDP, etc.)

4. **Progressive Resizing Schedule**
   - `PROG_RESIZING_FIXRES_SCHEDULE`: Dictionary mapping epochs to (resolution, use_train_augs, batch_size)

5. **Callback Settings**
   - `EARLY_STOPPING_PATIENCE`: Patience for early stopping
   - `SAVE_TOP_K`: Number of best models to save
   - `LOG_EVERY_N_STEPS`: Logging frequency

## Adding New Profiles

To add a new hardware profile:

1. Create a new file: `configs/my_profile_config.py`
2. Import from base config: `from .base_config import *`
3. Override necessary settings
4. Update `config_manager.py` to include the new profile
5. Add to choices in `train.py` argument parser

Example:
```python
# configs/my_profile_config.py
from .base_config import *

PROFILE_NAME = "my_profile"
PROFILE_DESCRIPTION = "My Custom Hardware Setup"

EPOCHS = 50
BATCH_SIZE = 128
NUM_WORKERS = 8
PRECISION = "16-mixed"

PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True, 256),
    20: (224, True, 128),
    40: (288, False, 64),
}
```

## Configuration Best Practices

### Batch Size Selection
- **Rule of thumb:** Use the largest batch size that fits in GPU memory
- Start with suggested values and monitor GPU memory usage
- Larger batch sizes = faster training but may affect convergence
- Use dynamic batch sizing with progressive resizing for efficiency

### Number of Workers
- **Rule of thumb:** 2-4 workers per GPU
- More workers = faster data loading but more CPU/RAM usage
- Too many workers can cause CPU bottleneck
- Monitor CPU usage to find optimal value

### Progressive Resizing
- Start with small resolution (128px) for fast initial training
- Gradually increase to target resolution (224px)
- End with FixRes (288px with test-time augmentations)
- Adjust batch sizes per stage to maintain GPU utilization

### Multi-GPU Training
- Use DDP (Distributed Data Parallel) strategy
- Effective batch size = batch_size × num_gpus
- Adjust learning rate accordingly (linear scaling rule)
- Ensure sufficient workers for all GPUs

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE`
- Reduce `NUM_WORKERS`
- Use gradient accumulation
- Enable gradient checkpointing

### Slow Data Loading
- Increase `NUM_WORKERS`
- Use SSD for data storage
- Enable pin_memory in dataloader
- Consider data caching

### Poor GPU Utilization
- Increase `BATCH_SIZE`
- Reduce `NUM_WORKERS` (CPU bottleneck)
- Check data loading speed
- Enable mixed precision

## File Structure

```
configs/
├── __init__.py              # Package initialization
├── README.md                # This file
├── base_config.py           # Shared configuration
├── config_manager.py        # Configuration management
├── local_config.py          # MacBook M4 Pro config
├── g5d_config.py           # AWS g5d.12xlarge config
└── p3_config.py            # AWS p3.16xlarge config
```

## Support

For questions or issues with configurations:
1. Check this README
2. Review the specific config file
3. Run `python train.py --list-configs` to see available options
4. Test with smaller settings first


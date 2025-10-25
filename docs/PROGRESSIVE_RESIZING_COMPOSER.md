# Progressive Resizing - MosaicML Composer Approach

## Overview

We've implemented the progressive resizing technique following [MosaicML Composer's proven approach](https://docs.mosaicml.com/projects/composer/en/stable/method_cards/progressive_resizing.html) for curriculum-like learning in ImageNet training.

## What is Progressive Resizing?

Progressive resizing is a training technique where:
1. **Early training** uses smaller images (faster training)
2. **Middle training** gradually increases resolution (curriculum learning)
3. **Late training** uses full resolution (optimal accuracy)

This creates a **curriculum learning** effect where the network learns from "easy" (small) to "hard" (large) examples.

## Benefits

✅ **Faster Training**: Early epochs run ~4x faster (112² vs 224² pixels = 75% less computation)
✅ **Curriculum Learning**: Progressive difficulty helps network learn better representations
✅ **Better Speed/Accuracy Tradeoff**: Train longer in less wall-clock time
✅ **Proven Results**: Validated by MosaicML on ResNet-50 ImageNet training

## MosaicML Recommended Settings

For ResNet-50 on ImageNet, MosaicML found these hyperparameters work well:

```python
initial_scale = 0.5        # Start at 50% of target size (112px for 224px target)
delay_fraction = 0.5       # Stay at initial scale for first 50% of training
finetune_fraction = 0.2    # Train at full size for last 20% of training
size_increment = 4         # Round sizes to multiples of 4 (alignment)
```

### Training Schedule Breakdown (60 epochs)

| Phase | Epochs | Duration | Resolution | Purpose |
|-------|--------|----------|------------|---------|
| **Delay** | 0-29 | 50% | 112px | Fast training, learn basic features |
| **Progressive** | 30-47 | 30% | 112→224px | Curriculum learning, gradual complexity |
| **Fine-tune** | 48-59 | 20% | 224px | Full resolution optimization |

## Implementation

### Helper Function

```python
from src.callbacks import create_progressive_resize_schedule

schedule = create_progressive_resize_schedule(
    total_epochs=60,
    target_size=224,          # Standard ImageNet resolution
    initial_scale=0.5,        # Start at 50% (112px)
    delay_fraction=0.5,       # First 50% at initial scale
    finetune_fraction=0.2,    # Last 20% at full size
    size_increment=4,         # Round to multiples of 4
    use_fixres=False,         # Optional FixRes phase
    fixres_size=256           # Higher resolution for FixRes
)
```

### Usage in Config

All config files (`local_config.py`, `g5_config.py`, `p3_config.py`) now use this approach:

```python
# configs/g5_config.py
from src.callbacks import create_progressive_resize_schedule

PROG_RESIZING_FIXRES_SCHEDULE = create_progressive_resize_schedule(
    total_epochs=EPOCHS,
    target_size=224,
    initial_scale=0.5,
    delay_fraction=0.5,
    finetune_fraction=0.2,
    size_increment=4,
    use_fixres=False,
    fixres_size=256
)
```

### Integration with Training

The schedule is automatically used by `ResolutionScheduleCallback`:

```python
# train.py
from src.callbacks import ResolutionScheduleCallback

resolution_callback = ResolutionScheduleCallback(
    schedule=config.prog_resizing_fixres_schedule
)

trainer = L.Trainer(
    callbacks=[resolution_callback, ...],
    ...
)
```

## Optional: FixRes Enhancement

Enable FixRes for an additional +1-2% accuracy boost:

```python
schedule = create_progressive_resize_schedule(
    ...
    use_fixres=True,    # Enable FixRes
    fixres_size=256     # Train at 256px with test-time augmentations
)
```

FixRes aligns training and testing distributions by using test-time augmentations (Resize + CenterCrop) in the final epochs.

## Visualization

To see the generated schedule:

```bash
python test_progressive_schedule.py
```

Example output:
```
================================================================================
                     60 Epochs - MosaicML Composer Approach                     
================================================================================
Epoch      Resolution      Augmentation                  
--------------------------------------------------------------------------------
0-29       112x112 px         Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)
30-30      112x112 px         Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)
31-31      120x120 px         Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)
...
48+        224x224 px         Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)
================================================================================
```

## Customization

You can adjust the schedule for different training scenarios:

### Faster Ramp-Up (More Aggressive)
```python
schedule = create_progressive_resize_schedule(
    total_epochs=60,
    initial_scale=0.5,
    delay_fraction=0.3,      # Shorter initial phase (18 epochs)
    finetune_fraction=0.3,   # Longer fine-tune (18 epochs)
    size_increment=4
)
```

### Longer Training
```python
schedule = create_progressive_resize_schedule(
    total_epochs=100,        # Extended training
    initial_scale=0.5,
    delay_fraction=0.5,
    finetune_fraction=0.2,
    size_increment=4
)
```

### Conservative (Smaller Initial Scale)
```python
schedule = create_progressive_resize_schedule(
    total_epochs=60,
    initial_scale=0.4,       # Start even smaller (90px)
    delay_fraction=0.6,      # Stay smaller longer
    finetune_fraction=0.2,
    size_increment=4
)
```

## Performance Impact

### Computational Savings

Resolution changes impact compute time:
- **112px**: 0.25x compute vs 224px (4x faster!)
- **160px**: 0.51x compute vs 224px (2x faster)
- **224px**: 1.0x compute (baseline)

For our 60-epoch schedule:
- Epochs 0-29 (50%): 4x faster → saves ~37.5% training time
- Epochs 30-47 (30%): 1.5-3x faster → saves ~10-15% training time
- Epochs 48-59 (20%): 1x (baseline)

**Total estimated speedup: ~35-40% faster training time!**

### Quality Impact

Based on MosaicML's findings:
- ✅ Similar or better accuracy compared to fixed resolution
- ✅ Better generalization from curriculum learning
- ✅ Can train for more epochs in same wall-clock time
- ⚠️ May require slight learning rate adjustments

## References

- [MosaicML Composer - Progressive Resizing](https://docs.mosaicml.com/projects/composer/en/stable/method_cards/progressive_resizing.html)
- [fast.ai - Progressive Resizing](https://docs.fast.ai/callback.schedule.html#progressive-resizing)
- [FixRes Paper (Touvron et al.)](https://arxiv.org/abs/1906.06423)

## Files Modified

- `src/callbacks/resolution_schedule_callback.py` - Added `create_progressive_resize_schedule()` function
- `src/callbacks/__init__.py` - Exported new function
- `configs/g5_config.py` - Updated to use Composer approach
- `configs/local_config.py` - Updated to use Composer approach
- `configs/p3_config.py` - Updated to use Composer approach
- `test_progressive_schedule.py` - Test script to visualize schedules


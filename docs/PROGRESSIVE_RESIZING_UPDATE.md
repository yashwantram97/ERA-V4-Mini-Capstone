# Progressive Resizing Update - Composer Approach

## Summary

✅ **Updated all configs to use MosaicML Composer's proven progressive resizing approach**

The configs now implement curriculum-like learning where training progresses from small (easy) to large (hard) images, following the exact hyperparameters that MosaicML found work well for ResNet-50 on ImageNet.

## Changes Made

### 1. New Helper Function
Created `create_progressive_resize_schedule()` in `src/callbacks/resolution_schedule_callback.py`:
- Follows MosaicML Composer's approach with percentage-based scheduling
- Supports all recommended parameters (initial_scale, delay_fraction, finetune_fraction, size_increment)
- Optional FixRes support for +1-2% accuracy boost

### 2. Updated All Configs
Modified `g5_config.py`, `local_config.py`, and `p3_config.py` to use:
```python
PROG_RESIZING_FIXRES_SCHEDULE = create_progressive_resize_schedule(
    total_epochs=EPOCHS,
    target_size=224,          # Standard ImageNet resolution
    initial_scale=0.5,        # Start at 50% (112px) - MosaicML recommended
    delay_fraction=0.5,       # First 50% at initial scale - MosaicML recommended
    finetune_fraction=0.2,    # Last 20% at full size - MosaicML recommended
    size_increment=4,         # Round to multiples of 4 - MosaicML recommended
    use_fixres=False,
    fixres_size=256
)
```

## What You Get Now

### Before (Your Manual Schedule)
```python
{
    0: (128, True),    # Start at 128px
    10: (224, True),   # Jump to 224px
    50: (256, True),   # Jump to 256px
}
```
❌ Fixed epoch boundaries
❌ Abrupt jumps in resolution
❌ Not optimized for different epoch counts

### After (Composer Approach)
```python
create_progressive_resize_schedule(
    total_epochs=60,
    initial_scale=0.5,
    delay_fraction=0.5,
    finetune_fraction=0.2,
    size_increment=4
)
```
✅ **Percentage-based** - adapts to any epoch count
✅ **Smooth progression** - gradual resolution increase (112→116→120→...→224)
✅ **Proven hyperparameters** - validated by MosaicML on ImageNet
✅ **Curriculum learning** - systematic easy→hard progression

## Example Schedule (60 Epochs)

| Phase | Epochs | Resolution | Purpose |
|-------|--------|------------|---------|
| Delay | 0-29 (50%) | 112px | Fast learning, basic features (~4x faster) |
| Progressive | 30-47 (30%) | 112→224px | Curriculum learning (smooth progression) |
| Fine-tune | 48-59 (20%) | 224px | Full resolution optimization |

**Expected speedup: ~35-40% faster training time!**

## Try It Out

### View the Schedule
```bash
python test_progressive_schedule.py
```

### Train as Before
```bash
python train.py --config g5
```

The schedule is automatically applied during training!

## Customization

### Enable FixRes (+1-2% accuracy)
```python
use_fixres=True,
fixres_size=256
```

### More Aggressive (Faster Ramp)
```python
delay_fraction=0.3,      # Less time at small size
finetune_fraction=0.3    # More time at full size
```

### Different Target Size
```python
target_size=288,         # Train for higher resolution
initial_scale=0.5,       # Start at 144px
```

## Benefits Over Previous Implementation

1. **Adaptable**: Works for any epoch count (30, 60, 100, etc.)
2. **Proven**: MosaicML validated these hyperparameters
3. **Smooth**: Gradual resolution increase, not sudden jumps
4. **Documented**: Clear rationale from research
5. **Curriculum**: True curriculum learning progression

## Next Steps

Your training will automatically use the new schedule. No code changes needed!

To experiment:
1. Try different `delay_fraction` values (0.3-0.7)
2. Enable FixRes for production runs
3. Adjust `initial_scale` for faster/slower start

## Documentation

See `docs/PROGRESSIVE_RESIZING_COMPOSER.md` for full details.


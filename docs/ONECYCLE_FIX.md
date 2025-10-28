# OneCycleLR Scheduler Fix - Max LR Timing Issue

## Problem Summary

The OneCycleLR scheduler was reaching maximum learning rate at epoch 13 instead of the expected epoch 27.

## Root Causes

### 1. **Critical Bug: Wrong Parameter Name**
```python
# WRONG ❌
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=self.learning_rate,
    steps=total_steps,  # Wrong parameter name!
    ...
)

# CORRECT ✅
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=self.learning_rate,
    total_steps=total_steps,  # Correct parameter name
    ...
)
```

The PyTorch OneCycleLR expects `total_steps` as the parameter name, not `steps`. Using the wrong parameter name can cause PyTorch to use a default value or raise an error.

### 2. **Hardcoded Values vs Config Values**
The scheduler was using hardcoded values that didn't match your `g5_config.py`:

| Parameter | Hardcoded Value | Config Value (g5_config.py) |
|-----------|----------------|---------------------------|
| `pct_start` | 0.3 | 0.2 |
| `div_factor` | 50.0 | 100.0 |

This mismatch meant:
- Expected max LR at: 90 epochs × 0.2 = **18 epochs**
- Actual max LR at: 90 epochs × 0.3 = **27 epochs**

### 3. **Misleading Print Statements**
The print statements showed the config values (0.2, 100.0) but the scheduler was actually using hardcoded values (0.3, 50.0), making debugging difficult.

## How It Should Work with Your Config

Given your g5_config.py settings:
- **Total epochs**: 90
- **Batch size**: 128
- **Accumulate grad batches**: 2
- **Num devices**: 4 (DDP)
- **Dataset size**: 1,281,167 (ImageNet-1K)
- **pct_start**: 0.2

### Calculation:
```
Samples per device = 1,281,167 / 4 = 320,292 samples
Batches per device per epoch = 320,292 / 128 = 2,502 batches
Optimizer steps per epoch = 2,502 / 2 = 1,251 steps (with grad accumulation)
Total optimizer steps = 1,251 × 90 = 112,590 steps

Max LR reached at step = 112,590 × 0.2 = 22,518 steps
Max LR reached at epoch = 22,518 / 1,251 = ~18 epochs ✅
```

## Changes Made

### 1. Fixed Parameter Name
- Changed `steps=total_steps` → `total_steps=total_steps`

### 2. Used Config Values
- Now imports and uses `onecycle_kwargs` from config
- No more hardcoded values

### 3. Enhanced Debug Output
Added comprehensive logging to help debug LR schedule:
```python
print(f"🔄 Created OneCycleLR Scheduler:")
print(f"   Max LR: {self.learning_rate:.4e}")
print(f"   Initial LR: {self.learning_rate/div_factor:.4e}")
print(f"   Final LR: {self.learning_rate/(div_factor*final_div_factor):.4e}")
print(f"   Total steps: {total_steps}")
print(f"   Steps per epoch: {steps_per_epoch:.1f}")
print(f"   Total epochs: {self.trainer.max_epochs}")
print(f"   Num devices: {self.trainer.num_devices}")
print(f"   Accumulate grad batches: {self.trainer.accumulate_grad_batches}")
print(f"   Strategy: {self.trainer.strategy.__class__.__name__}")
print(f"   Pct start: {pct_start}")
print(f"   Div factor: {div_factor}")
print(f"   Final div factor: {final_div_factor}")
print(f"   Max LR reached at step: {max_lr_step} (≈ epoch {max_lr_epoch:.1f})")
```

## Gradient Accumulation Impact

**Your question: "Is it because of grad accumulation?"**

Yes, gradient accumulation affects the calculation, **but Lightning's `estimated_stepping_batches` should handle it correctly**. Here's how:

### Without Gradient Accumulation:
- 2,502 batches/epoch → 2,502 optimizer steps/epoch

### With Accumulation = 2:
- 2,502 batches/epoch → 1,251 optimizer steps/epoch (half the steps!)

The key is that `estimated_stepping_batches` returns **optimizer steps**, not batch iterations, so it already accounts for gradient accumulation.

## Verification Steps

1. **Run training and check the console output**:
   ```bash
   python train.py --config g5
   ```

2. **Look for the scheduler creation message**:
   ```
   🔄 Created OneCycleLR Scheduler:
      Max LR reached at step: 22518 (≈ epoch 18.0)
   ```

3. **Monitor learning rate during training**:
   - Check TensorBoard: `train/learning_rate` metric
   - Or watch the console output at each epoch start

4. **Verify max LR timing**:
   - With `pct_start=0.2`, max LR should be reached at ~epoch 18
   - With `pct_start=0.3`, max LR should be reached at ~epoch 27

## Expected Behavior After Fix

With your g5_config.py (`pct_start=0.2`):
- **Warmup phase**: Epochs 0-18 (20% of training)
  - LR increases from `0.15/100 = 0.0015` to `0.15`
- **Annealing phase**: Epochs 18-90 (80% of training)
  - LR decreases from `0.15` to `0.15/(100×1000) = 0.0000015`

## Additional Notes

### Why OneCycleLR is Sensitive to total_steps:
OneCycleLR divides training into:
1. Warmup: 0 to `pct_start × total_steps`
2. Annealing: `pct_start × total_steps` to `total_steps`

If `total_steps` is incorrect, the entire schedule is wrong!

### Lightning's estimated_stepping_batches:
```python
estimated_stepping_batches = (
    num_training_batches  
    // accumulate_grad_batches  
    * max_epochs
)
```

This automatically accounts for:
- Dataset size
- Batch size
- Number of devices (DDP)
- Gradient accumulation
- Max epochs

## Files Modified

1. **src/models/resnet_module.py**:
   - Fixed `steps` → `total_steps` parameter
   - Added import for `onecycle_kwargs`
   - Updated scheduler to use config values
   - Enhanced debug output with epoch calculation

## Testing Checklist

- [ ] Verify max LR is reached at correct epoch (~18 with pct_start=0.2)
- [ ] Check initial LR is `learning_rate/div_factor`
- [ ] Check final LR is `learning_rate/(div_factor×final_div_factor)`
- [ ] Confirm print output shows correct values
- [ ] Monitor TensorBoard `train/learning_rate` plot


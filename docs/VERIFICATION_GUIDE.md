# Training Components Verification Guide

Before running your training on expensive GPU/AWS instances, use these verification scripts to ensure everything is configured correctly.

## Quick Start

### 1. Run Visual Verification (Recommended - Fast & Visual)

This creates plots and images showing your augmentations, LR schedule, and more:

```bash
python verify_visual.py
```

**Output:** Creates a `verification_outputs/` directory with these plots:
- `augmentations.png` - Shows train augmentations at different resolutions and FixRes augmentations
- `onecycle_schedule.png` - Plots your LR and momentum schedule across all epochs
- `resolution_schedule.png` - Timeline showing when resolutions change
- `blurpool_verification.png` - Confirms BlurPool is integrated

**Time:** ~30-60 seconds

---

### 2. Run Comprehensive Tests (Optional - More Thorough)

This runs detailed programmatic tests on all components:

```bash
python verify_training_components.py
```

**Tests:**
1. ✅ Image Augmentation - Verifies transforms are applied correctly
2. ✅ OneCycle Policy - Checks LR scheduler configuration
3. ✅ Resolution Schedule - Tests progressive resizing
4. ✅ BlurPool - Verifies anti-aliased pooling is active
5. ✅ FixRes - Confirms test-time augmentations work

**Time:** ~2-3 minutes

---

## What to Check

### ✅ Augmentations (`augmentations.png`)

**Expected:**
- **Row 1 (Train 224px):** Images should vary - different crops, flips, colors
- **Row 2 (Train 288px):** Images should vary - different crops, flips, colors  
- **Row 3 (FixRes 288px):** All images should be IDENTICAL - deterministic center crop

**Issues?**
- If train images look identical → Random augmentations not working
- If FixRes images vary → Test-time augmentations not deterministic

---

### ✅ OneCycle Schedule (`onecycle_schedule.png`)

**Expected:**
- **LR (top plot):** 
  - Starts low (~2.11e-5)
  - Rises to max (~2.11e-3) at 20% of training
  - Gradually decreases to very low (~2.11e-6) by end
- **Momentum (bottom plot):**
  - Starts high (0.95)
  - Drops to low (0.85) during LR warmup
  - Returns to high (0.95) during annealing

**Issues?**
- If LR doesn't rise → Warmup not working
- If LR doesn't fall → Annealing not working
- If momentum is flat → Cycle momentum disabled

---

### ✅ Resolution Schedule (`resolution_schedule.png`)

**Expected (60 epochs):**
- **Epochs 0-14:** 128px with Train Augs (fast training, low res)
- **Epochs 15-49:** 224px with Train Augs (main training phase)
- **Epochs 50-59:** 288px with FixRes (fine-tuning, high res, test augs)

**Issues?**
- If no FixRes phase → Add epoch with `use_train_augs=False`
- If resolutions don't match config → Check `local_config.py`

---

### ✅ BlurPool (`blurpool_verification.png`)

**Expected:**
- Should show BlurPool layers detected in the model
- If 0 BlurPool layers, check that `antialiased_cnns` is imported

**Issues?**
- If no BlurPool → Verify you're using `antialiased_cnns.resnet50()`
- Forward pass should still work even if layers aren't explicitly named "BlurPool"

---

## Configuration Files

Your training configuration comes from:

```
configs/
├── local_config.py    ← For MacBook local testing
├── p4_config.py       ← For AWS p4 instances
└── g5_config.py       ← For AWS g5 instances
```

Key parameters to verify:

```python
# Training
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 2.11e-3

# Resolution Schedule
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (128, True),    # Start: 128px, train augs
    15: (224, True),   # Middle: 224px, train augs  
    50: (288, False)   # End: 288px, FixRes (test augs)
}

# OneCycle
ONECYCLE_KWARGS = {
    'pct_start': 0.2,           # 20% warmup
    'div_factor': 100.0,        # Initial LR = max_lr/100
    'final_div_factor': 1000.0  # Final LR = max_lr/100/1000
}
```

---

## Common Issues & Fixes

### Issue: "No module named 'antialiased_cnns'"

```bash
pip install antialiased-cnns
```

### Issue: "No training images found"

Check that your dataset is at:
```
dataset/imagenet-mini/
├── train/
│   ├── n01558993/
│   ├── n01692333/
│   └── ...
└── val/
    └── ...
```

### Issue: Augmentations not random

In `src/utils/utils.py`, verify you have:
```python
A.RandomResizedCrop(...)  # Not just Resize
A.HorizontalFlip(p=0.5)   # p > 0
```

### Issue: FixRes not working

In your schedule, make sure you have:
```python
50: (288, False)  # ← False = test-time augmentations
```

Not:
```python
50: (288, True)  # ← True = train augmentations
```

---

## Checklist Before GPU Training

- [ ] Run `python verify_visual.py` 
- [ ] Check all 4 plots look correct
- [ ] Verify augmentations show variation (train) vs consistency (FixRes)
- [ ] Verify OneCycle LR rises then falls
- [ ] Verify resolution schedule has 3 phases
- [ ] Verify BlurPool is integrated
- [ ] (Optional) Run `python verify_training_components.py` for detailed tests

---

## Next Steps

Once verification passes:

1. **Local dry run (1-2 epochs):**
   ```bash
   python train.py --config local --max_epochs 2
   ```

2. **AWS training:**
   ```bash
   python train.py --config p4 --max_epochs 90
   ```

3. **Monitor with TensorBoard:**
   ```bash
   tensorboard --logdir logs/
   ```

---

## Questions?

- Image augmentations: Check `src/utils/utils.py:get_transforms()`
- OneCycle policy: Check `src/models/resnet_module.py:configure_optimizers()`
- Resolution schedule: Check `src/callbacks/resolution_schedule_callback.py`
- FixRes: Check `configs/local_config.py:PROG_RESIZING_FIXRES_SCHEDULE`
- BlurPool: Check `src/models/resnet_module.py` (uses `antialiased_cnns`)


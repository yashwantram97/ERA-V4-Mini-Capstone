# Training Analysis: Why We Didn't Reach >75% Accuracy

**Date**: October 21, 2025  
**Final Accuracy**: 64.28% (Target: >75%)  
**Training Duration**: 60 epochs, ~80 minutes

---

## 🔍 Problem Diagnosis

### 1. **Critical Issue: FixRes Phase Overfitting** 🚨

**The Evidence:**
- Epochs 1-51: Gradual improvement, train/val accuracy relatively aligned
- **Epochs 52-60**: Sudden divergence
  - Train accuracy: 56% → 67-71% (massive jump)
  - Val accuracy: 61% → 64% (plateaus)
  - **Gap**: 7% overfitting

**Root Cause:**
The FixRes phase (epochs 50-59) switches to **256px resolution with TEST augmentations** (no training augmentations). This means:
- ✅ Higher resolution = more detail
- ❌ No augmentation = severe overfitting
- ❌ Model memorizes training data instead of generalizing

**Original Schedule:**
```python
{
    0: (128, True),    # Epochs 0-9: train augs ✅
    10: (224, True),   # Epochs 10-49: train augs ✅
    50: (256, False),  # Epochs 50-59: TEST augs ❌❌❌
}
```

This is a **misunderstanding of the FixRes paper**. FixRes fine-tunes at higher resolution but should still use training augmentations!

---

### 2. **Insufficient Initial Learning Phase**

**Problem:**
- Only 10 epochs at 128px (17% of training)
- Not enough time to learn robust low-level features
- Jumping to 224px too early

**Impact:**
- Slower convergence at 224px
- Model hasn't built strong feature representations

---

### 3. **Weak Regularization**

**Problem 1: Weight Decay Too Low**
- Current: `1e-4` (0.0001)
- For 60-epoch ImageNet training: Should be `5e-4` to `1e-3`
- Result: Insufficient regularization → overfitting

**Problem 2: Conservative MixUp**
- Current: `mixup_alpha=0.2` (relatively weak)
- CutMix: Disabled (`cutmix_alpha=0.0`)
- Result: Not enough augmentation diversity

**Problem 3: Weak Data Augmentation**
- Color jitter: `brightness_limit=0.2` (conservative)
- CoarseDropout: 50% probability, single hole
- No blur augmentation
- Result: Model overfits to training data appearance

---

### 4. **Training Dynamics**

From the logs, we can see:
```
Epoch  | Train Acc | Val Acc | Gap    | Notes
-------|-----------|---------|--------|---------------------------
1-9    | 1-18%     | 2-18%   | ~0%    | 128px, learning basics
10-11  | 18-20%    | 17-17%  | +3%    | 224px transition, val dip
12-29  | 23-45%    | 23-48%  | -3%    | Val ahead! Good learning
30-51  | 46-56%    | 47-62%  | -6%    | Val still ahead
52-60  | 67-71%    | 63-64%  | +7%    | 256px + test augs = OVERFIT
```

**Key Observations:**
- Epochs 12-51: Val accuracy **ahead** of train accuracy (good sign!)
- Epochs 52-60: Sudden reversal due to FixRes phase

---

## ✅ Applied Fixes

### **Fix 1: Revised Resolution Schedule**

**New Schedule:**
```python
PROG_RESIZING_FIXRES_SCHEDULE = {
    0: (160, True),    # Epochs 0-14: 160px, train augs (25%)
    15: (224, True),   # Epochs 15-44: 224px, train augs (50%)
    45: (256, True),   # Epochs 45-59: 256px, TRAIN augs (25%)
}
```

**Why This Works:**
1. **Longer warmup**: 15 epochs at 160px (vs 10 at 128px)
   - Better initial feature learning
   - 160px is a good middle ground (not too small, not too large)

2. **Extended main phase**: 30 epochs at 224px (vs 40)
   - Still plenty of time at target resolution
   - Better balanced across phases

3. **Keep augmentations**: 256px with **train augmentations**
   - Higher resolution for fine details
   - Still regularized by augmentations
   - **This is the key fix!**

**Expected Impact:** +5-8% accuracy

---

### **Fix 2: Increased Weight Decay**

```python
WEIGHT_DECAY = 5e-4  # Increased from 1e-4
```

**Why This Works:**
- Stronger L2 regularization
- Prevents overfitting in later epochs
- Standard for ImageNet training

**Expected Impact:** +2-3% accuracy

---

### **Fix 3: Enhanced MixUp/CutMix**

```python
MIXUP_KWARGS = {
    'mixup_alpha': 0.4,      # Increased from 0.2
    'cutmix_alpha': 1.0,     # Enabled (was 0.0)
    'switch_prob': 0.5,      # 50/50 mix
    # ... other params
}
```

**Why This Works:**
- **MixUp (alpha=0.4)**: Stronger label smoothing and feature mixing
- **CutMix (alpha=1.0)**: Combines spatial mixing with MixUp
- **Combined effect**: Forces model to learn more robust features

**Expected Impact:** +3-5% accuracy

---

### **Fix 4: Stronger Data Augmentation**

**Enhanced Augmentations:**
1. **Color Augmentation** (increased probability: 0.5 → 0.7)
   - Brightness/Contrast: 0.2 → 0.3
   - Hue/Sat/Val: More aggressive
   - Added ColorJitter variant

2. **Spatial Dropout** (CoarseDropout)
   - Holes: 1 → 1-3 (can drop multiple patches)
   - Size: up to 30% (was 25%)
   - Probability: 0.5 → 0.7

3. **Blur Augmentation** (new)
   - GaussianBlur or MotionBlur
   - 20% probability
   - Prevents overfitting to sharp edges

**Why This Works:**
- More diverse training data
- Forces model to be robust to variations
- Prevents memorization

**Expected Impact:** +2-4% accuracy

---

## 📊 Expected Results

### **Conservative Estimate:**
- Fix 1 (Resolution + augs): +5%
- Fix 2 (Weight decay): +2%
- Fix 3 (MixUp/CutMix): +3%
- Fix 4 (Data augs): +2%
- **Total: +12%** → **~76% accuracy** ✅

### **Optimistic Estimate:**
- Fix 1: +8%
- Fix 2: +3%
- Fix 3: +5%
- Fix 4: +4%
- **Total: +20%** → **~84% accuracy** 🎯

### **Realistic Estimate:**
- **Expected: 75-80% accuracy**
- Should comfortably exceed the 75% target

---

## 🎯 Key Takeaways

### **What Went Wrong:**
1. ❌ FixRes phase removed augmentations → severe overfitting
2. ❌ Too short warmup phase (only 10 epochs at 128px)
3. ❌ Weak regularization (weight decay too low)
4. ❌ Conservative augmentation strategy

### **What We Fixed:**
1. ✅ Keep training augmentations at all resolutions
2. ✅ Longer warmup with better resolution schedule
3. ✅ Stronger regularization (weight decay + MixUp + CutMix)
4. ✅ More aggressive data augmentation

### **Core Lesson:**
> **FixRes doesn't mean "no augmentation"** - it means training at higher resolution for fine-tuning. You still need strong regularization!

---

## 🚀 Next Steps

### **To Validate These Changes:**
```bash
# On g5.12xlarge instance
python train.py
```

### **Expected Timeline:**
- Training time: ~3-4 hours (60 epochs)
- Should reach >70% by epoch 50
- Should reach >75% by epoch 58-60

### **Monitoring During Training:**
1. **Epochs 0-14 (160px)**: Should reach ~35-40% val accuracy
2. **Epochs 15-44 (224px)**: Should reach ~65-70% val accuracy
3. **Epochs 45-59 (256px)**: Should reach >75% val accuracy
4. **Train/Val gap**: Should stay < 5% throughout

### **If Still Not Reaching 75%:**

**Additional Options:**
1. **Increase to 80 epochs** (current schedule might need more time)
2. **Try learning rate of 0.025-0.028** (slightly higher)
3. **Add Stochastic Depth** (drop_path_rate=0.1)
4. **Try EMA (Exponential Moving Average)** of model weights

---

## 📚 References

1. **FixRes Paper**: "Fixing the train-test resolution discrepancy" (Touvron et al., 2019)
   - Key point: Fine-tune at higher resolution, **not** remove augmentation

2. **Progressive Resizing**: Fast.ai best practices
   - Start small for fast training
   - Gradually increase for fine details

3. **MixUp**: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
4. **CutMix**: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)

---

## 🔧 Files Modified

1. `configs/g5_config.py`:
   - Resolution schedule (lines 63-67)
   - Weight decay (line 49)
   - MixUp/CutMix settings (lines 108-116)

2. `src/utils/utils.py`:
   - Data augmentation pipeline (lines 22-53)

**Backup Before Training:**
```bash
git commit -am "Apply fixes to reach >75% accuracy"
```


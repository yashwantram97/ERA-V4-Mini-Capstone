# MixUp Testing Documentation

## Overview
This document describes the comprehensive MixUp testing that has been added to verify that MixUp augmentation is working correctly in the training pipeline.

## Test Files Modified

### 1. `verify_training_components.py`
Added **TEST 6: MixUp Verification** with 7 comprehensive sub-tests.

### 2. `verify_visual.py`
Added **visualize_mixup()** function that creates visual outputs for MixUp verification.

## Test Coverage

### TEST 6: MixUp Verification (`verify_training_components.py`)

#### 6.1 - MixUp Initialization (Enabled)
- **Purpose**: Verify MixUp initializes correctly when enabled
- **Checks**:
  - Model creates mixup_cutmix_fn when alpha > 0
  - Configuration parameters are applied correctly
  - Initialization message is displayed

#### 6.2 - MixUp Initialization (Disabled)
- **Purpose**: Verify MixUp is disabled when alpha=0
- **Checks**:
  - Model sets mixup_cutmix_fn to None when alpha=0
  - No MixUp is applied during training

#### 6.3 - MixUp with None kwargs
- **Purpose**: Verify MixUp handles None configuration
- **Checks**:
  - Model sets mixup_cutmix_fn to None when kwargs=None
  - Training proceeds without MixUp

#### 6.4 - MixUp Transformation on Data
- **Purpose**: Verify MixUp transforms data correctly
- **Checks**:
  - Image shapes remain consistent after MixUp
  - Labels are converted to soft labels (one-hot encoded)
  - Label dimensions are correct (batch_size, num_classes)
  - Soft labels sum to 1.0 (valid probability distribution)
  - Labels are actually mixed (not just one-hot)
  - Shows examples of mixed labels with probabilities

#### 6.5 - MixUp Image Mixing Verification
- **Purpose**: Verify MixUp actually mixes image pixels
- **Checks**:
  - Creates two distinct images (black and white)
  - Applies MixUp multiple times
  - Verifies randomness (different lambda values)
  - Checks mixed pixel values are between extremes
  - Reports lambda value range

#### 6.6 - MixUp Integration in Training Step
- **Purpose**: Verify MixUp works in actual training step
- **Checks**:
  - Training step with MixUp enabled works correctly
  - Loss computation handles soft labels
  - Training step without MixUp works correctly
  - Loss computation with hard labels and label smoothing

#### 6.7 - MixUp Configuration from Config
- **Purpose**: Verify configuration from config files is valid
- **Checks**:
  - mixup_alpha is in valid range [0.0, 1.0]
  - prob is in valid range [0.0, 1.0]
  - mode is one of ['batch', 'pair', 'elem']
  - All configuration values are reasonable

### Visual Tests (`verify_visual.py`)

#### visualize_mixup()
Creates three comprehensive visualizations:

##### 1. MixUp Image Visualization (`mixup_visualization.png`)
- Shows two original images from different classes
- Displays 10 mixed images with different lambda values
- Shows how images blend at different mixing ratios
- Labels each mixed image with λ value and percentage mix

##### 2. MixUp Label Visualization (`mixup_labels.png`)
- Shows 10 samples with soft label distributions
- Bar charts showing top 3 classes for each sample
- Demonstrates how labels are mixed into probability distributions
- Color-coded for easy interpretation

##### 3. MixUp Statistics (`mixup_statistics.png`)
- Histogram of lambda distribution over 1000 samples
- Shows Beta(alpha, alpha) distribution characteristics
- Statistics panel with:
  - Mean, std dev, min, max, median of lambda
  - Configuration parameters
  - Expected behavior indicators

## Running the Tests

### Run Component Tests (Non-Visual)
```bash
cd /Users/yash/Documents/ERA/mini-capstone/ERA-V4-Mini-Capstone
python tests/verify_training_components.py
```

Expected output:
- TEST 6: MIXUP VERIFICATION with all 7 sub-tests
- Detailed statistics and verification messages
- ✅ TEST 6 PASSED if all checks pass

### Run Visual Tests
```bash
cd /Users/yash/Documents/ERA/mini-capstone/ERA-V4-Mini-Capstone
python tests/verify_visual.py
```

Expected output:
- Three new visualization files in `tests/verification_outputs/`:
  - `mixup_visualization.png`
  - `mixup_labels.png`
  - `mixup_statistics.png`

## What to Look For

### In Component Tests
1. ✅ All sub-tests should pass
2. Mixed labels should sum to 1.0
3. Lambda values should vary (showing randomness)
4. Training step should work with and without MixUp
5. No errors or warnings

### In Visual Tests
1. **Image Mixing**: Mixed images should be visible blends of originals
2. **Label Mixing**: Soft labels should show non-zero values for multiple classes
3. **Lambda Distribution**: Should follow Beta(0.2, 0.2) distribution
   - Mean should be around 0.5
   - Distribution should be U-shaped (higher density near 0 and 1)

## Understanding MixUp

### What is MixUp?
MixUp is a data augmentation technique that creates synthetic training examples by mixing pairs of samples:

```
mixed_image = λ * image_a + (1-λ) * image_b
mixed_label = λ * label_a + (1-λ) * label_b
```

Where λ (lambda) is sampled from Beta(α, α) distribution.

### Benefits
- Improves generalization
- Reduces memorization
- Acts as regularization
- Smooths decision boundaries
- Better calibration of predictions

### Configuration
- **mixup_alpha**: Controls mixing strength (0.2-1.0 recommended)
  - 0.0 = disabled
  - 0.2 = moderate mixing (recommended)
  - 1.0 = uniform mixing
- **prob**: Probability of applying MixUp (typically 1.0)
- **mode**: How to mix samples ('batch', 'pair', 'elem')

## Troubleshooting

### Test Fails: "MixUp initialization failed"
- Check that `mixup_alpha > 0` in config
- Verify timm library is installed: `pip install timm`

### Test Fails: "Labels don't sum to 1.0"
- This indicates a bug in MixUp implementation
- Check timm version compatibility

### Test Fails: "No labels were mixed"
- May occur randomly with small batches
- Run test multiple times to verify

### Visual Test: Lambda Distribution Not U-shaped
- With alpha=0.2, distribution should have higher density near edges
- If uniform, check mixup_alpha configuration

## Integration with Training

MixUp is integrated into the training pipeline:

1. **Model Initialization**: MixUp is created in `ResnetLightningModule.__init__()`
2. **Training Step**: Applied in `training_step()` before forward pass
3. **Loss Calculation**: Uses soft labels when MixUp is enabled
4. **Validation**: MixUp is NOT applied during validation (as expected)

## Configuration Files

MixUp is configured in:
- `configs/local_config.py`: MIXUP_KWARGS
- `configs/p4_config.py`: MIXUP_KWARGS
- `configs/g5_config.py`: MIXUP_KWARGS

Current default configuration:
```python
MIXUP_KWARGS = {
    'mixup_alpha': 0.2,      # Moderate mixing
    'cutmix_alpha': 0.0,     # CutMix disabled
    'prob': 1.0,             # Always apply
    'mode': 'batch',         # Batch-wise mixing
    'label_smoothing': 0.1,  # Label smoothing
}
```

## References

- MixUp Paper: https://arxiv.org/abs/1710.09412
- timm MixUp: https://github.com/huggingface/pytorch-image-models
- Implementation: `src/models/resnet_module.py`

## Future Enhancements

Possible additional tests:
- CutMix verification (when enabled)
- MixUp with different alpha values
- Performance impact measurement
- Accuracy improvement verification


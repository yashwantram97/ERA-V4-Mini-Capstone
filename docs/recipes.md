# Training Recipes - Medium 🌶️🌶️

This document explains the techniques used in our ImageNet training pipeline to achieve 78-79.5% accuracy efficiently.

Recipies were taken from [mosaicml](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/resnet_imagenet)

---

## Speed-Up Techniques

### ✅ BlurPool (Antialiased Downsampling)

**What it is:**  
BlurPool replaces traditional max pooling and strided convolutions with blur-pooled versions to make CNNs shift-invariant. It applies a blur filter before downsampling to prevent aliasing artifacts.

**How we use it:**
```python
import antialiased_cnns
self.model = antialiased_cnns.resnet50(pretrained=False, filter_size=4)
```

**Benefits:**
- Improves generalization and accuracy by ~0.5-1%
- Makes model more robust to small translations in input
- Minimal computational overhead

**Paper:** [Making Convolutional Networks Shift-Invariant Again (Zhang, 2019)](https://arxiv.org/abs/1904.11486)

---

### ✅ FixRes (Fixed Resolution Fine-tuning)

**What it is:**  
FixRes addresses the train-test resolution discrepancy by fine-tuning the model at a higher resolution than training, but using test-time augmentations (center crop) instead of training augmentations (random crop).

**How we use it:**
```python
# Epochs 85-90: Switch to higher resolution (288px) with test-time augmentations
res_schedule = {
    0: (128, True, 128),    # Train augs
    10: (224, True, 64),    # Train augs
    85: (288, False, 32)    # Test augs (FixRes phase)
}
```

In the FixRes phase (`use_train_augs=False`), we use:
- `Resize` + `CenterCrop` instead of `RandomResizedCrop`
- No random horizontal flips

**Benefits:**
- +1-2% accuracy boost with just 5 epochs of fine-tuning
- Aligns training and test data distributions

**Paper:** [Fixing the train-test resolution discrepancy (Touvron et al., 2019)](https://arxiv.org/abs/1906.06423)

---

### ✅ Label Smoothing

**What it is:**  
Label smoothing prevents the model from becoming overconfident by replacing hard targets (0/1) with soft targets (e.g., 0.1/0.9). This regularizes the model and improves generalization.

**How we use it:**
```python
self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Instead of targeting `[0, 0, 1, 0, ...]`, we target `[0.025, 0.025, 0.9, 0.025, ...]` (with 0.1 smoothing and 1000 classes).

**Benefits:**
- Prevents overfitting
- +0.5-1% accuracy improvement
- More calibrated predictions

**Paper:** [Rethinking the Inception Architecture (Szegedy et al., 2016)](https://arxiv.org/abs/1512.00567)

---

### ✅ Progressive Resizing

**What it is:**  
Start training with small images (128px) and gradually increase resolution (224px, then 288px). This speeds up early training when the model learns coarse features.

**How we use it:**
```python
res_schedule = {
    0: (128, True, 128),    # Epochs 0-10: 128x128
    10: (224, True, 64),    # Epochs 10-85: 224x224
    85: (288, False, 32)    # Epochs 85-90: 288x288
}
```

**Benefits:**
- 2-3x faster training in early epochs
- Better convergence
- Reduces training time by 30-40%

**Popularized by:** Fast.ai ([Tricks from Deep Learning for Coders](https://course.fast.ai/))

---

### ✅ MixUp

**What it is:**  
MixUp augments data by creating synthetic training examples through linear interpolation of pairs of images and their labels.

**How we use it:**
```python
lam = Beta(0.2, 0.2).sample()  # Sample mixing coefficient
mixed_images = lam * images + (1 - lam) * images[shuffled]
mixed_labels = lam * labels_a + (1 - lam) * labels_b
```

For example: `new_image = 0.7 * cat_image + 0.3 * dog_image` with label `[0.7, 0.3]`.

**Benefits:**
- Strong regularization effect
- +1-2% accuracy improvement
- More robust to adversarial examples

**Paper:** [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412)

---

### ✅ SAM (Sharpness Aware Minimization)

**What it is:**  
SAM simultaneously minimizes loss value AND loss sharpness by finding parameters in flat regions of the loss landscape. This is done through a two-step optimization process.

**How we use it:**
```python
from pytorch_optimizer import SAM

optimizer = SAM(
    self.parameters(),
    base_optimizer=torch.optim.SGD,
    lr=0.1, momentum=0.9, weight_decay=1e-4
)

# Two-step update in training_step:
loss = compute_loss(model(x), y)
loss.backward()
optimizer.first_step(zero_grad=True)  # Ascent step

loss2 = compute_loss(model(x), y)
loss2.backward()
optimizer.second_step(zero_grad=True)  # Descent step
```

**Benefits:**
- +1-2% accuracy boost
- Better generalization
- More robust models
- **Note:** ~2x slower training due to double forward-backward pass

**Paper:** [Sharpness-Aware Minimization (Foret et al., 2020)](https://arxiv.org/abs/2010.01412)

---

## Additional Optimizations

### ✅ Channels Last Memory Format

**What it is:**  
Changes memory layout from NCHW (channels first) to NHWC (channels last), which is more efficient for modern GPUs with Tensor Cores.

**How we use it:**
```python
model = model.to(memory_format=torch.channels_last)
```

**Benefits:**
- 10-30% faster training on modern GPUs
- Better memory access patterns
- No accuracy impact

**Reference:** [PyTorch Channels Last Memory Format](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

---

### ✅ Dynamic Batch Sizing

**What it is:**  
Adjust batch size based on input resolution to maximize GPU utilization. Small images (128px) can use larger batches, while large images (288px) need smaller batches to fit in memory.

**How we use it:**
```python
# p3.16xlarge (8 GPUs):
res_schedule = {
    0: (128, True, 512),    # Large batch for small images
    10: (224, True, 320),   # Medium batch
    85: (288, False, 256)   # Smaller batch for large images
}

# MacBook M4 Pro (1 GPU):
res_schedule = {
    0: (128, True, 128),    # Adjusted for single GPU
    10: (224, True, 64),
    85: (288, False, 32)
}
```

**Benefits:**
- Maximizes GPU memory utilization
- 30-50% faster training in early epochs
- No accuracy impact

---

## Combined Impact

When used together, these techniques provide:

- **Training Time:** 80-200 minutes (vs 600+ minutes baseline) on p3.16xlarge
- **Accuracy:** 78.1-79.5% on ImageNet-1K
- **Cost Efficiency:** ~$50-70 per training run on AWS

### Timeline & Accuracy Expectations:

| Epoch | Resolution | Expected Val Accuracy | Key Milestones |
|-------|-----------|---------------------|----------------|
| 0-5   | 128px     | 35% → 68%          | Rapid learning |
| 6-10  | 128px     | 68% → 72%          | Low-res convergence |
| 10-20 | 224px     | 70% → 74%          | Resolution jump recovery |
| 40-60 | 224px     | 75% → 77%          | Steady improvement |
| 70-85 | 224px     | 77% → 78%          | Near convergence |
| 85-90 | 288px     | 78% → 79%+         | FixRes boost |

---

## References

1. Zhang, R. (2019). Making Convolutional Networks Shift-Invariant Again. ICML.
2. Touvron, H., et al. (2019). Fixing the train-test resolution discrepancy. NeurIPS.
3. Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR.
4. Zhang, H., et al. (2017). mixup: Beyond Empirical Risk Minimization. ICLR.
5. Foret, P., et al. (2020). Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR.
6. Howard, J., & Gugger, S. (2020). Deep Learning for Coders with fastai and PyTorch. O'Reilly.

---

## Hardware Configurations

### AWS p3.16xlarge (Production)
- 8x NVIDIA V100 GPUs (16GB VRAM each)
- 64 vCPUs
- Batch sizes: 512 → 320 → 256 per GPU
- Total batch: 4,096 → 2,560 → 2,048
- Training time: ~10-11 hours for full ImageNet-1K

### MacBook M4 Pro (Local/Development)
- 1x Apple Silicon GPU (MPS)
- 4 data loading workers
- Batch sizes: 128 → 64 → 32
- Best for ImageNet-Mini (~35k images)
- Training time: ~2-3 hours for ImageNet-Mini


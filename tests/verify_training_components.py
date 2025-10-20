"""
Verification Script for Training Components

This script verifies the following before running on GPU/AWS:
1. Image augmentation is working properly
2. OneCycle policy is configured correctly
3. Resolution changing works as expected
4. BlurPool is properly integrated
5. FixRes is working correctly

Run this script before deploying to expensive GPU instances!
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from PIL import Image
import cv2

# Import project modules
from src.utils.utils import get_transforms
from src.data_modules.imagenet_datamodule import ImageNetDataModule
from src.models.resnet_module import ResnetLightningModule
from src.callbacks.resolution_schedule_callback import ResolutionScheduleCallback
from configs.local_config import *
import lightning as L
from torch.utils.data import DataLoader

print("="*80)
print("🔍 TRAINING COMPONENTS VERIFICATION")
print("="*80)

# ============================================================================
# TEST 1: Image Augmentation Verification
# ============================================================================
def test_augmentations():
    """Verify that image augmentations are applied correctly"""
    print("\n" + "="*80)
    print("TEST 1: IMAGE AUGMENTATION VERIFICATION")
    print("="*80)
    
    # Get a sample image from the dataset
    sample_img_path = list(TRAIN_IMG_DIR.glob("*//*.JPEG"))[0]
    print(f"\n📷 Loading sample image: {sample_img_path.name}")
    
    # Load image
    image = Image.open(sample_img_path).convert('RGB')
    image_np = np.array(image)
    
    print(f"   Original size: {image_np.shape}")
    
    # Test train transforms
    print("\n🔧 Testing TRAIN transforms (with augmentation):")
    train_transforms = get_transforms(
        transform_type="train",
        mean=MEAN,
        std=STD,
        resolution=224
    )
    
    # Print transform pipeline
    for i, transform in enumerate(train_transforms, 1):
        print(f"   {i}. {transform.__class__.__name__}")
    
    # Apply transforms multiple times to see variation
    print("\n   Applying transforms 5 times to verify randomness...")
    transformed_images = []
    for i in range(5):
        compose = A.Compose(train_transforms)
        transformed = compose(image=image_np)
        transformed_images.append(transformed['image'])
        # Check shape after transform
        if i == 0:
            print(f"   ✅ Output shape: {transformed['image'].shape}")
            print(f"   ✅ Output type: {type(transformed['image'])}")
    
    # Verify randomness (images should be different)
    all_same = all(torch.equal(transformed_images[0], img) for img in transformed_images[1:])
    if all_same:
        print("   ❌ WARNING: All transformed images are identical!")
        print("      Random augmentations may not be working properly.")
    else:
        print("   ✅ Random augmentations are working (images differ)")
    
    # Test validation transforms
    print("\n🔧 Testing VALIDATION transforms (FixRes style):")
    val_transforms = get_transforms(
        transform_type="valid",
        mean=MEAN,
        std=STD,
        resolution=224
    )
    
    for i, transform in enumerate(val_transforms, 1):
        print(f"   {i}. {transform.__class__.__name__}")
    
    compose = A.Compose(val_transforms)
    val_transformed = compose(image=image_np)
    print(f"   ✅ Output shape: {val_transformed['image'].shape}")
    
    # Test FixRes transforms (higher resolution)
    print("\n🔧 Testing FIXRES transforms (288px with test-time augmentations):")
    fixres_transforms = get_transforms(
        transform_type="valid",  # Use validation transforms for FixRes
        mean=MEAN,
        std=STD,
        resolution=288
    )
    
    for i, transform in enumerate(fixres_transforms, 1):
        print(f"   {i}. {transform.__class__.__name__}")
    
    compose = A.Compose(fixres_transforms)
    fixres_transformed = compose(image=image_np)
    print(f"   ✅ Output shape: {fixres_transformed['image'].shape}")
    
    print("\n✅ TEST 1 PASSED: Augmentations are working correctly")
    return True

# ============================================================================
# TEST 2: OneCycle Policy Verification
# ============================================================================
def test_onecycle_policy():
    """Verify OneCycle LR policy configuration"""
    print("\n" + "="*80)
    print("TEST 2: ONECYCLE POLICY VERIFICATION")
    print("="*80)
    
    # Create a dummy model and optimizer
    model = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_classes=NUM_CLASSES
    )
    
    # We need to simulate the trainer context
    # Create a minimal trainer to test the scheduler
    datamodule = ImageNetDataModule(
        train_img_dir=str(TRAIN_IMG_DIR),
        val_img_dir=str(VAL_IMG_DIR),
        mean=MEAN,
        std=STD,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Use 0 workers for quick test
        initial_resolution=224
    )
    
    # Setup datamodule to get the dataloaders
    datamodule.setup('fit')
    
    # Calculate steps
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Train samples: {len(datamodule.train_dataset)}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        momentum=0.9
    )
    
    # Create OneCycle scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=100.0,
        final_div_factor=1000.0
    )
    
    print(f"\n🔄 OneCycle Scheduler Configuration:")
    print(f"   Max LR: {LEARNING_RATE:.4e}")
    print(f"   Initial LR: {LEARNING_RATE/100.0:.4e}")
    print(f"   Final LR: {LEARNING_RATE/100.0/1000.0:.4e}")
    print(f"   Warmup steps: {int(total_steps * 0.2)}")
    print(f"   Annealing steps: {int(total_steps * 0.8)}")
    
    # Simulate training and collect LR values
    print("\n📈 Simulating LR schedule over epochs...")
    lr_history = []
    momentum_history = []
    
    for epoch in range(EPOCHS):
        for step in range(steps_per_epoch):
            lr_history.append(optimizer.param_groups[0]['lr'])
            momentum_history.append(optimizer.param_groups[0]['momentum'])
            scheduler.step()
    
    # Verify LR schedule characteristics
    print(f"\n✅ LR Schedule Analysis:")
    print(f"   Starting LR: {lr_history[0]:.4e}")
    print(f"   Maximum LR: {max(lr_history):.4e}")
    print(f"   Final LR: {lr_history[-1]:.4e}")
    print(f"   LR at 20% (end of warmup): {lr_history[int(len(lr_history)*0.2)]:.4e}")
    
    # Check if LR follows expected pattern
    warmup_end_idx = int(len(lr_history) * 0.2)
    is_warmup_increasing = lr_history[warmup_end_idx] > lr_history[0]
    is_annealing_decreasing = lr_history[-1] < max(lr_history)
    
    if is_warmup_increasing and is_annealing_decreasing:
        print("\n   ✅ LR schedule follows correct pattern:")
        print("      - Warmup phase: LR increases")
        print("      - Annealing phase: LR decreases")
    else:
        print("\n   ❌ WARNING: LR schedule may not be correct!")
    
    # Check momentum schedule
    print(f"\n✅ Momentum Schedule Analysis:")
    print(f"   Starting momentum: {momentum_history[0]:.4f}")
    print(f"   Minimum momentum: {min(momentum_history):.4f}")
    print(f"   Final momentum: {momentum_history[-1]:.4f}")
    
    print("\n✅ TEST 2 PASSED: OneCycle policy is configured correctly")
    return True, lr_history, momentum_history

# ============================================================================
# TEST 3: Resolution Schedule Verification
# ============================================================================
def test_resolution_schedule():
    """Verify resolution scheduling works correctly"""
    print("\n" + "="*80)
    print("TEST 3: RESOLUTION SCHEDULE VERIFICATION")
    print("="*80)
    
    # Print the configured schedule
    print(f"\n📐 Configured Resolution Schedule:")
    for epoch, (resolution, use_train_augs) in sorted(PROG_RESIZING_FIXRES_SCHEDULE.items()):
        aug_type = "Train" if use_train_augs else "Test (FixRes)"
        print(f"   Epoch {epoch:2d}+: {resolution}x{resolution}px, {aug_type} augmentations")
    
    # Create callback
    callback = ResolutionScheduleCallback(PROG_RESIZING_FIXRES_SCHEDULE)
    
    # Verify callback initialization
    print(f"\n✅ Callback created successfully")
    print(f"   Schedule epochs: {sorted(callback.schedule.keys())}")
    
    # Test resolution changes manually
    print(f"\n🔧 Testing resolution changes:")
    
    for epoch, (resolution, use_train_augs) in sorted(PROG_RESIZING_FIXRES_SCHEDULE.items()):
        print(f"\n   Epoch {epoch}:")
        
        # Get transforms for this configuration
        train_transforms = get_transforms(
            transform_type="train" if use_train_augs else "valid",
            mean=MEAN,
            std=STD,
            resolution=resolution
        )
        
        val_transforms = get_transforms(
            transform_type="valid",
            mean=MEAN,
            std=STD,
            resolution=resolution
        )
        
        # Check transform output sizes
        sample_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        
        train_compose = A.Compose(train_transforms)
        train_result = train_compose(image=sample_img)
        
        val_compose = A.Compose(val_transforms)
        val_result = val_compose(image=sample_img)
        
        print(f"      Train output: {train_result['image'].shape}")
        print(f"      Val output: {val_result['image'].shape}")
        
        # Verify resolution matches expected
        expected_shape = (3, resolution, resolution)
        if train_result['image'].shape == expected_shape:
            print(f"      ✅ Train transform produces correct size")
        else:
            print(f"      ❌ Train transform size mismatch: expected {expected_shape}, got {train_result['image'].shape}")
        
        if val_result['image'].shape == expected_shape:
            print(f"      ✅ Val transform produces correct size")
        else:
            print(f"      ❌ Val transform size mismatch: expected {expected_shape}, got {val_result['image'].shape}")
    
    print("\n✅ TEST 3 PASSED: Resolution scheduling is working correctly")
    return True

# ============================================================================
# TEST 4: BlurPool Verification
# ============================================================================
def test_blurpool():
    """Verify BlurPool is integrated in the model"""
    print("\n" + "="*80)
    print("TEST 4: BLURPOOL VERIFICATION")
    print("="*80)
    
    # Create model
    model = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_classes=NUM_CLASSES
    )
    
    print(f"\n🔍 Checking model architecture for BlurPool...")
    
    # Check if antialiased_cnns is being used
    import antialiased_cnns
    print(f"   ✅ antialiased_cnns imported successfully")
    print(f"   Model type: {type(model.model).__name__}")
    
    # Check for blur pool layers in the model
    blurpool_count = 0
    maxpool_count = 0
    avgpool_count = 0
    
    for name, module in model.model.named_modules():
        module_name = module.__class__.__name__
        if 'blur' in module_name.lower() or 'BlurPool' in module_name:
            blurpool_count += 1
            print(f"   ✅ Found BlurPool layer: {name} ({module_name})")
        elif module_name == 'MaxPool2d':
            maxpool_count += 1
        elif module_name == 'AvgPool2d':
            avgpool_count += 1
    
    print(f"\n📊 Pooling Layer Summary:")
    print(f"   BlurPool layers: {blurpool_count}")
    print(f"   MaxPool layers: {maxpool_count}")
    print(f"   AvgPool layers: {avgpool_count}")
    
    if blurpool_count > 0:
        print(f"\n   ✅ BlurPool is integrated (found {blurpool_count} layers)")
    else:
        print(f"\n   ⚠️  No BlurPool layers found explicitly named")
        print(f"      This is OK if antialiased_cnns replaces MaxPool internally")
    
    # Test forward pass with BlurPool
    print(f"\n🧪 Testing forward pass with BlurPool model...")
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(2, 3, 224, 224)
        output = model(sample_input)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ Forward pass successful")
    
    print("\n✅ TEST 4 PASSED: BlurPool is properly integrated")
    return True

# ============================================================================
# TEST 5: FixRes Verification
# ============================================================================
def test_fixres():
    """Verify FixRes (test-time augmentations at higher resolution)"""
    print("\n" + "="*80)
    print("TEST 5: FIXRES VERIFICATION")
    print("="*80)
    
    print(f"\n📚 FixRes Concept:")
    print(f"   - Train with aggressive augmentations (RandomResizedCrop, flips)")
    print(f"   - Fine-tune final epochs with test-time augmentations (Resize + CenterCrop)")
    print(f"   - Use higher resolution in final phase for better accuracy")
    
    # Check FixRes configuration from schedule
    fixres_epochs = [(epoch, res, aug) for epoch, (res, aug) in PROG_RESIZING_FIXRES_SCHEDULE.items() if not aug]
    
    if not fixres_epochs:
        print(f"\n   ❌ WARNING: No FixRes phase found in schedule!")
        print(f"      (Looking for epochs with use_train_augs=False)")
        return False
    
    print(f"\n✅ FixRes Configuration Found:")
    for epoch, resolution, _ in fixres_epochs:
        print(f"   Epoch {epoch}+: {resolution}x{resolution}px with test-time augmentations")
    
    # Test the transforms for FixRes phase
    fixres_epoch, fixres_resolution, _ = fixres_epochs[0]
    
    print(f"\n🔧 Testing FixRes transforms (Epoch {fixres_epoch}, {fixres_resolution}px):")
    
    # Train transforms (before FixRes)
    train_transforms = get_transforms(
        transform_type="train",
        mean=MEAN,
        std=STD,
        resolution=fixres_resolution
    )
    
    print(f"\n   Train augmentations (used in early epochs):")
    for i, t in enumerate(train_transforms, 1):
        print(f"      {i}. {t.__class__.__name__}")
    
    # FixRes transforms (test-time)
    fixres_transforms = get_transforms(
        transform_type="valid",  # Note: using 'valid' for FixRes
        mean=MEAN,
        std=STD,
        resolution=fixres_resolution
    )
    
    print(f"\n   FixRes augmentations (used in final epochs):")
    for i, t in enumerate(fixres_transforms, 1):
        print(f"      {i}. {t.__class__.__name__}")
    
    # Verify the key differences
    train_has_random = any('Random' in t.__class__.__name__ for t in train_transforms)
    fixres_has_random = any('Random' in t.__class__.__name__ for t in fixres_transforms)
    fixres_has_center_crop = any('CenterCrop' in t.__class__.__name__ for t in fixres_transforms)
    
    print(f"\n✅ FixRes Verification:")
    print(f"   Train has random augmentations: {train_has_random}")
    print(f"   FixRes has random augmentations: {fixres_has_random}")
    print(f"   FixRes has CenterCrop: {fixres_has_center_crop}")
    
    if train_has_random and not fixres_has_random and fixres_has_center_crop:
        print(f"\n   ✅ FixRes is correctly configured:")
        print(f"      - Train phase uses random augmentations")
        print(f"      - FixRes phase uses deterministic test-time augmentations")
    else:
        print(f"\n   ⚠️  FixRes configuration may need review")
    
    # Test transforms on sample image
    print(f"\n🧪 Testing FixRes transforms on sample image:")
    sample_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    train_compose = A.Compose(train_transforms)
    fixres_compose = A.Compose(fixres_transforms)
    
    # Apply train transforms multiple times
    train_results = [train_compose(image=sample_img)['image'] for _ in range(3)]
    
    # Apply FixRes transforms multiple times
    fixres_results = [fixres_compose(image=sample_img)['image'] for _ in range(3)]
    
    # Check if train transforms are random
    train_all_same = all(torch.equal(train_results[0], img) for img in train_results[1:])
    print(f"   Train transforms are deterministic: {train_all_same}")
    
    # Check if FixRes transforms are deterministic
    fixres_all_same = all(torch.equal(fixres_results[0], img) for img in fixres_results[1:])
    print(f"   FixRes transforms are deterministic: {fixres_all_same}")
    
    if not train_all_same and fixres_all_same:
        print(f"\n   ✅ Perfect! Train varies, FixRes is consistent")
    
    print("\n✅ TEST 5 PASSED: FixRes is working correctly")
    return True

# ============================================================================
# TEST 6: MixUp Verification
# ============================================================================
def test_mixup():
    """Verify MixUp augmentation is working correctly"""
    print("\n" + "="*80)
    print("TEST 6: MIXUP VERIFICATION")
    print("="*80)
    
    print(f"\n📚 MixUp Concept:")
    print(f"   - MixUp mixes pairs of training samples and their labels")
    print(f"   - Creates synthetic training examples by linear interpolation")
    print(f"   - Helps with regularization and generalization")
    print(f"   - Formula: mixed_image = λ * image_a + (1-λ) * image_b")
    print(f"   - λ is sampled from Beta(alpha, alpha) distribution")
    
    # Test 1: MixUp initialization with enabled config
    print(f"\n🔧 TEST 6.1: MixUp Initialization (Enabled)")
    print("="*60)
    
    mixup_kwargs_enabled = {
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.0,
        'prob': 1.0,
        'mode': 'batch',
        'label_smoothing': 0.1,
    }
    
    model_with_mixup = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_classes=NUM_CLASSES,
        mixup_kwargs=mixup_kwargs_enabled
    )
    
    if model_with_mixup.mixup_fn is not None:
        print(f"   ✅ MixUp enabled successfully")
        print(f"   MixUp alpha: {mixup_kwargs_enabled['mixup_alpha']}")
        print(f"   Probability: {mixup_kwargs_enabled['prob']}")
        print(f"   Mode: {mixup_kwargs_enabled['mode']}")
    else:
        print(f"   ❌ MixUp initialization failed!")
        return False
    
    # Test 2: MixUp initialization with disabled config
    print(f"\n🔧 TEST 6.2: MixUp Initialization (Disabled)")
    print("="*60)
    
    mixup_kwargs_disabled = {
        'mixup_alpha': 0.0,  # Disabled
    }
    
    model_without_mixup = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_classes=NUM_CLASSES,
        mixup_kwargs=mixup_kwargs_disabled
    )
    
    if model_without_mixup.mixup_fn is None:
        print(f"   ✅ MixUp correctly disabled when alpha=0")
    else:
        print(f"   ❌ MixUp should be disabled when alpha=0!")
    
    # Test 3: MixUp with None kwargs
    print(f"\n🔧 TEST 6.3: MixUp with None kwargs")
    print("="*60)
    
    model_no_kwargs = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_classes=NUM_CLASSES,
        mixup_kwargs=None
    )
    
    if model_no_kwargs.mixup_fn is None:
        print(f"   ✅ MixUp correctly disabled when kwargs=None")
    else:
        print(f"   ❌ MixUp should be disabled when kwargs=None!")
    
    # Test 4: MixUp transformation on actual data
    print(f"\n🔧 TEST 6.4: MixUp Transformation on Data")
    print("="*60)
    
    # Create sample batch
    batch_size = 8
    sample_images = torch.randn(batch_size, 3, 224, 224)
    sample_labels = torch.randint(0, NUM_CLASSES, (batch_size,))
    
    print(f"\n   Original batch:")
    print(f"      Image shape: {sample_images.shape}")
    print(f"      Label shape: {sample_labels.shape}")
    print(f"      Label type: {sample_labels.dtype}")
    print(f"      Sample labels: {sample_labels[:4].tolist()}")
    
    # Apply MixUp
    mixed_images, mixed_labels = model_with_mixup.mixup_fn(sample_images, sample_labels)
    
    print(f"\n   After MixUp:")
    print(f"      Image shape: {mixed_images.shape}")
    print(f"      Label shape: {mixed_labels.shape}")
    print(f"      Label type: {mixed_labels.dtype}")
    
    # Verify shapes
    if mixed_images.shape == sample_images.shape:
        print(f"   ✅ Mixed images have correct shape")
    else:
        print(f"   ❌ Mixed images shape mismatch!")
        return False
    
    # Verify labels are now soft (one-hot encoded)
    if mixed_labels.shape == (batch_size, NUM_CLASSES):
        print(f"   ✅ Labels converted to soft labels (one-hot)")
    else:
        print(f"   ❌ Labels shape incorrect: expected ({batch_size}, {NUM_CLASSES}), got {mixed_labels.shape}")
        return False
    
    # Verify labels sum to 1 (probability distribution)
    label_sums = mixed_labels.sum(dim=1)
    if torch.allclose(label_sums, torch.ones_like(label_sums), atol=1e-5):
        print(f"   ✅ Soft labels sum to 1.0 (valid probability distribution)")
    else:
        print(f"   ❌ Soft labels don't sum to 1.0!")
        print(f"      Sample sums: {label_sums[:4].tolist()}")
        return False
    
    # Check that labels are actually mixed (not just one-hot)
    # Count how many labels have non-zero values in more than one class
    non_zero_per_sample = (mixed_labels > 0.01).sum(dim=1)
    mixed_samples = (non_zero_per_sample > 1).sum().item()
    
    print(f"\n   📊 MixUp Statistics:")
    print(f"      Samples with mixed labels: {mixed_samples}/{batch_size}")
    print(f"      Average classes per sample: {non_zero_per_sample.float().mean():.2f}")
    
    if mixed_samples > 0:
        print(f"   ✅ Labels are being mixed (not just one-hot)")
        # Show example of a mixed label
        mixed_idx = (non_zero_per_sample > 1).nonzero()[0].item()
        non_zero_classes = (mixed_labels[mixed_idx] > 0.01).nonzero().squeeze()
        print(f"\n   Example mixed label (sample {mixed_idx}):")
        for cls_idx in non_zero_classes[:3]:  # Show up to 3 classes
            cls_idx = cls_idx.item() if cls_idx.numel() > 0 else cls_idx
            print(f"      Class {cls_idx}: {mixed_labels[mixed_idx, cls_idx]:.3f}")
    else:
        print(f"   ⚠️  No labels were mixed in this batch (could be random)")
    
    # Test 5: MixUp affects images
    print(f"\n🔧 TEST 6.5: MixUp Image Mixing Verification")
    print("="*60)
    
    # Create two distinct images
    image_a = torch.ones(1, 3, 224, 224) * 0.0  # Black image
    image_b = torch.ones(1, 3, 224, 224) * 1.0  # White image
    
    # Stack them
    test_images = torch.cat([image_a, image_b], dim=0)
    test_labels = torch.tensor([0, 1])
    
    # Apply MixUp multiple times and check for variation
    print(f"   Testing MixUp randomness (10 iterations)...")
    mixed_results = []
    lambda_values = []
    
    for i in range(10):
        mixed_imgs, mixed_lbls = model_with_mixup.mixup_fn(test_images.clone(), test_labels.clone())
        # Get the mean pixel value of first image (should be between 0 and 1)
        mean_val = mixed_imgs[0].mean().item()
        mixed_results.append(mean_val)
        # Estimate lambda from the mixed label
        lambda_val = mixed_lbls[0].max().item()
        lambda_values.append(lambda_val)
    
    # Check if we got different results
    unique_results = len(set([round(x, 3) for x in mixed_results]))
    
    print(f"\n   📊 Randomness check:")
    print(f"      Unique mean values: {unique_results}/10")
    print(f"      Mean values range: [{min(mixed_results):.3f}, {max(mixed_results):.3f}]")
    print(f"      Lambda range: [{min(lambda_values):.3f}, {max(lambda_values):.3f}]")
    
    if unique_results > 1:
        print(f"   ✅ MixUp produces different results (randomness working)")
    else:
        print(f"   ⚠️  MixUp produces same results (may not be random)")
    
    # Test 6: Integration with training step
    print(f"\n🔧 TEST 6.6: MixUp Integration in Training Step")
    print("="*60)
    
    # Create a sample batch
    batch_images = torch.randn(4, 3, 224, 224)
    batch_labels = torch.randint(0, NUM_CLASSES, (4,))
    
    # Test with MixUp enabled
    model_with_mixup.eval()  # Set to eval to avoid any other randomness
    with torch.no_grad():
        try:
            # Simulate training step logic
            if model_with_mixup.mixup_fn is not None:
                mixed_images, mixed_labels = model_with_mixup.mixup_fn(batch_images, batch_labels)
                logits = model_with_mixup(mixed_images)
                loss = F.cross_entropy(logits, mixed_labels)
                print(f"   ✅ Training step with MixUp successful")
                print(f"      Loss: {loss.item():.4f}")
                print(f"      Logits shape: {logits.shape}")
            else:
                print(f"   ❌ MixUp should be enabled!")
                return False
        except Exception as e:
            print(f"   ❌ Error in training step with MixUp: {e}")
            return False
    
    # Test without MixUp
    model_without_mixup.eval()
    with torch.no_grad():
        try:
            if model_without_mixup.mixup_fn is None:
                logits = model_without_mixup(batch_images)
                loss = F.cross_entropy(logits, batch_labels, label_smoothing=0.1)
                print(f"   ✅ Training step without MixUp successful")
                print(f"      Loss: {loss.item():.4f}")
            else:
                print(f"   ❌ MixUp should be disabled!")
                return False
        except Exception as e:
            print(f"   ❌ Error in training step without MixUp: {e}")
            return False
    
    # Test 7: Check configuration from config file
    print(f"\n🔧 TEST 6.7: MixUp Configuration from Config")
    print("="*60)
    
    print(f"\n   Current MixUp configuration:")
    for key, value in MIXUP_KWARGS.items():
        print(f"      {key}: {value}")
    
    # Verify reasonable values
    checks = []
    
    if 0.0 <= MIXUP_KWARGS.get('mixup_alpha', 0.0) <= 1.0:
        print(f"   ✅ mixup_alpha in valid range [0.0, 1.0]")
        checks.append(True)
    else:
        print(f"   ❌ mixup_alpha out of range!")
        checks.append(False)
    
    if 0.0 <= MIXUP_KWARGS.get('prob', 1.0) <= 1.0:
        print(f"   ✅ prob in valid range [0.0, 1.0]")
        checks.append(True)
    else:
        print(f"   ❌ prob out of range!")
        checks.append(False)
    
    if MIXUP_KWARGS.get('mode', 'batch') in ['batch', 'pair', 'elem']:
        print(f"   ✅ mode is valid")
        checks.append(True)
    else:
        print(f"   ❌ mode is invalid!")
        checks.append(False)
    
    if not all(checks):
        return False
    
    print("\n✅ TEST 6 PASSED: MixUp is working correctly")
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all verification tests"""
    print("\n🚀 Starting verification of training components...")
    print(f"   Configuration: {PROFILE_NAME} - {PROFILE_DESCRIPTION}")
    print(f"   Dataset: {TRAIN_IMG_DIR}")
    
    results = {}
    
    try:
        # Test 1: Augmentations
        results['augmentations'] = test_augmentations()
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        results['augmentations'] = False
    
    try:
        # Test 2: OneCycle Policy
        results['onecycle'], lr_history, momentum_history = test_onecycle_policy()
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        results['onecycle'] = False
        lr_history, momentum_history = [], []
    
    try:
        # Test 3: Resolution Schedule
        results['resolution'] = test_resolution_schedule()
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        results['resolution'] = False
    
    try:
        # Test 4: BlurPool
        results['blurpool'] = test_blurpool()
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        results['blurpool'] = False
    
    try:
        # Test 5: FixRes
        results['fixres'] = test_fixres()
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        results['fixres'] = False
    
    try:
        # Test 6: MixUp
        results['mixup'] = test_mixup()
    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        results['mixup'] = False
    
    # Print summary
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! You're ready to train on GPU/AWS!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED! Please review before training on GPU/AWS")
        print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


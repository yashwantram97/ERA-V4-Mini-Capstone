"""
Visual Verification Script - Quick visual checks with plots and images

This creates visual outputs to verify:
1. Augmentation samples
2. LR schedule plot
3. Resolution schedule timeline
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from PIL import Image
import cv2

# Import project modules
from src.utils.utils import get_transforms
from configs.local_config import *

print("="*80)
print("🎨 VISUAL VERIFICATION OF TRAINING COMPONENTS")
print("="*80)

# Create output directory for verification images (inside tests folder)
output_dir = Path(__file__).parent / "verification_outputs"
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Saving verification outputs to: {output_dir}")

# ============================================================================
# 1. Visualize Augmentations
# ============================================================================
def visualize_augmentations():
    """Create a grid showing different augmentations"""
    print("\n" + "="*80)
    print("1️⃣  VISUALIZING AUGMENTATIONS")
    print("="*80)
    
    # Get a sample image
    sample_img_path = list(TRAIN_IMG_DIR.glob("*/*.JPEG"))[0]
    print(f"\n📷 Sample image: {sample_img_path.name}")
    
    image = Image.open(sample_img_path).convert('RGB')
    image_np = np.array(image)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Image Augmentation Verification', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Train augmentations at 224px
    train_transforms_224 = get_transforms("train", MEAN, STD, 224)
    
    for idx in range(1, 5):
        transformed = train_transforms_224(image=image_np)
        img_tensor = transformed['image']
        
        # Denormalize for visualization
        img_display = img_tensor.permute(1, 2, 0).numpy()
        img_display = img_display * np.array(STD) + np.array(MEAN)
        img_display = np.clip(img_display, 0, 1)
        
        axes[0, idx].imshow(img_display)
        axes[0, idx].set_title(f'Train Aug #{idx}\n(224px)', fontsize=10)
        axes[0, idx].axis('off')
    
    # Train augmentations at 288px (higher resolution)
    train_transforms_288 = get_transforms("train", MEAN, STD, 288)
    
    for idx in range(5):
        transformed = train_transforms_288(image=image_np)
        img_tensor = transformed['image']
        
        # Denormalize for visualization
        img_display = img_tensor.permute(1, 2, 0).numpy()
        img_display = img_display * np.array(STD) + np.array(MEAN)
        img_display = np.clip(img_display, 0, 1)
        
        axes[1, idx].imshow(img_display)
        axes[1, idx].set_title(f'Train Aug #{idx+1}\n(288px)', fontsize=10)
        axes[1, idx].axis('off')
    
    # FixRes augmentations (test-time) at 288px
    fixres_transforms = get_transforms("valid", MEAN, STD, 288)
    
    # FixRes should be deterministic, so all should be same
    for idx in range(5):
        transformed = fixres_transforms(image=image_np)
        img_tensor = transformed['image']
        
        # Denormalize for visualization
        img_display = img_tensor.permute(1, 2, 0).numpy()
        img_display = img_display * np.array(STD) + np.array(MEAN)
        img_display = np.clip(img_display, 0, 1)
        
        axes[2, idx].imshow(img_display)
        axes[2, idx].set_title(f'FixRes #{idx+1}\n(288px, test augs)', fontsize=10)
        axes[2, idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "augmentations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved augmentation visualization: {output_path}")
    plt.close()
    
    # Verify that FixRes transforms are deterministic
    fixres_1 = fixres_transforms(image=image_np)['image']
    fixres_2 = fixres_transforms(image=image_np)['image']
    
    if torch.equal(fixres_1, fixres_2):
        print("✅ FixRes transforms are deterministic (same output each time)")
    else:
        print("⚠️  FixRes transforms vary - they should be deterministic!")
    
    # Verify that train transforms are random
    train_1 = train_transforms_224(image=image_np)['image']
    train_2 = train_transforms_224(image=image_np)['image']
    
    if not torch.equal(train_1, train_2):
        print("✅ Train transforms are random (different output each time)")
    else:
        print("⚠️  Train transforms don't vary - randomness may not be working!")

# ============================================================================
# 2. Visualize OneCycle LR Schedule
# ============================================================================
def visualize_onecycle():
    """Plot the OneCycle learning rate and momentum schedule"""
    print("\n" + "="*80)
    print("2️⃣  VISUALIZING ONECYCLE LR SCHEDULE")
    print("="*80)
    
    # Calculate training steps
    # Approximate number of batches
    approx_train_samples = 130000  # ImageNet-mini
    steps_per_epoch = approx_train_samples // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    
    print(f"\n📊 Training info:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    
    # Create dummy optimizer
    dummy_model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(
        dummy_model.parameters(),
        lr=LEARNING_RATE,
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
    
    # Collect LR and momentum values
    lr_history = []
    momentum_history = []
    step_numbers = []
    
    for step in range(total_steps):
        lr_history.append(optimizer.param_groups[0]['lr'])
        momentum_history.append(optimizer.param_groups[0]['momentum'])
        step_numbers.append(step)
        scheduler.step()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Learning Rate Schedule
    epochs_x = np.array(step_numbers) / steps_per_epoch
    ax1.plot(epochs_x, lr_history, linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_title('OneCycle Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for schedule phases
    warmup_epoch = EPOCHS * 0.2
    ax1.axvline(warmup_epoch, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    ax1.legend()
    
    # Add annotations
    ax1.annotate(f'Max LR: {max(lr_history):.2e}', 
                xy=(warmup_epoch, max(lr_history)),
                xytext=(warmup_epoch + 5, max(lr_history)),
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.annotate(f'Start LR: {lr_history[0]:.2e}', 
                xy=(0, lr_history[0]),
                xytext=(5, lr_history[0] * 2),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax1.annotate(f'Final LR: {lr_history[-1]:.2e}', 
                xy=(EPOCHS, lr_history[-1]),
                xytext=(EPOCHS - 10, lr_history[-1] * 2),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 2: Momentum Schedule
    ax2.plot(epochs_x, momentum_history, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Momentum', fontsize=12, fontweight='bold')
    ax2.set_title('OneCycle Momentum Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(warmup_epoch, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    ax2.legend()
    
    plt.tight_layout()
    output_path = output_dir / "onecycle_schedule.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved OneCycle schedule plot: {output_path}")
    plt.close()
    
    # Print key statistics
    print(f"\n📈 OneCycle Statistics:")
    print(f"   Initial LR: {lr_history[0]:.6e}")
    print(f"   Max LR: {max(lr_history):.6e}")
    print(f"   Final LR: {lr_history[-1]:.6e}")
    print(f"   LR Range: {max(lr_history) / lr_history[0]:.1f}x")
    print(f"   Warmup ends at epoch: {warmup_epoch:.1f}")

# ============================================================================
# 3. Visualize Resolution Schedule
# ============================================================================
def visualize_resolution_schedule():
    """Create a timeline showing resolution changes"""
    print("\n" + "="*80)
    print("3️⃣  VISUALIZING RESOLUTION SCHEDULE")
    print("="*80)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort schedule by epoch
    schedule_items = sorted(PROG_RESIZING_FIXRES_SCHEDULE.items())
    
    # Create bars for each phase
    colors = ['#3A86FF', '#8338EC', '#FF006E']
    
    for i, (start_epoch, (resolution, use_train_augs)) in enumerate(schedule_items):
        # Calculate duration
        if i < len(schedule_items) - 1:
            end_epoch = schedule_items[i + 1][0]
        else:
            end_epoch = EPOCHS
        
        duration = end_epoch - start_epoch
        
        # Create bar
        aug_type = "Train Augs" if use_train_augs else "FixRes (Test Augs)"
        label = f"{resolution}px - {aug_type}"
        
        ax.barh(0, duration, left=start_epoch, height=0.5, 
               color=colors[i % len(colors)], alpha=0.8,
               edgecolor='black', linewidth=2,
               label=label)
        
        # Add text annotation
        mid_epoch = start_epoch + duration / 2
        ax.text(mid_epoch, 0, f"{resolution}px\n{aug_type}\n({duration} epochs)",
               ha='center', va='center', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title(f'Progressive Resizing + FixRes Schedule ({EPOCHS} epochs)', 
                fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    output_path = output_dir / "resolution_schedule.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved resolution schedule plot: {output_path}")
    plt.close()
    
    # Print schedule details
    print(f"\n📐 Resolution Schedule Details:")
    for start_epoch, (resolution, use_train_augs) in schedule_items:
        aug_type = "Train" if use_train_augs else "FixRes"
        print(f"   Epoch {start_epoch:2d}+: {resolution}x{resolution}px ({aug_type})")

# ============================================================================
# 4. Test BlurPool with visual comparison
# ============================================================================
def visualize_blurpool():
    """Show that BlurPool is integrated"""
    print("\n" + "="*80)
    print("4️⃣  BLURPOOL VERIFICATION")
    print("="*80)
    
    from src.models.resnet_module import ResnetLightningModule
    
    # Create model
    model = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        num_classes=NUM_CLASSES
    )
    
    # Count different pooling layers
    blurpool_layers = []
    maxpool_layers = []
    
    for name, module in model.model.named_modules():
        module_name = module.__class__.__name__
        if 'blur' in module_name.lower() or 'BlurPool' in module_name:
            blurpool_layers.append(name)
        elif module_name == 'MaxPool2d':
            maxpool_layers.append(name)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = ['BlurPool Layers', 'MaxPool Layers']
    counts = [len(blurpool_layers), len(maxpool_layers)]
    colors_bar = ['#06D6A0', '#FF6B6B']
    
    bars = ax.bar(categories, counts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}',
               ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    ax.set_ylabel('Number of Layers', fontsize=14, fontweight='bold')
    ax.set_title('BlurPool Integration in ResNet50', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.3 if max(counts) > 0 else 10)
    
    # Add explanation text
    if len(blurpool_layers) > 0:
        status_text = f"✅ BlurPool is ACTIVE\n{len(blurpool_layers)} layers using anti-aliased pooling"
        color_text = 'green'
    else:
        status_text = f"⚠️ BlurPool layers not explicitly found\n(may be integrated internally by antialiased_cnns)"
        color_text = 'orange'
    
    ax.text(0.5, 0.95, status_text,
           transform=ax.transAxes,
           ha='center', va='top',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=color_text, alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / "blurpool_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved BlurPool verification plot: {output_path}")
    plt.close()
    
    print(f"\n🔍 BlurPool Details:")
    print(f"   BlurPool layers found: {len(blurpool_layers)}")
    if blurpool_layers:
        for layer in blurpool_layers[:5]:  # Show first 5
            print(f"      - {layer}")
    print(f"   MaxPool layers found: {len(maxpool_layers)}")

# ============================================================================
# 5. Visualize MixUp
# ============================================================================
def visualize_mixup():
    """Visualize how MixUp mixes images and labels"""
    print("\n" + "="*80)
    print("5️⃣  VISUALIZING MIXUP")
    print("="*80)
    
    from src.models.resnet_module import ResnetLightningModule
    from timm.data.mixup import Mixup
    
    # Create model with MixUp enabled
    mixup_kwargs = {
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.0,
        'prob': 1.0,
        'mode': 'batch',
        'label_smoothing': 0.1,
        'num_classes': NUM_CLASSES
    }
    
    model = ResnetLightningModule(
        learning_rate=LEARNING_RATE,
        num_classes=NUM_CLASSES,
        mixup_kwargs=mixup_kwargs
    )
    
    if model.mixup_fn is None:
        print("⚠️  MixUp is disabled. Skipping visualization.")
        return
    
    # Load two sample images from different classes
    class_dirs = list(TRAIN_IMG_DIR.glob("*"))[:2]  # Get first 2 classes
    
    if len(class_dirs) < 2:
        print("⚠️  Need at least 2 classes in dataset. Skipping visualization.")
        return
    
    # Get one image from each class
    img1_path = list(class_dirs[0].glob("*.JPEG"))[0]
    img2_path = list(class_dirs[1].glob("*.JPEG"))[0]
    
    print(f"\n📷 Loading sample images:")
    print(f"   Image 1: {img1_path.parent.name}/{img1_path.name}")
    print(f"   Image 2: {img2_path.parent.name}/{img2_path.name}")
    
    # Load and preprocess images
    from src.utils.utils import get_transforms
    
    # Get transforms
    transforms = get_transforms("train", MEAN, STD, 224)
    
    # Load images
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Apply transforms (without mixup)
    compose = A.Compose(transforms)
    img1_tensor = compose(image=img1_np)['image']
    img2_tensor = compose(image=img2_np)['image']
    
    # Create batch
    images_batch = torch.stack([img1_tensor, img2_tensor], dim=0)
    labels_batch = torch.tensor([0, 1])  # Different labels
    
    # Apply MixUp with different lambda values
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('MixUp Visualization - Mixing Two Images', fontsize=16, fontweight='bold')
    
    # Row 1: Original images
    def denormalize(img_tensor):
        """Denormalize image for visualization"""
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array(STD) + np.array(MEAN)
        img = np.clip(img, 0, 1)
        return img
    
    axes[0, 0].imshow(denormalize(img1_tensor))
    axes[0, 0].set_title('Image A\n(Class 0)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denormalize(img2_tensor))
    axes[0, 1].set_title('Image B\n(Class 1)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hide remaining axes in first row
    for i in range(2, 5):
        axes[0, i].axis('off')
    
    # Row 2 & 3: MixUp with different lambda values (random from Beta distribution)
    print(f"\n🎨 Generating MixUp examples...")
    
    for idx in range(10):
        row = 1 + (idx // 5)
        col = idx % 5
        
        # Apply MixUp
        mixed_imgs, mixed_lbls = model.mixup_fn(images_batch.clone(), labels_batch.clone())
        
        # Get the mixed version of first image
        mixed_img = denormalize(mixed_imgs[0])
        
        # Get lambda (mixing coefficient) from the label
        lambda_val = mixed_lbls[0, 0].item()  # Proportion of class 0
        
        axes[row, col].imshow(mixed_img)
        axes[row, col].set_title(f'λ={lambda_val:.2f}\n({lambda_val*100:.0f}% A, {(1-lambda_val)*100:.0f}% B)', 
                                fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "mixup_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved MixUp visualization: {output_path}")
    plt.close()
    
    # Create a second plot showing label mixing
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('MixUp Label Mixing - Soft Labels', fontsize=16, fontweight='bold')
    
    print(f"\n📊 Generating label mixing examples...")
    
    # Create batch with multiple classes for visualization
    num_samples = 10
    sample_images = torch.randn(num_samples, 3, 224, 224)
    sample_labels = torch.arange(num_samples) % NUM_CLASSES
    
    # Apply MixUp
    mixed_imgs, mixed_lbls = model.mixup_fn(sample_images, sample_labels)
    
    for idx in range(10):
        row = idx // 5
        col = idx % 5
        
        # Get top 3 classes for this sample
        top_probs, top_classes = torch.topk(mixed_lbls[idx], k=min(3, NUM_CLASSES))
        
        # Create bar chart
        ax = axes[row, col]
        bars = ax.bar(range(len(top_probs)), top_probs.numpy(), color='skyblue', edgecolor='black')
        
        # Color the bars
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(range(len(top_probs)))
        ax.set_xticklabels([f'C{c}' for c in top_classes.numpy()], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_title(f'Sample {idx+1}', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, top_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / "mixup_labels.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved MixUp label visualization: {output_path}")
    plt.close()
    
    # Statistics plot
    print(f"\n📈 Creating MixUp statistics plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate many MixUp samples to show lambda distribution
    num_iterations = 1000
    lambda_values = []
    
    for _ in range(num_iterations):
        test_imgs = torch.randn(2, 3, 224, 224)
        test_lbls = torch.tensor([0, 1])
        mixed_imgs, mixed_lbls = model.mixup_fn(test_imgs, test_lbls)
        lambda_val = mixed_lbls[0, 0].item()
        lambda_values.append(lambda_val)
    
    # Plot 1: Lambda distribution
    axes[0].hist(lambda_values, bins=50, color='#3A86FF', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Lambda (λ)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'MixUp Lambda Distribution\n(Beta({mixup_kwargs["mixup_alpha"]}, {mixup_kwargs["mixup_alpha"]}) over {num_iterations} samples)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axvline(0.5, color='red', linestyle='--', label='λ=0.5 (equal mix)')
    axes[0].legend()
    
    # Plot 2: Statistics
    lambda_array = np.array(lambda_values)
    stats_text = f"""
    MixUp Statistics:
    
    Alpha Parameter: {mixup_kwargs['mixup_alpha']}
    
    Lambda Statistics:
    • Mean: {lambda_array.mean():.3f}
    • Std Dev: {lambda_array.std():.3f}
    • Min: {lambda_array.min():.3f}
    • Max: {lambda_array.max():.3f}
    • Median: {np.median(lambda_array):.3f}
    
    Configuration:
    • Probability: {mixup_kwargs['prob']*100:.0f}%
    • Mode: {mixup_kwargs['mode']}
    • Label Smoothing: {mixup_kwargs['label_smoothing']}
    """
    
    axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "mixup_statistics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved MixUp statistics: {output_path}")
    plt.close()
    
    print(f"\n✅ MixUp visualization complete!")
    print(f"   Lambda mean: {lambda_array.mean():.3f} (should be ~0.5)")
    print(f"   Lambda std: {lambda_array.std():.3f}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n🚀 Generating visual verifications...\n")
    
    try:
        visualize_augmentations()
    except Exception as e:
        print(f"❌ Error in augmentation visualization: {e}")
    
    try:
        visualize_onecycle()
    except Exception as e:
        print(f"❌ Error in OneCycle visualization: {e}")
    
    try:
        visualize_resolution_schedule()
    except Exception as e:
        print(f"❌ Error in resolution schedule visualization: {e}")
    
    try:
        visualize_blurpool()
    except Exception as e:
        print(f"❌ Error in BlurPool visualization: {e}")
    
    try:
        visualize_mixup()
    except Exception as e:
        print(f"❌ Error in MixUp visualization: {e}")
    
    print("\n" + "="*80)
    print("✅ VISUAL VERIFICATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Check the '{output_dir}' directory for generated plots:")
    print(f"   - augmentations.png: See train vs FixRes augmentations")
    print(f"   - onecycle_schedule.png: LR and momentum over training")
    print(f"   - resolution_schedule.png: Progressive resizing timeline")
    print(f"   - blurpool_verification.png: BlurPool integration status")
    print(f"   - mixup_visualization.png: See how MixUp mixes images")
    print(f"   - mixup_labels.png: See soft label distributions")
    print(f"   - mixup_statistics.png: Lambda distribution and statistics")
    print("\n💡 Review these plots to ensure everything is configured correctly!")
    print("="*80)

if __name__ == "__main__":
    main()


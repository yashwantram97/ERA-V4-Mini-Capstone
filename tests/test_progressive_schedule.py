"""
Test script to visualize the progressive resizing schedule with FixRes support

Usage:
    python test_progressive_schedule.py

This script shows how different progressive resizing schedules look
for various epoch counts and hyperparameter settings.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.callbacks import create_progressive_resize_schedule


def print_schedule(schedule, title):
    """Pretty print a resolution schedule with FixRes support"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print('='*80)
    print(f"{'Epoch':<10} {'Resolution':<15} {'Transform Mode':<30}")
    print('-'*80)
    
    sorted_epochs = sorted(schedule.keys())
    for i, epoch in enumerate(sorted_epochs):
        size, transform_mode = schedule[epoch]
        
        mode_display = {
            "train": "Train (Full augmentation)",
            "valid": "Validation (Resize + CenterCrop)",
            "fixres": "FixRes (Minimal augmentation)"
        }.get(transform_mode, transform_mode)
        
        # Calculate range
        if i < len(sorted_epochs) - 1:
            end_epoch = sorted_epochs[i+1] - 1
            epoch_range = f"{epoch}-{end_epoch}"
        else:
            epoch_range = f"{epoch}+"
        
        emoji = "⚡" if transform_mode == "fixres" else "📊"
        print(f"{emoji} {epoch_range:<8} {size}x{size} px{'':<8} {mode_display:<30}")
    
    print('='*80)


# def print_schedule(schedule, title):
#     """Pretty print a resolution schedule"""
#     print(f"\n{'='*80}")
#     print(f"{title:^80}")
#     print('='*80)
#     print(f"{'Epoch':<10} {'Resolution':<15} {'Augmentation':<30}")
#     print('-'*80)
    
#     sorted_epochs = sorted(schedule.keys())
#     for i, epoch in enumerate(sorted_epochs):
#         size, use_train_augs = schedule[epoch]
#         aug_type = "Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)" if use_train_augs else "Test (Resize + CenterCrop) - FixRes"
        
#         # Calculate range
#         if i < len(sorted_epochs) - 1:
#             end_epoch = sorted_epochs[i+1] - 1
#             epoch_range = f"{epoch}-{end_epoch}"
#         else:
#             epoch_range = f"{epoch}+"
        
#         print(f"{epoch_range:<10} {size}x{size} px{'':<8} {aug_type:<30}")
    
#     print('='*80)


def main():
    """Generate and display schedules for all configurations"""
    
    # Test 1: Standard training with FixRes (90 epochs - g5 config)
    print("\n🎯 Test 1: Standard Training + FixRes (90 epochs)")
    g5_schedule = create_progressive_resize_schedule(
        total_epochs=120,
        target_size=224,          # Standard ImageNet resolution
        initial_scale=1.0,        # Start at full resolution (224px)
        delay_fraction=0.0,       # No delay, start at target size
        finetune_fraction=1.0,    # Train at 224px for most of training
        size_increment=4,         # Round to multiples of 4
        use_fixres=True,          # Enable FixRes for +1-2% accuracy boost
        fixres_size=256,          # Higher resolution for FixRes phase
        fixres_epochs=20           # Last 9 epochs (10%) for FixRes
    )
    print_schedule(g5_schedule, "90 Epochs - Standard Training + FixRes (g5 config)")
    
    
    print("\n✅ All schedules generated successfully!")
    print("\n💡 FixRes Key Benefits:")
    print("   • Addresses train-test distribution mismatch")
    print("   • Train: RandomResizedCrop (random crops) vs Test: CenterCrop (center only)")
    print("   • Fine-tune at higher resolution with minimal augmentation")
    print("   • Expected improvement: +1-2% validation accuracy")
    print("\n📊 Transform Modes:")
    print("   • 'train':  Full augmentation (RandomResizedCrop + ColorJitter + RandomErasing)")
    print("   • 'fixres': Minimal augmentation (Resize + RandomCrop + Flip only)")
    print("   • 'valid':  Test preprocessing (Resize + CenterCrop)")
    print()


if __name__ == "__main__":
    main()


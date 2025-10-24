"""
Test script to visualize the progressive resizing schedule

Usage:
    python test_progressive_schedule.py

This script shows how different progressive resizing schedules look
for various epoch counts and hyperparameter settings.
"""


def create_progressive_resize_schedule(
    total_epochs: int,
    target_size: int = 224,
    initial_scale: float = 0.5,
    delay_fraction: float = 0.5,
    finetune_fraction: float = 0.2,
    size_increment: int = 4,
    use_fixres: bool = False,
    fixres_size: int = 256
):
    """
    Create a progressive resizing schedule following MosaicML Composer's approach.
    (Standalone version for testing - mirrors the implementation in callbacks)
    """
    # Calculate epoch boundaries
    delay_epochs = int(total_epochs * delay_fraction)
    finetune_start_epoch = total_epochs - int(total_epochs * finetune_fraction)
    
    # Calculate initial size (rounded to size_increment)
    initial_size = int(target_size * initial_scale)
    initial_size = (initial_size // size_increment) * size_increment
    
    # Number of epochs for progressive phase
    progressive_epochs = finetune_start_epoch - delay_epochs
    
    schedule = {}
    
    # Phase 1: Delay phase - stay at initial scale
    if delay_epochs > 0:
        schedule[0] = (initial_size, True)
    
    # Phase 2: Progressive phase - linearly increase resolution
    if progressive_epochs > 0:
        for epoch in range(delay_epochs, finetune_start_epoch):
            # Linear interpolation from initial_size to target_size
            progress = (epoch - delay_epochs) / progressive_epochs
            current_size = initial_size + int(progress * (target_size - initial_size))
            
            # Round to nearest multiple of size_increment
            current_size = round(current_size / size_increment) * size_increment
            current_size = min(current_size, target_size)  # Cap at target
            
            # Only add to schedule if size changes
            if epoch == delay_epochs or current_size != schedule[list(schedule.keys())[-1]][0]:
                schedule[epoch] = (current_size, True)
    
    # Phase 3: Fine-tune phase - full resolution
    if finetune_start_epoch < total_epochs:
        schedule[finetune_start_epoch] = (target_size, True)
    
    # Optional Phase 4: FixRes phase - even higher resolution with test augmentations
    if use_fixres:
        fixres_start = total_epochs - max(1, int(total_epochs * 0.1))  # Last 10% for FixRes
        schedule[fixres_start] = (fixres_size, False)  # False = use test augmentations
    
    return schedule


def print_schedule(schedule, title):
    """Pretty print a resolution schedule"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print('='*80)
    print(f"{'Epoch':<10} {'Resolution':<15} {'Augmentation':<30}")
    print('-'*80)
    
    sorted_epochs = sorted(schedule.keys())
    for i, epoch in enumerate(sorted_epochs):
        size, use_train_augs = schedule[epoch]
        aug_type = "Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)" if use_train_augs else "Test (Resize + CenterCrop) - FixRes"
        
        # Calculate range
        if i < len(sorted_epochs) - 1:
            end_epoch = sorted_epochs[i+1] - 1
            epoch_range = f"{epoch}-{end_epoch}"
        else:
            epoch_range = f"{epoch}+"
        
        print(f"{epoch_range:<10} {size}x{size} px{'':<8} {aug_type:<30}")
    
    print('='*80)


def main():
    """Generate and display schedules for all configurations"""
    
    # Test 1: MosaicML Composer recommended settings (60 epochs)
    print("\n🎯 MosaicML Composer Recommended Settings")
    composer_schedule = create_progressive_resize_schedule(
    total_epochs=60,
    target_size=224,          # Standard ImageNet resolution
    initial_scale=0.64,       # Start at 64% (144px) - IMPROVED from 0.5
    delay_fraction=0.3,       # First 30% at initial scale - IMPROVED from 0.5
    finetune_fraction=0.3,    # Last 30% at full size - IMPROVED from 0.2
    size_increment=4,         # Round to multiples of 4
    use_fixres=False,         # Disable FixRes for local dev (faster)
    fixres_size=256   
    )
    print_schedule(composer_schedule, "60 Epochs - MosaicML Composer Approach")
    
    # Show detailed progression
    print("\n📊 Detailed Epoch-by-Epoch Breakdown:")
    print("-" * 80)
    phase_1 = 30  # 50% of 60
    phase_2 = 48  # 80% of 60
    total_epochs = 60
    
    print(f"Phase 1 (Delay):       Epochs 0-{phase_1-1} ({phase_1} epochs, {phase_1/total_epochs*100:.0f}%)")
    print(f"                      → Stay at 112px (50% of 224px)")
    print(f"                      → Fast training, learn basic features")
    print()
    print(f"Phase 2 (Progressive): Epochs {phase_1}-{phase_2-1} ({phase_2-phase_1} epochs, {(phase_2-phase_1)/total_epochs*100:.0f}%)")
    print(f"                      → Linear progression from 112px to 224px")
    print(f"                      → Curriculum learning: gradually increase complexity")
    print()
    print(f"Phase 3 (Fine-tune):   Epochs {phase_2}-{total_epochs-1} ({total_epochs-phase_2} epochs, {(total_epochs-phase_2)/total_epochs*100:.0f}%)")
    print(f"                      → Full 224px resolution")
    print(f"                      → Fine-tune at target resolution")
    
    # Test 2: With FixRes enabled
    print("\n\n🎯 With FixRes Enabled (+1-2% Accuracy Boost)")
    fixres_schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,          # Standard ImageNet resolution
        initial_scale=0.64,       # Start at 64% (144px) - IMPROVED from 0.5
        delay_fraction=0.3,       # First 30% at initial scale - IMPROVED from 0.5
        finetune_fraction=0.3,    # Last 30% at full size - IMPROVED from 0.2
        size_increment=4,         # Round to multiples of 4
        use_fixres=True,          # Enable FixRes for demonstration
        fixres_size=256           # Higher resolution for FixRes phase
    )
    print_schedule(fixres_schedule, "60 Epochs - With FixRes Phase")
    
    # Test 3: Different settings (100 epochs)
    print("\n\n🎯 Alternative: 100 Epochs Training")
    long_schedule = create_progressive_resize_schedule(
        total_epochs=100,
        target_size=224,          # Standard ImageNet resolution
        initial_scale=0.64,       # Start at 64% (144px) - IMPROVED from 0.5
        delay_fraction=0.3,       # First 30% at initial scale - IMPROVED from 0.5
        finetune_fraction=0.3,    # Last 30% at full size - IMPROVED from 0.2
        size_increment=4,         # Round to multiples of 4
        use_fixres=False,         # Disable FixRes for local dev (faster)
        fixres_size=256   
    )
    print_schedule(long_schedule, "100 Epochs - Extended Training")
    
    # Test 4: More aggressive (faster ramp-up)
    print("\n\n🎯 Aggressive: Faster Ramp-Up")
    aggressive_schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,          # Standard ImageNet resolution
        initial_scale=0.64,       # Start at 64% (144px) - IMPROVED from 0.5
        delay_fraction=0.3,       # First 30% at initial scale - IMPROVED from 0.5
        finetune_fraction=0.3,    # Last 30% at full size - IMPROVED from 0.2
        size_increment=4,         # Round to multiples of 4
        use_fixres=False,         # Disable FixRes for local dev (faster)
        fixres_size=256   
    )
    print_schedule(aggressive_schedule, "60 Epochs - Faster Ramp-Up (30% delay, 30% finetune)")
    
    print("\n✅ All schedules generated successfully!")
    print("\n💡 Key Benefits:")
    print("   • Early epochs train ~4x faster (112² vs 224² pixels)")
    print("   • Curriculum learning: easy → hard (small → large images)")
    print("   • Proven to improve speed/accuracy tradeoff")
    print("   • Optional FixRes phase aligns train/test distributions")
    print()


if __name__ == "__main__":
    main()


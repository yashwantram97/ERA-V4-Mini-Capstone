"""
Resolution Schedule Callback for PyTorch Lightning

Implements Progressive Resizing and FixRes techniques:
- Progressive Resizing: Start with small images, gradually increase resolution
- FixRes: Fine-tune at higher resolution with test-time augmentations

Benefits:
- Faster training in early epochs (smaller images)
- Better accuracy in later epochs (larger images)
- FixRes phase aligns train/test distributions (+1-2% accuracy)
- DDP-safe with rank_zero_only decorators for print statements
"""

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


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
    
    Args:
        total_epochs: Total number of training epochs
        target_size: Target/full resolution (e.g., 224 for ImageNet)
        initial_scale: Starting scale factor (0.5 = 50% of target_size)
        delay_fraction: Fraction of training to stay at initial_scale (e.g., 0.5 = first 50%)
        finetune_fraction: Fraction of training at full size (e.g., 0.2 = last 20%)
        size_increment: Round sizes to nearest multiple (e.g., 4 for alignment)
        use_fixres: Whether to add a FixRes phase at the end
        fixres_size: Resolution for FixRes phase (typically > target_size)
    
    Returns:
        Dictionary mapping epoch to (resolution, use_train_augs)
    
    Example:
        For 60 epochs, target_size=224, initial_scale=0.5, delay_fraction=0.5, finetune_fraction=0.2:
        - Epochs 0-29 (50%): 112px (initial scale)
        - Epochs 30-47 (30%): Progressive 112px → 224px
        - Epochs 48-59 (20%): 224px (fine-tune)
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

class ResolutionScheduleCallback(Callback):
    """
    Dynamically adjust image resolution and augmentation strategy during training.
    
    Args:
        schedule: Dictionary mapping epoch to (resolution, use_train_augs)
                  Example: {
                      0: (128, True),    # Epoch 0-9: 128px, train augs
                      10: (224, True),   # Epoch 10-84: 224px, train augs
                      85: (288, False)   # Epoch 85-90: 288px, test augs (FixRes)
                  }
    """
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self._last_applied_epoch = -1
        self._resume_handled = False

    def _get_resolution_for_epoch(self, epoch: int):
        """
        Get the correct resolution for a given epoch.
        Handles cases where epoch isn't explicitly in schedule.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (resolution, use_train_augs) or None if before first schedule entry
        """
        # Find the most recent schedule entry at or before this epoch
        applicable_epochs = [e for e in sorted(self.schedule.keys()) if e <= epoch]
        
        if not applicable_epochs:
            return None
            
        latest_epoch = max(applicable_epochs)
        return self.schedule[latest_epoch]

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the start of each training epoch"""
        current_epoch = trainer.current_epoch
        
        # Handle checkpoint resume: ensure correct resolution on first resumed epoch
        if not self._resume_handled and current_epoch > 0:
            # This is a resume (not starting from epoch 0)
            correct_config = self._get_resolution_for_epoch(current_epoch)
            
            if correct_config is not None:
                size, use_train_augs = correct_config
                
                if trainer.is_global_zero:
                    msg = f"\n{'='*60}\n"
                    msg += f"🔄 CHECKPOINT RESUME - Restoring Resolution State\n"
                    msg += f"   Resumed at epoch: {current_epoch}\n"
                    msg += f"   Expected resolution: {size}x{size}px\n"
                    msg += f"   Augmentation: {'Train' if use_train_augs else 'Test (FixRes)'}\n"
                    msg += f"{'='*60}"
                    print(msg)
                
                # Force update to correct resolution
                if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                    trainer.datamodule.update_resolution(size, use_train_augs)
                    
                    if trainer.is_global_zero:
                        self._verify_dataloader_changes(trainer, size)
                    
                    if hasattr(trainer.strategy, 'barrier'):
                        trainer.strategy.barrier()
            
            self._resume_handled = True
        
        # Check if we need to apply a schedule change
        if current_epoch in self.schedule:
            config = self.schedule[current_epoch]
            size, use_train_augs = config

            # Log the change (only on rank 0 to avoid console spam in DDP)
            if trainer.is_global_zero:
                msg = f"\n{'='*60}\n"
                msg += f"📐 Resolution Schedule - Epoch {current_epoch}\n"
                msg += f"   Resolution: {size}x{size}px\n"
                msg += f"   Augmentation: {'Train (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)' if use_train_augs else 'Test (Resize + CenterCrop) - FixRes'}\n"
                msg += f"{'='*60}"
                print(msg)

            # Update the datamodule's parameters
            if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                trainer.datamodule.update_resolution(size, use_train_augs)

                # ✅ Verify the changes actually took effect (only on rank 0)
                if trainer.is_global_zero:
                    self._verify_dataloader_changes(trainer, size)
                
                # Synchronize all processes after dataloader changes (DDP-safe)
                if hasattr(trainer.strategy, 'barrier'):
                    trainer.strategy.barrier()
                
                self._last_applied_epoch = current_epoch
            else:
                raise RuntimeError("No datamodule found in trainer. Make sure you're using ImageNetDataModule.")

    def _verify_dataloader_changes(self, trainer: L.Trainer, expected_resolution: int):
        """
        Verify that dataloader changes actually took effect by inspecting the datamodule.
        
        Args:
            trainer: Lightning Trainer
            expected_resolution: Expected image resolution
        """
        try:
            print("\n🔍 VERIFICATION - Checking dataloader configuration:")
            print("-" * 60)
            
            if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                dm = trainer.datamodule
                
                # Check DataModule settings
                print(f"   DataModule Resolution: {dm.resolution}")
                print(f"   DataModule Use Train Augs: {dm.use_train_augs}")
                
                # Verify against expected
                if dm.resolution == expected_resolution:
                    print(f"   ✅ Resolution matches: {expected_resolution}x{expected_resolution}")
                else:
                    print(f"   ❌ Resolution MISMATCH: expected {expected_resolution}, got {dm.resolution}")
                
                # ✅ Check transforms WITHOUT consuming batches
                print("\n   📝 Active Transforms (Train Dataset):")
                if hasattr(dm, 'train_dataset') and dm.train_dataset is not None:
                    if hasattr(dm.train_dataset, 'transform'):
                        transform = dm.train_dataset.transform
                        self._print_transforms(transform, indent="      ")
                else:
                    print("      ⚠️  Train dataset not yet created")
                
                print("-" * 60)
            else:
                print("⚠️  Cannot verify: datamodule not found")
            
        except Exception as e:
            print(f"⚠️  Verification failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_transforms(self, transform, indent=""):
        """
        Recursively print transform names and their key parameters.
        
        Args:
            transform: Transform object (Compose, list, or individual transform)
            indent: Indentation string for nested transforms
        """
        import torchvision.transforms as T
        
        try:
            # Handle torchvision Compose
            if isinstance(transform, T.Compose):
                for i, t in enumerate(transform.transforms, 1):
                    self._print_single_transform(t, i, indent)
            
            # Handle list of transforms
            elif isinstance(transform, list):
                for i, t in enumerate(transform, 1):
                    if isinstance(t, T.Compose):
                        self._print_transforms(t, indent)
                    else:
                        self._print_single_transform(t, i, indent)
            
            # Single transform
            else:
                self._print_single_transform(transform, 1, indent)
                
        except Exception as e:
            print(f"{indent}⚠️  Error listing transforms: {e}")
    
    def _print_single_transform(self, transform, index, indent=""):
        """
        Print a single transform with its key parameters.
        
        Args:
            transform: Single transform object
            index: Transform index number
            indent: Indentation string
        """
        transform_name = transform.__class__.__name__
        
        # Extract key parameters for common transforms
        params = []
        
        # For resize/crop operations
        if hasattr(transform, 'height') and hasattr(transform, 'width'):
            params.append(f"size={transform.height}x{transform.width}")
        elif hasattr(transform, 'size'):
            if isinstance(transform.size, (list, tuple)):
                params.append(f"size={transform.size[0]}x{transform.size[1]}")
            else:
                params.append(f"size={transform.size}")
        
        # For probability
        if hasattr(transform, 'p') and transform.p < 1.0:
            params.append(f"p={transform.p}")
        
        # For scale/ratio (RandomResizedCrop)
        if hasattr(transform, 'scale'):
            params.append(f"scale={transform.scale}")
        if hasattr(transform, 'ratio'):
            params.append(f"ratio={transform.ratio}")
        
        # For normalization
        if hasattr(transform, 'mean') and hasattr(transform, 'std'):
            if transform.mean is not None:
                params.append(f"mean={transform.mean[:3] if len(transform.mean) > 3 else transform.mean}")
        
        # Format output
        param_str = f" ({', '.join(params)})" if params else ""
        print(f"{indent}{index}. {transform_name}{param_str}")

    @rank_zero_only
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the start of training - log the schedule (only on rank 0 to avoid duplication)"""
        print("\n" + "="*60)
        print("📐 Resolution Schedule Configuration")
        print("="*60)
        for epoch, config in sorted(self.schedule.items()):
            size, use_train_augs = config
            aug_type = "Train" if use_train_augs else "Test (FixRes)"
            print(f"   Epoch {epoch:2d}+: {size}x{size}px, {aug_type} augmentations")
        print("="*60 + "\n")
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

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the start of each training epoch"""
        current_epoch = trainer.current_epoch
        
        # Check if we need to apply a schedule change
        if current_epoch in self.schedule:
            config = self.schedule[current_epoch]
            size, use_train_augs = config

            # Log the change (only on rank 0 to avoid console spam in DDP)
            if trainer.is_global_zero:
                msg = f"\n{'='*60}\n"
                msg += f"📐 Resolution Schedule - Epoch {current_epoch}\n"
                msg += f"   Resolution: {size}x{size}px\n"
                msg += f"   Augmentation: {'Train (RandomResizedCrop + Flip)' if use_train_augs else 'Test (Resize + CenterCrop) - FixRes'}\n"
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
        import albumentations as A
        
        try:
            # Handle Albumentations Compose
            if isinstance(transform, A.Compose):
                for i, t in enumerate(transform.transforms, 1):
                    self._print_single_transform(t, i, indent)
            
            # Handle list of transforms
            elif isinstance(transform, list):
                for i, t in enumerate(transform, 1):
                    if isinstance(t, A.Compose):
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
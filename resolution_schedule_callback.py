"""
Resolution Schedule Callback for PyTorch Lightning

Implements Progressive Resizing and FixRes techniques:
- Progressive Resizing: Start with small images, gradually increase resolution
- FixRes: Fine-tune at higher resolution with test-time augmentations

Benefits:
- Faster training in early epochs (smaller images)
- Better accuracy in later epochs (larger images)
- FixRes phase aligns train/test distributions (+1-2% accuracy)
"""

import lightning as L
from lightning.pytorch.callbacks import Callback

class ResolutionScheduleCallback(Callback):
    """
    Dynamically adjust image resolution, augmentation strategy, and batch size during training.
    
    Args:
        schedule: Dictionary mapping epoch to (resolution, use_train_augs, batch_size)
                  Example: {
                      0: (128, True, 512),    # Epoch 0-9: 128px, train augs, BS=512
                      10: (224, True, 320),   # Epoch 10-84: 224px, train augs, BS=320
                      85: (288, False, 256)   # Epoch 85-90: 288px, test augs (FixRes), BS=256
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

            # Handle both old format (size, use_train_augs) and new format (size, use_train_augs, batch_size)
            if len(config) == 2:
                size, use_train_augs = config
                batch_size = None
            else:
                size, use_train_augs, batch_size = config

        # Log the change
        msg = f"\n{'='*60}\n"
        msg += f"📐 Resolution Schedule - Epoch {current_epoch}\n"
        msg += f"   Resolution: {size}x{size}px\n"
        msg += f"   Augmentation: {'Train (RandomResizedCrop + Flip)' if use_train_augs else 'Test (Resize + CenterCrop) - FixRes'}\n"
        if batch_size:
            msg += f"   Batch Size: {batch_size}\n"
        msg += f"{'='*60}"
        print(msg)

        # Update the datamodule's parameters
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            trainer.datamodule.update_resolution(size, use_train_augs, batch_size)
            
            # Force recreation of dataloaders with new transforms
            # This is necessary for Lightning to pick up the changes
            self._reset_dataloaders(trainer)
            
            self._last_applied_epoch = current_epoch
        else:
            raise RuntimeError("No datamodule found in trainer. Make sure you're using ImageNetDataModule.")
    
    def _reset_dataloaders(self, trainer: L.Trainer):
        """
        Force Lightning to recreate dataloaders with updated settings.
        
        This is necessary because Lightning caches dataloaders for performance.
        When we change resolution/augmentations, we need fresh dataloaders.
        """
        try:
            # Reset train dataloader
            if hasattr(trainer, 'fit_loop') and hasattr(trainer.fit_loop, '_data_source'):
                trainer.fit_loop._data_source.instance = None
                trainer.fit_loop.setup_data()
            
            # Reset validation dataloader
            if hasattr(trainer, '_evaluation_loop') and hasattr(trainer._evaluation_loop, '_data_source'):
                trainer._evaluation_loop._data_source.instance = None
            
            print("✅ Dataloaders reset successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not reset dataloaders automatically: {e}")
            print("   Dataloaders will update on next epoch")

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the start of training - log the schedule"""
        print("\n" + "="*60)
        print("📐 Resolution Schedule Configuration")
        print("="*60)
        for epoch, config in sorted(self.schedule.items()):
            if len(config) == 2:
                size, use_train_augs = config
                aug_type = "Train" if use_train_augs else "Test (FixRes)"
                print(f"   Epoch {epoch:2d}+: {size}x{size}px, {aug_type} augmentations")
            else:
                size, use_train_augs, batch_size = config
                aug_type = "Train" if use_train_augs else "Test (FixRes)"
                print(f"   Epoch {epoch:2d}+: {size}x{size}px, {aug_type} augmentations, BS={batch_size}")
        print("="*60 + "\n")
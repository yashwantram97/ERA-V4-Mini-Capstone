import lightning as L
from ffcv.loader import Loader, OrderOption
from src.utils.utils import get_transforms
from lightning.pytorch.utilities import rank_zero_only

class ImageNetDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Imagenet1K dataset with FFCV
    
    This handles all data operations using FFCV for fast data loading:
    - Setup FFCV loaders
    - Create train/val dataloaders with FFCV pipelines
    - Handle data transformations via FFCV pipelines
    
    Note: Expects .beton files created with write_data.py
    
    Reference: https://lightning.ai/docs/pytorch/stable/data/alternatives.html#ffcv
    """
    def __init__(
        self,
        train_beton_path: str,
        val_beton_path: str,
        mean: tuple,
        std: tuple,
        batch_size: int = 64,
        num_workers: int = 4,
        initial_resolution: int = 224,  # Starting resolution
        use_train_augs: bool = True,    # Whether to use train augmentations
        os_cache: bool = True,          # Let OS manage caching
        quasi_random: bool = False,     # Use quasi-random for large datasets
        drop_last: bool = True,         # Drop last incomplete batch
    ):
        """
        Initialize the DataModule for FFCV
        
        Args:
            train_beton_path: Path to training .beton file
            val_beton_path: Path to validation .beton file
            mean: Normalization mean values (tuple of 3 floats)
            std: Normalization std values (tuple of 3 floats)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading (per GPU in DDP mode)
            initial_resolution: Starting image resolution (default 224)
            use_train_augs: Whether to use training augmentations (default True)
            os_cache: If True, OS manages caching. If False, FFCV manages it (default True)
            quasi_random: Use QUASI_RANDOM ordering for memory efficiency (default False)
            drop_last: Drop last incomplete batch (default True, needed for MixUp)
        """
        super().__init__()

        # Store hyperparameters - Lightning will log these automatically
        self.save_hyperparameters()

        # Store dataset paths and normalization constants
        self.train_beton_path = train_beton_path
        self.val_beton_path = val_beton_path
        self.mean = mean
        self.std = std

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.os_cache = os_cache
        self.quasi_random = quasi_random
        self.drop_last = drop_last

        # Store resolution and augmentation settings
        self.resolution = initial_resolution
        self.use_train_augs = use_train_augs
        
        # Will be set in setup()
        self.train_loader = None
        self.val_loader = None

    def update_resolution(self, resolution: int, use_train_augs: bool):
        """
        Update resolution and augmentation type.
        Called by ResolutionScheduleCallback during training.
        Note: This must run on all ranks to update loaders on each GPU.
        
        Args:
            resolution: New image resolution (e.g., 128, 224, 288)
            use_train_augs: True for train augmentations, False for test augmentations (FixRes)
        """
        self.resolution = resolution
        self.use_train_augs = use_train_augs
        
        # Recreate FFCV loaders with new pipelines (must happen on all ranks)
        self.setup(stage='fit')

    @rank_zero_only
    def prepare_data(self):
        """
        Called once to prepare data (download, etc.)
        Use this for operations that should be done on only one GPU in distributed training
        """
        # Data should already be converted to .beton format using write_data.py
        print("📁 FFCV data already prepared (.beton files)")

    def setup(self, stage: str = None):
        """
        Called on every GPU in distributed training
        Setup FFCV loaders for train/val
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Generate FFCV pipelines dynamically based on current settings
            train_image_pipeline, train_label_pipeline = get_transforms(
                transform_type="train" if self.use_train_augs else "valid",
                mean=self.mean,
                std=self.std,
                resolution=self.resolution
            )
            
            valid_image_pipeline, valid_label_pipeline = get_transforms(
                transform_type="valid",
                mean=self.mean,
                std=self.std,
                resolution=self.resolution
            )
            
            # Determine ordering strategy
            train_order = OrderOption.QUASI_RANDOM if self.quasi_random else OrderOption.RANDOM
            
            # Create FFCV Loader for training
            # Note: FFCV handles distributed training automatically
            self.train_loader = Loader(
                self.train_beton_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=train_order,
                os_cache=self.os_cache,
                drop_last=self.drop_last,
                pipelines={
                    'image': train_image_pipeline,
                    'label': train_label_pipeline
                },
                distributed=True  # FFCV handles DDP automatically
            )
            
            # Create FFCV Loader for validation
            self.val_loader = Loader(
                self.val_beton_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=OrderOption.SEQUENTIAL,  # No shuffling for validation
                os_cache=self.os_cache,
                drop_last=self.drop_last,
                pipelines={
                    'image': valid_image_pipeline,
                    'label': valid_label_pipeline
                },
                distributed=True  # FFCV handles DDP automatically
            )
            
            # Print info (only on rank 0 to avoid spam in DDP)
            should_print = True
            if hasattr(self, 'trainer') and self.trainer is not None:
                should_print = self.trainer.is_global_zero
            
            if should_print:
                aug_type = "Train" if self.use_train_augs else "Test (FixRes)"
                print(f"📊 FFCV Loaders @ {self.resolution}x{self.resolution}px:")
                print(f"   Train: {self.train_beton_path}")
                print(f"   Val:   {self.val_beton_path}")
                print(f"   Batch size: {self.batch_size}")
                print(f"   Num workers: {self.num_workers}")
                print(f"   Train augmentation: {aug_type}")
                print(f"   Order: {'QUASI_RANDOM' if self.quasi_random else 'RANDOM'} (train), SEQUENTIAL (val)")

    def train_dataloader(self):
        """
        Return training dataloader (FFCV Loader)
        
        Note: FFCV Loader already handles distributed training, so Lightning's
        use_distributed_sampler should be disabled in Trainer if needed.
        """
        return self.train_loader

    def val_dataloader(self):
        """Return validation dataloader (FFCV Loader)"""
        return self.val_loader

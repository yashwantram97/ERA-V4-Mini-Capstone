import torchvision
import lightning as L
from torch.utils.data import DataLoader
from src.utils.utils import get_transforms
from src.data_modules.imagenet_dataset import ImageNetDataset

class ImageNetDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Imagenet1K dataset
    
    This handles all data operations:
    - Setup datasets
    - Create train/val dataloaders
    - Handle data transformations
    """
    def __init__(
        self,
        train_img_dir,
        val_img_dir,
        mean,
        std,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        initial_resolution: int = 224,  # Starting resolution
        use_train_augs: bool = True,    # Whether to use train augmentations
    ):
        """
        Initialize the DataModule
        
        Args:
            train_img_dir: Path to training images directory
            val_img_dir: Path to validation images directory
            mean: Normalization mean values
            std: Normalization std values
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            initial_resolution: Starting image resolution (default 224)
            use_train_augs: Whether to use training augmentations (default True)        
        """
        super().__init__()

        # Store hyperparameters - Lightning will log these automatically
        self.save_hyperparameters()

        # Store dataset paths and normalization constants
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        self.mean = mean
        self.std = std

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Store resolution and augmentation settings
        self.resolution = initial_resolution
        self.use_train_augs = use_train_augs
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None

    def update_resolution(self, resolution: int, use_train_augs: bool, batch_size: int = None):
        """
        Update resolution, augmentation type, and optionally batch size.
        Called by ResolutionScheduleCallback during training.
        
        Args:
            resolution: New image resolution (e.g., 128, 224, 288)
            use_train_augs: True for train augmentations, False for test augmentations (FixRes)
            batch_size: New batch size (optional, None keeps current batch size)
        """
        self.resolution = resolution
        self.use_train_augs = use_train_augs
        
        if batch_size is not None:
            self.batch_size = batch_size
        
        # Recreate datasets with new transforms
        self.setup(stage='fit')
        
        print(f"✅ DataModule updated: {resolution}x{resolution}px, "
              f"{'Train' if use_train_augs else 'Test'} augs, BS={self.batch_size}")

    def prepare_data(self):
        """
        Called once to prepare data (download, etc.)
        Use this for operations that should be done on only one GPU in distributed training
        """
        # In my case, data is already downloaded and prepared
        # This is where you'd put download logic if needed
        print("📁 Data already prepared")

    def setup(self, stage: str = None):
        """
        Called on every GPU in distributed training
        Setup datasets for train/val/test
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Generate transforms dynamically based on current settings
            train_transforms = get_transforms(
                transform_type="train" if self.use_train_augs else "valid",
                mean=self.mean,
                std=self.std,
                resolution=self.resolution
            )
            
            valid_transforms = get_transforms(
                transform_type="valid",
                mean=self.mean,
                std=self.std,
                resolution=self.resolution
            )
            
            # Create train dataset
            self.train_dataset = ImageNetDataset(
                root=self.train_img_dir,
                transform=train_transforms
            )

            self.val_dataset = ImageNetDataset(
                root=self.val_img_dir,
                transform=valid_transforms
            )

        # Print dataset splits - only print what exists
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            aug_type = "Train" if self.use_train_augs else "Test (FixRes)"
            print(f"📊 Dataset @ {self.resolution}x{self.resolution}px:")
            print(f"   Train: {len(self.train_dataset)} samples ({aug_type} augmentation)")
            print(f"   Val:   {len(self.val_dataset)} samples (Test augmentation)")

    def train_dataloader(self):
        """Return training dataloader"""
        # Lightning automatically sets up a DistributedSampler for you when you specify: Trainer(strategy="ddp", devices=4) 
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Lightning automatically replaces your shuffle=True with the correct sampler in DDP.
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

import torchvision
import lightning as L
from torch.utils.data import DataLoader
from config import train_img_dir, val_img_dir

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
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transforms = None,
        valid_transforms = None,
    ):
        """
        Initialize the DataModule
        """
        super().__init__()

        # Store hyperparameters - Lightning will log these automatically
        # EXCLUDE transforms from hyperparameters to avoid conflicts
        self.save_hyperparameters(ignore=['train_transforms', 'valid_transforms'])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Define transforms
        self.train_transforms = train_transforms
        
        self.valid_transforms = valid_transforms
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None

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
            # Create train dataset
            self.train_dataset = torchvision.datasets.ImageFolder(
                root=train_img_dir,
                transform=self.train_transforms
            )

            self.val_dataset = torchvision.datasets.ImageFolder(
                root=val_img_dir,
                transform=self.valid_transforms
            )

        # Print dataset splits - only print what exists
        print(f"📊 Dataset:")
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            print(f"   Train: {len(self.train_dataset)} samples (with augmentation)")
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            print(f"   Val:   {len(self.val_dataset)} samples (without augmentation)")

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

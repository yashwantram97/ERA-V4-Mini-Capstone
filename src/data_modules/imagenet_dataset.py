import torchvision
from torch.utils.data import Dataset
import numpy as np
import albumentations as A


class ImageNetDataset(Dataset):
    """
    Wrapper around ImageNet data folder to make it compatible with Albumentations.
    
    TorchVision's ImageFolder passes images as positional args: transform(image)
    Albumentations expects named args: transform(image=image)
    
    This wrapper converts between the two formats.
    """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root: Root directory path
            transform: Albumentations transform (expects list of transforms)
        """
        self.root = root
        self.transform = transform
        
        # Use ImageFolder to handle directory structure and labels
        self.dataset = torchvision.datasets.ImageFolder(root)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image and label from ImageFolder
        image, label = self.dataset[idx]
        
        # Convert PIL Image to numpy array (RGB format)
        image = np.array(image)
        
        # Apply Albumentations transforms
        if self.transform is not None:
            # get_transforms() always returns A.Compose object
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
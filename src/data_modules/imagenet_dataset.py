import torchvision
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """
    Wrapper around ImageNet data folder with torchvision transforms.
    
    This class uses torchvision.datasets.ImageFolder to handle directory structure
    and applies torchvision transforms directly to PIL images.
    """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root: Root directory path
            transform: torchvision.transforms.Compose object
        """
        self.root = root
        self.transform = transform
        
        # Use ImageFolder to handle directory structure and labels
        self.dataset = torchvision.datasets.ImageFolder(root)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image and label from ImageFolder (image is PIL Image)
        image, label = self.dataset[idx]
        
        # Apply torchvision transforms directly to PIL Image
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
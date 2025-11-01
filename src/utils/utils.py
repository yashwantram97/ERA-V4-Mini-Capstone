from pathlib import Path
import torchvision.transforms as T
from config import PROJECT_ROOT

def get_transforms(transform_type="train", mean=None, std=None, resolution=224):
    """
    Get transforms for training, validation, or FixRes fine-tuning.
    
    Transform Types:
    - "train": Full training augmentations with RandomResizedCrop
    - "valid": Standard validation transforms with Resize + CenterCrop
    - "fixres": FixRes fine-tuning with minimal augmentation at higher resolution
    
    FixRes (Fixed Resolution) Explained:
    The FixRes technique addresses train-test distribution mismatch:
    1. Training uses RandomResizedCrop which samples random crop locations/scales
    2. Testing uses CenterCrop which always crops from center
    3. This mismatch hurts accuracy by ~1-2%
    
    Solution: Fine-tune at higher resolution with minimal augmentation
    - Trains the model to handle the different crop distribution
    - Uses higher resolution (e.g., 256 or 288 vs 224) for better features
    - Minimal augmentation (only horizontal flip) to adapt to test distribution
    
    Args:
        transform_type: "train", "valid", or "fixres"
        mean: Normalization mean (list or tuple)
        std: Normalization std (list or tuple)
        resolution: Target image resolution (default 224)
    
    Returns:
        torchvision.transforms.Compose object with all transforms
    """
    if transform_type == "train":
        # Training transforms with balanced augmentation strategy
        # ColorJitter + RandomErasing + MixUp(0.2) - proven combo for 75%+ in 90 epochs
        transforms = T.Compose([
            T.RandomResizedCrop(resolution, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    elif transform_type == "fixres":
        transforms = T.Compose([
            T.Resize(int(resolution * 256 / 224)),  # Scale proportionally
            T.RandomCrop(resolution),  # Random crop (not center) to maintain some variation
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:  # "valid" or default
        transforms = T.Compose([
            T.Resize(int(resolution * 256 / 224)),  # Scale proportionally (256 for 224px)
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return transforms

def serialize_transforms(transform_compose):
    """
    Convert transforms to a serializable format
    
    Args:
        transform_compose: torchvision.transforms.Compose object
        
    Returns:
        list: List of transform dictionaries with parameters
    """
    if not isinstance(transform_compose, T.Compose):
        return str(transform_compose)

    serialized_transforms = []

    for transform in transform_compose.transforms:
        transform_info = {
            "name": transform.__class__.__name__,
            "module": transform.__class__.__module__
        }

        # Extract parameters for common transforms
        if hasattr(transform, '__dict__'):
            params = {}
            for key, value in transform.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    # Convert non-serializable types to strings
                    if isinstance(value, (int, float, str, bool, list, tuple)):
                        params[key] = value
                    elif value is None:
                        params[key] = None
                    else:
                        params[key] = str(value)
            transform_info["parameters"] = params

        serialized_transforms.append(transform_info)

    return serialized_transforms

def get_relative_path(path):
    """
    Convert absolute path to relative path from project root
    
    Args:
        path: Path object or string path
        
    Returns:
        str: Relative path from project root
    """
    try:
        path = Path(path)
        if path.is_absolute():
            # Try to get relative path from project root
            relative_path = path.relative_to(PROJECT_ROOT)
            return str(relative_path)
        else:
            # Already relative
            return str(path)
    except ValueError:
        # Path is not under project root, return just the filename
        return path.name if hasattr(path, 'name') else str(path)

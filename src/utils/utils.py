from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from config import PROJECT_ROOT

def get_transforms(transform_type="train", mean=None, std=None, resolution=224):
    """
    Get transforms for training or validation
    
    Args:
        transform_type: "train" or "valid"
        mean: Normalization mean
        std: Normalization std
        resolution: Target image resolution (default 224)
    
    Returns:
        List of Albumentations transforms
    """
    transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    if transform_type == "train":
        transforms = [
            A.RandomResizedCrop(
                size=[resolution, resolution],
                scale=[0.5, 1],
                ratio=[0.75, 1.3333333333333333],
                interpolation=cv2.INTER_LINEAR
            ),
            A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(
            #     shift_limit=(-0.0625, 0.0625),
            #     scale_limit=(-0.1, 0.1),
            #     rotate_limit=(-45, 45),
            #     interpolation=cv2.INTER_LINEAR,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     rotate_method="largest_box",
            #     p=0.5
            # ),
            # A.CoarseDropout(
            #     num_holes_range=(1, 1),
            #     hole_height_range=(16, 16),
            #     hole_width_range=(16, 16),
            #     fill=mean,
            #     p=0.5
            # )
        ]+ transforms

    else:
        # Validation/Test transforms - FixRes compatible
        transforms = [
            A.Resize(
                height=int(resolution * 256 / 224),  # Scale proportionally
                width=int(resolution * 256 / 224),
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            ),
            A.CenterCrop(
                height=resolution,
                width=resolution,
                p=1.0
            )
        ] + transforms

    return transforms

def serialize_transforms(transform_compose):
    """
    Convert transforms to a serializable format
    
    Args:
        transform_compose: albumentations.Compose object
        
    Returns:
        list: List of transform dictionaries with parameters
    """
    if not isinstance(transform_compose, A.Compose):
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

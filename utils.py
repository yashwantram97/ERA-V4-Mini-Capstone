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

def get_batch_size_from_resolution_schedule(resolution_schedule, epochs):
    """
    Extract batch sizes from resolution schedule dictionary.
    
    Args:
        resolution_schedule: Dict with format {epoch: (resolution, use_train_augs, batch_size)}
                             Example: {0: (128, True, 512), 10: (224, True, 320), 85: (288, False, 256)}
        epochs: Total number of epochs
    
    Returns:
        list: List of batch sizes for each epoch
        
    Example:
        >>> schedule = {0: (128, True, 512), 10: (224, True, 320), 85: (288, False, 256)}
        >>> get_batch_size_from_resolution_schedule(schedule, 90)
        [512, 512, ..., 320, 320, ..., 256, ...]  # 512 for epochs 0-9, 320 for 10-84, 256 for 85-89
    """
    if resolution_schedule is None:
        return None
    
    # Create a list to hold batch size for each epoch
    batch_sizes = []
    
    # Sort the schedule by epoch
    sorted_epochs = sorted(resolution_schedule.keys()) # [0, 10, 85]
    
    for epoch in range(epochs): # 0, 1, 2, ..., 89
        # Find which schedule entry applies to this epoch
        applicable_batch_size = None
        
        for schedule_epoch in sorted_epochs: # 0, 10, 85
            if epoch >= schedule_epoch: # 10 >= 0, 10 >= 10, 10 >= 85
                # This schedule applies
                config = resolution_schedule[schedule_epoch] # config = (128, True, 512)
                if len(config) >= 3: # len(config) = 3
                    applicable_batch_size = config[2]  # batch_size is 3rd element
            else:
                break
        
        batch_sizes.append(applicable_batch_size)
    
    return batch_sizes

def get_total_num_steps(dataset_size, batch_size, batch_size_schedule, epochs, dynamic_batch_size=False):
    """
    Calculate total number of steps for training.

    Args:
        dataset_size (int): Size of the dataset.
        batch_size (int): Default/fallback batch size if no schedule provided
        batch_size_schedule (list of int or None): List of batch sizes for each epoch, or None
        epochs (int): Total number of epochs to train for.
        dynamic_batch_size (bool): Whether to use dynamic batch size scheduling

    Returns:
        int: Total number of steps to be run based on the batch size schedule.
        
    Example:
        dataset_size = 1281167
        batch_size_schedule = [512]*10 + [320]*75 + [256]*5  # From resolution schedule
        epochs = 90
        get_total_num_steps(dataset_size, 128, batch_size_schedule, epochs, True)
    """
    if not dynamic_batch_size or batch_size_schedule is None:
        # Fixed batch size throughout training
        return epochs * ((dataset_size + batch_size - 1) // batch_size)
    
    steps = 0
    for epoch in range(epochs):
        # Get batch size for this epoch
        if epoch < len(batch_size_schedule):
            bs = batch_size_schedule[epoch]
        else:
            # Use last batch size if schedule is shorter than epochs
            bs = batch_size_schedule[-1] if batch_size_schedule else batch_size
        
        # Handle None values (use default batch_size)
        if bs is None:
            bs = batch_size
            
        # Add steps for this epoch (ceil division to include all samples)
        steps += (dataset_size + bs - 1) // bs
    
    return steps
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from config import PROJECT_ROOT

def get_transforms(transform_type="train", mean=None, std=None):
    """
        To get the transform
    """
    transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    if transform_type == "train":
        transforms = [
            A.RandomResizedCrop(
                size=[224, 224],
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
        transforms = [
            A.Resize(
                height=256,
                width=256,
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            ),
            A.CenterCrop(
                height=224,
                width=224,
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

def get_total_num_steps(dataset_size, batch_size, batch_size_schedule, epochs, dynamic_batch_size=False):
    """
    Calculate total number of steps for training.

    Args:
        dataset_size (int): Size of the dataset.
        batch_size_schedule (list of int): List of batch sizes for each phase/epoch/step.
        epochs (int): Total number of epochs to train for. (Required if schedule shorter than epochs)

    Returns:
        int: Total number of steps to be run based on the batch size schedule.
    Example:
        dataset_size = 1000
        batch_size_schedule = [256, 128, 64]  # e.g., 5 epochs at 256, 5 at 128, rest at 64
        epochs = 15
        get_total_num_steps(dataset_size, batch_size_schedule, epochs)
    """
    if not dynamic_batch_size:
        return epochs * (dataset_size // batch_size) # ceil division to get all samples
    
    steps = 0
    num_sched = len(batch_size_schedule)
    for epoch in range(epochs):
        # Use last batch size value if epochs > schedule
        if epoch < num_sched:
            bs = batch_size_schedule[epoch]
        else:
            bs = batch_size_schedule[-1]
        steps += (dataset_size + bs - 1) // bs  # ceil division to get all samples
    return steps
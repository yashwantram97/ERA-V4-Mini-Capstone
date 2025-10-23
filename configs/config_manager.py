"""
Configuration manager for easy hardware profile selection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import importlib
import sys
from pathlib import Path


@dataclass
class ConfigProfile:
    """Data class to hold configuration settings."""
    
    # Profile info
    profile_name: str
    profile_description: str
    
    # Paths
    project_root: Path
    train_img_dir: Path
    val_img_dir: Path
    logs_dir: Path
    
    # Dataset settings
    dataset_size: int
    num_classes: int
    input_size: tuple
    mean: tuple
    std: tuple
    
    # Training settings
    experiment_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    
    # Hardware settings
    num_workers: int
    precision: str
    
    # Scheduler settings
    scheduler_type: str
    
    # Fields with defaults (must come after non-default fields)
    num_devices: Optional[int] = None
    strategy: Optional[str] = None
    onecycle_kwargs: Optional[Dict[str, Any]] = None
    lr_finder_kwargs: Optional[Dict[str, Any]] = None
    mixup_kwargs: Optional[Dict[str, Any]] = None
    prog_resizing_fixres_schedule: Optional[Dict[int, tuple]] = None
    early_stopping_patience: int = 5
    save_top_k: int = 3
    save_last: bool = True
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 0.5
    accumulate_grad_batches: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'profile_name': self.profile_name,
            'profile_description': self.profile_description,
            'project_root': self.project_root,
            'train_img_dir': self.train_img_dir,
            'val_img_dir': self.val_img_dir,
            'logs_dir': self.logs_dir,
            'dataset_size': self.dataset_size,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'mean': self.mean,
            'std': self.std,
            'experiment_name': self.experiment_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_workers': self.num_workers,
            'precision': self.precision,
            'num_devices': self.num_devices,
            'strategy': self.strategy,
            'scheduler_type': self.scheduler_type,
            'onecycle_kwargs': self.onecycle_kwargs,
            'lr_finder_kwargs': self.lr_finder_kwargs,
            'mixup_kwargs': self.mixup_kwargs,
            'prog_resizing_fixres_schedule': self.prog_resizing_fixres_schedule,
            'early_stopping_patience': self.early_stopping_patience,
            'save_top_k': self.save_top_k,
            'save_last': self.save_last,
            'log_every_n_steps': self.log_every_n_steps,
            'check_val_every_n_epoch': self.check_val_every_n_epoch,
            'gradient_clip_val': self.gradient_clip_val,
            'accumulate_grad_batches': self.accumulate_grad_batches,
        }
    
    def __repr__(self) -> str:
        """Pretty print config."""
        return f"""
╔═══════════════════════════════════════════════════════════════════
║ Configuration Profile: {self.profile_name}
║ {self.profile_description}
╠═══════════════════════════════════════════════════════════════════
║ Training Settings:
║   • Epochs: {self.epochs}
║   • Batch Size: {self.batch_size}
║   • Learning Rate: {self.learning_rate}
║   • Weight Decay: {self.weight_decay}
║
║ Hardware Settings:
║   • Workers: {self.num_workers}
║   • Precision: {self.precision}
║   • Devices: {self.num_devices if self.num_devices else 'auto'}
║   • Strategy: {self.strategy if self.strategy else 'auto'}
║
║ Dataset:
║   • Size: {self.dataset_size:,} images
║   • Classes: {self.num_classes}
║   • Train Dir: {self.train_img_dir}
║
║ Progressive Resizing:
║   • Enabled: {bool(self.prog_resizing_fixres_schedule)}
║   • Stages: {len(self.prog_resizing_fixres_schedule) if self.prog_resizing_fixres_schedule else 0}
║
║ Logging:
║   • Experiment: {self.experiment_name}
║   • Log Dir: {self.logs_dir}
╚═══════════════════════════════════════════════════════════════════
"""


def _load_config_module(profile_name: str):
    """Load configuration module by profile name."""
    config_map = {
        'local': 'configs.local_config',
        'g5': 'configs.g5_config',
        'p3': 'configs.p3_config',
    }
    
    if profile_name not in config_map:
        available = ', '.join(config_map.keys())
        raise ValueError(
            f"Unknown profile: '{profile_name}'. "
            f"Available profiles: {available}"
        )
    
    module_name = config_map[profile_name]
    return importlib.import_module(module_name)


def get_config(profile_name: str = 'local') -> ConfigProfile:
    """
    Get configuration for specified hardware profile.
    
    Args:
        profile_name: One of 'local', 'g5', or 'p3'
        
    Returns:
        ConfigProfile object with all settings
        
    Examples:
        >>> config = get_config('local')
        >>> config = get_config('g5')
        >>> config = get_config('p3')
    """
    module = _load_config_module(profile_name)
    
    # Create ConfigProfile from module attributes
    config = ConfigProfile(
        profile_name=getattr(module, 'PROFILE_NAME'),
        profile_description=getattr(module, 'PROFILE_DESCRIPTION'),
        project_root=getattr(module, 'PROJECT_ROOT'),
        train_img_dir=getattr(module, 'TRAIN_IMG_DIR'),
        val_img_dir=getattr(module, 'VAL_IMG_DIR'),
        logs_dir=getattr(module, 'LOGS_DIR'),
        dataset_size=getattr(module, 'DATASET_SIZE'),
        num_classes=getattr(module, 'NUM_CLASSES'),
        input_size=getattr(module, 'INPUT_SIZE'),
        mean=getattr(module, 'MEAN'),
        std=getattr(module, 'STD'),
        experiment_name=getattr(module, 'EXPERIMENT_NAME'),
        epochs=getattr(module, 'EPOCHS'),
        batch_size=getattr(module, 'BATCH_SIZE'),
        learning_rate=getattr(module, 'LEARNING_RATE'),
        weight_decay=getattr(module, 'WEIGHT_DECAY'),
        accumulate_grad_batches=getattr(module, 'ACCUMULATE_GRAD_BATCHES', 1),
        num_workers=getattr(module, 'NUM_WORKERS'),
        precision=getattr(module, 'PRECISION'),
        num_devices=getattr(module, 'NUM_DEVICES', None),
        strategy=getattr(module, 'STRATEGY', None),
        scheduler_type=getattr(module, 'SCHEDULER_TYPE'),
        onecycle_kwargs=getattr(module, 'ONECYCLE_KWARGS'),
        lr_finder_kwargs=getattr(module, 'LR_FINDER_KWARGS'),
        mixup_kwargs=getattr(module, 'MIXUP_KWARGS', None),
        prog_resizing_fixres_schedule=getattr(module, 'PROG_RESIZING_FIXRES_SCHEDULE'),
        early_stopping_patience=getattr(module, 'EARLY_STOPPING_PATIENCE', 5),
        save_top_k=getattr(module, 'SAVE_TOP_K', 3),
        save_last=getattr(module, 'SAVE_LAST', True),
        log_every_n_steps=getattr(module, 'LOG_EVERY_N_STEPS', 50),
        check_val_every_n_epoch=getattr(module, 'CHECK_VAL_EVERY_N_EPOCH', 1),
        gradient_clip_val=getattr(module, 'GRADIENT_CLIP_VAL', 0.5),
    )
    
    return config


def list_configs() -> Dict[str, str]:
    """
    List all available configuration profiles.
    
    Returns:
        Dictionary mapping profile names to descriptions
    """
    profiles = {}
    
    for profile_name in ['local', 'g5', 'p3']:
        try:
            module = _load_config_module(profile_name)
            profiles[profile_name] = getattr(module, 'PROFILE_DESCRIPTION')
        except Exception as e:
            profiles[profile_name] = f"Error loading config: {e}"
    
    return profiles


def print_all_configs():
    """Print information about all available configs."""
    print("\n" + "=" * 70)
    print("Available Hardware Configuration Profiles")
    print("=" * 70)
    
    configs = list_configs()
    for name, description in configs.items():
        print(f"\n📋 Profile: {name}")
        print(f"   {description}")
    
    print("\n" + "=" * 70)
    print("Usage:")
    print("  python train.py --config local")
    print("  python train.py --config g5")
    print("  python train.py --config p3")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Demo: print all configs
    print_all_configs()
    
    # Demo: load and display each config
    for profile in ['local', 'g5', 'p3']:
        config = get_config(profile)
        print(config)


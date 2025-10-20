"""
Learning Rate Finder Script

Finds optimal learning rate for training using the LR range test.

Usage:
    python find_lr.py --config local
    python find_lr.py --config g5
    python find_lr.py --config p3
"""

import argparse
import torch
import torch.nn.functional as F
from src.utils.lr_finder_utils import run_lr_finder
from src.utils.utils import get_transforms
from src.data_modules.imagenet_datamodule import ImageNetDataModule
from src.models.resnet_module import ResnetLightningModule
from configs import get_config, list_configs


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal learning rate for your hardware configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='local',
        choices=['local', 'g5', 'p3'],
        help='Hardware configuration profile (default: local)'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=None,
        dest='learning_rate',
        help='Starting learning rate (overrides config value if provided)'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available configuration profiles and exit'
    )
    
    args = parser.parse_args()
    
    # List configs if requested
    if args.list_configs:
        print("\n" + "=" * 70)
        print("Available Hardware Configuration Profiles")
        print("=" * 70)
        configs = list_configs()
        for name, description in configs.items():
            print(f"\n📋 {name:10s} - {description}")
        print("\n" + "=" * 70)
        return
    
    # Load configuration
    print(f"\n🔧 Loading configuration: {args.config}")
    config = get_config(args.config)
    
    # Override learning rate if provided via command line
    if args.learning_rate is not None:
        print(f"⚡ Overriding learning rate: {config.learning_rate:.2e} → {args.learning_rate:.2e}")
        config.learning_rate = args.learning_rate
    
    print(config)
    
    # Setup transforms
    train_transforms = get_transforms(transform_type="train", mean=config.mean, std=config.std)

    # Setup data module
    imagenet_dm = ImageNetDataModule(
        train_img_dir=config.train_img_dir,
        val_img_dir=config.val_img_dir,
        mean=config.mean,
        std=config.std,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )


    imagenet_dm.setup(stage='fit')

    experiment_dir = config.logs_dir / config.experiment_name

    # Instantiate the Lightning module to get the torch model
    lit_module = ResnetLightningModule(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_classes=config.num_classes,
        train_transforms=train_transforms
    )
    model = lit_module.model

    train_loader = imagenet_dm.train_dataloader()
    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay, 
        momentum=0.9
    )

    print("\n🔍 Starting LR Finder...")
    print(f"   Range: {config.lr_finder_kwargs['start_lr']:.2e} to {config.lr_finder_kwargs['end_lr']:.2e}")
    print(f"   Iterations: {config.lr_finder_kwargs['num_iter']}")
    print(f"   Number of classes: {config.num_classes}")
    
    suggested_lr = run_lr_finder(
        model,
        train_loader,
        loss_fn,
        optimizer,
        **config.lr_finder_kwargs,
        save_path=experiment_dir / "lr_finder.png",
        logger=None
    )

    print(f"\n✅ Suggested LR: {suggested_lr:.2e}")
    print(f"📊 Plot saved to: {experiment_dir / 'lr_finder.png'}")
    print(f"\n💡 Update your config file with this learning rate for optimal training.")


if __name__ == "__main__":
    main()
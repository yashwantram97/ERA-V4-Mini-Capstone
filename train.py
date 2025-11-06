"""
PyTorch Lightning Training Script

This script demonstrates how to use Lightning for training:
- Simple, clean training code
- Automatic logging and checkpointing
- Easy experiment tracking
- Built-in best practices

Benefits over manual training loops:
- Less boilerplate code
- Automatic device handling
- Built-in validation
- Easy to scale and extend

Usage:
    python train.py --config local   # For MacBook M4 Pro
    python train.py --config g5     # For AWS g5.12xlarge
    python train.py --config p4      # For AWS p4d.24xlarge
"""

import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

# Import our Lightning components
from src.utils.utils import get_relative_path
from src.data_modules.imagenet_datamodule import ImageNetDataModule
from src.models.resnet_module import ResnetLightningModule
from src.callbacks.text_logging_callback import TextLoggingCallback
from src.callbacks.resolution_schedule_callback import ResolutionScheduleCallback
from lightning.pytorch.utilities import rank_zero_only

from time import time

# Import new config system
from configs import get_config, list_configs, ConfigProfile

def train_with_lightning(
    config: ConfigProfile,
    resume_from_checkpoint: str = None,
):
    """
    Train ImageNet dataset on ResNet50 using PyTorch Lightning
    
    Args:
        config: ConfigProfile object with all training settings
        resume_from_checkpoint: Path to checkpoint to resume training from where it stopped
    """
    print("🌩️ Starting PyTorch Lightning Training")
    print("=" * 60)
    print(config)

    # Determine initial resolution and transform mode from schedule or use defaults
    initial_resolution = 224
    transform_mode = "train"  # Default to training augmentations
    
    if config.prog_resizing_fixres_schedule:
        # Get the config for epoch 0 (or first epoch in schedule)
        first_epoch = min(config.prog_resizing_fixres_schedule.keys())
        schedule_config = config.prog_resizing_fixres_schedule[first_epoch]
        initial_resolution = schedule_config[0]
        use_train_augs = schedule_config[1]
        # Convert boolean to transform mode string
        transform_mode = "train" if use_train_augs else "fixres"
    
    # 1. Create DataModule
    # DataModule handles all data operations
    print("📊 Setting up data...")
    datamodule = ImageNetDataModule(
        train_img_dir=config.train_img_dir,
        val_img_dir=config.val_img_dir,
        mean=config.mean,
        std=config.std,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        initial_resolution=initial_resolution,
        transform_mode=transform_mode
    )

    # Calculate total_steps for OneCycleLR scheduler
    # Using fixed batch size throughout training
    # actual_total_steps = config.epochs * ((config.dataset_size + config.batch_size - 1) // config.batch_size)
    # print(f"📊 Calculated total_steps for OneCycleLR: {actual_total_steps:,}")
    # print(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}, Dataset Size: {config.dataset_size:,}")
    
    # 2. Create Lightning Module (Model)
    print("🧠 Setting up model...")
    model = ResnetLightningModule(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_classes=config.num_classes,
        # total_steps=actual_total_steps,
        mixup_kwargs=config.mixup_kwargs
    )
    
    # 3. Setup Callbacks
    # Callbacks add functionality without cluttering the main code
    print("⚙️ Setting up callbacks...")
    # Model Checkpointing - saves best models automatically
    log_dir = config.s3_dir if config.s3_dir else config.logs_dir
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + config.experiment_name + "-checkpoints-"+str(time()),
        filename="imagenet1k-{epoch:02d}-{val/accuracy:.3f}",
        monitor="val/accuracy",  # Metric to monitor
        mode="max",             # Save model with highest accuracy
        save_top_k=config.save_top_k,
        save_last=config.save_last,
        verbose=True
    )

    # Early Stopping - stops training if no improvement
    early_stop_callback = EarlyStopping(
        monitor="val/accuracy",  # Metric to monitor (changed from val/loss to val/accuracy)
        patience=config.early_stopping_patience,
        verbose=True,
        mode="max"              # Stop when val_accuracy stops increasing (changed from min)
    )

    # Rich Progress Bar - beautiful progress bars
    progress_bar = RichProgressBar()

    # Text Logging Callback - creates detailed text logs
    # Use S3 for logs if configured, otherwise use local logs directory
    text_logger = TextLoggingCallback(
        log_dir=log_dir,
        experiment_name=config.experiment_name
    )

    # Resolution Schedule Callback (Progressive Resizing + FixRes)
    callbacks_list = [
        checkpoint_callback,
        early_stop_callback,
        progress_bar,
        text_logger,
    ]

    # Add resolution schedule callback if provided
    if config.prog_resizing_fixres_schedule:
        resolution_callback = ResolutionScheduleCallback(schedule=config.prog_resizing_fixres_schedule)
        callbacks_list.append(resolution_callback)
        print("✅ Resolution schedule enabled with FixRes")

    # 4. Setup Logger
    # Lightning integrates with many loggers (TensorBoard, Weights & Biases, etc.)
    logger = TensorBoardLogger(
        save_dir=config.s3_dir+config.experiment_name+"/lightning_logs",
        name=config.experiment_name,
        version=None  # Auto-increment version numbers
    )

    # 5. Create Trainer
    # Trainer orchestrates the entire training process
    print("⚡ Setting up Lightning Trainer...")
    
    # Build trainer kwargs
    trainer_kwargs = {
        "max_epochs": config.epochs,
        "callbacks": callbacks_list,
        "logger": logger,
        "accelerator": "auto",
        "devices": config.num_devices if config.num_devices else "auto",
        "deterministic": True,
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "log_every_n_steps": config.log_every_n_steps,
        "precision": config.precision,
        "gradient_clip_val": config.gradient_clip_val,
        "enable_progress_bar": True,
        "reload_dataloaders_every_n_epochs": 1,
        "accumulate_grad_batches": config.accumulate_grad_batches,
    }
    
    # Add strategy for multi-GPU training
    if config.strategy:
        trainer_kwargs["strategy"] = config.strategy

    if config.s3_dir:
        trainer_kwargs["default_root_dir"] = config.s3_dir
        
    trainer = L.Trainer(**trainer_kwargs)

    # 6. Start Training!
    print("🚀 Starting training...")
    print(f"📁 Logs will be saved to: {get_relative_path(logger.log_dir)}")
    print(f"💾 Checkpoints will be saved to: {get_relative_path(checkpoint_callback.dirpath)}")
    print("=" * 60)

    # Automatic checkpoint detection for resuming training
    if resume_from_checkpoint is None:
        # Try to find last checkpoint automatically (only create dir on rank 0 to avoid race condition)
        if trainer.is_global_zero:
            Path(checkpoint_callback.dirpath).mkdir(parents=True, exist_ok=True)
        
        last_ckpt = Path(checkpoint_callback.dirpath) / "last.ckpt"
        if last_ckpt.exists():
            resume_from_checkpoint = str(last_ckpt)
            if trainer.is_global_zero:
                print(f"🔄 Found existing checkpoint, resuming from: {get_relative_path(resume_from_checkpoint)}")
        else:
            if trainer.is_global_zero:
                print("🆕 No checkpoint found, starting fresh training")
    else:
        if trainer.is_global_zero:
            print(f"🔄 Resuming from specified checkpoint: {get_relative_path(resume_from_checkpoint)}")
    
    print("=" * 60)

    # Fit the model (train + validate) - pass checkpoint path
    trainer.fit(model, datamodule, ckpt_path=resume_from_checkpoint)

    # 7. Test the model (Need to implement test step in the LightningModule)
    # print("\n🧪 Testing the model...")
    # trainer.test(model, datamodule)

    # 8. Print results
    print("\n✅ Training completed!")
    print(f"📊 Best model checkpoint: {get_relative_path(checkpoint_callback.best_model_path)}")
    print(f"🏆 Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"📈 View training logs: tensorboard --logdir {get_relative_path(logger.log_dir)}")

    return None

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on ImageNet with hardware-specific configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on local MacBook M4 Pro
  python train.py --config local
  
  # Train on AWS g5.12xlarge
  python train.py --config g5
  
  # Train on AWS p4d.24xlarge
  python train.py --config p4
  
  # Custom learning rate
  python train.py --config g5 --lr 0.001
  
  # Resume from checkpoint
  python train.py --config g5 --resume logs/experiment/checkpoints/last.ckpt
  
  # List available configs
  python train.py --list-configs
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='local',
        choices=['local', 'g5', 'p4'],
        help='Hardware configuration profile (default: local)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=None,
        dest='learning_rate',
        help='Learning rate (overrides config value if provided)'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configuration profiles and exit'
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
    
    # Start training
    train_with_lightning(
        config=config,
        resume_from_checkpoint=args.resume
    )
    
    print("\n🎯 To view training progress:")
    print(f"tensorboard --logdir {config.logs_dir / config.experiment_name / 'lightning_logs'}")
    print("\n🔍 To load the best model:")
    print("model = ResnetLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')")


if __name__ == "__main__":
    main()
    
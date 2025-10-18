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
"""
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

# Import our Lightning components
from imagenet_datamodule import ImageNetDataModule
from resnet_module import ResnetLightningModule
from text_logging_callback import TextLoggingCallback
from resolution_schedule_callback import ResolutionScheduleCallback
from config import weight_decay, learning_rate, logs_dir, epochs, batch_size, experiment_name, mean, std, total_steps, num_classes, progressive_resizing_fixres_schedule
from utils import get_relative_path

def train_with_lightning(
    max_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = learning_rate,
    experiment_name: str = "imagenet_training",
    resume_from_checkpoint: str = None, # Resume training from last checkpoint path 
    use_sam: bool = False,
    resolution_schedule: dict = None
):
    """
    Train Imagenet dataset on Resnet50 using PyTorch Lightning
    
    Args:
        max_epochs: Maximum number of epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        experiment_name: Name for the experiment (used in logging)
        resume_from_checkpoint: Path to checkpoint to resume training from where it stopped
        use_sam: Whether to use SAM optimizer
        resolution_schedule: Dict for progressive resizing/FixRes
                             Example: {0: (128, True, 512), 10: (224, True, 320), 85: (288, False, 256)}
    """
    print("🌩️ Starting PyTorch Lightning Training")
    print("=" * 60)

    # Determine initial resolution from schedule or use default
    initial_resolution = 224
    use_train_augs = True
    if resolution_schedule:
        # Get the config for epoch 0 (or first epoch in schedule)
        first_epoch = min(resolution_schedule.keys())
        config = resolution_schedule[first_epoch]
        initial_resolution = config[0]
        use_train_augs = config[1]
        if len(config) > 2:
            batch_size = config[2]  # Override batch size if specified
    
    # 1. Create DataModule
    # DataModule handles all data operations
    print("📊 Setting up data...")
    datamodule = ImageNetDataModule(
        batch_size=batch_size,
        num_workers=8,  # Adjust based on your CPU
        initial_resolution=initial_resolution,
        use_train_augs=use_train_augs
    )

    # 2. Create Lightning Module (Model)
    print("🧠 Setting up model...")
    model = ResnetLightningModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        train_transforms=train_transforms,
        total_steps=total_steps,
        num_classes=num_classes
    )
    
    # 3. Setup Callbacks
    # Callbacks add functionality without cluttering the main code
    print("⚙️ Setting up callbacks...")
    # Model Checkpointing - saves best models automatically
    checkpoint_callback = ModelCheckpoint(
        dirpath=logs_dir / experiment_name / "lightning_checkpoints",
        filename="imagenet1k-{epoch:02d}-{val/accuracy:.3f}",
        monitor="val/accuracy",  # Metric to monitor
        mode="max",             # Save model with highest accuracy
        save_top_k=1,           # Keep top 3 models
        save_last=True,         # Save the last model
        verbose=True
    )

    # Early Stopping - stops training if no improvement
    early_stop_callback = EarlyStopping(
        monitor="val/loss",     # Metric to monitor
        patience=5,             # Number of epochs to wait
        verbose=True,
        mode="min"             # Stop when val_loss stops decreasing
    )

    # Rich Progress Bar - beautiful progress bars
    progress_bar = RichProgressBar()

    # Text Logging Callback - creates detailed text logs
    text_logger = TextLoggingCallback(
        log_dir=logs_dir,
        experiment_name=experiment_name
    )

    # 4. Setup Logger
    # Lightning integrates with many loggers (TensorBoard, Weights & Biases, etc.)
    logger = TensorBoardLogger(
        save_dir=logs_dir / experiment_name / "lightning_logs",
        name=experiment_name,
        version=None  # Auto-increment version numbers
    )

    # 5. Create Trainer
    # Trainer orchestrates the entire training process
    print("⚡ Setting up Lightning Trainer...")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        
        # Callbacks
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            progress_bar,
            text_logger,  # Add text logging
            # model_summary_callback
        ],
        
        # Logger
        logger=logger,
        
        # Hardware settings
        accelerator="auto",      # Automatically choose GPU/CPU
        devices="auto",          # Use all available devices
        
        # Training settings
        # gradient_clip_val=0.5,   # Gradient clipping for stability
        deterministic=True,      # For reproducibility
        
        # Validation settings
        check_val_every_n_epoch=1,  # Validate every epoch
        
        # Logging settings
        log_every_n_steps=50,    # Log metrics every 50 steps
        
        # Performance settings
        precision="16-mixed",     # Use 16-bit mixed precision for speed (you can use 32-true for more precision)
        
        # Progress bar
        enable_progress_bar=True,
        # enable_model_summary=True,
    )

    # 6. Start Training!
    print("🚀 Starting training...")
    print(f"📁 Logs will be saved to: {get_relative_path(logger.log_dir)}")
    print(f"💾 Checkpoints will be saved to: {get_relative_path(checkpoint_callback.dirpath)}")
    print("=" * 60)

    # Automatic checkpoint detection for resuming training
    if resume_from_checkpoint is None:
        # Try to find last checkpoint automatically
        checkpoint_callback.dirpath.mkdir(parents=True, exist_ok=True)
        last_ckpt = checkpoint_callback.dirpath / "last.ckpt"
        if last_ckpt.exists():
            resume_from_checkpoint = str(last_ckpt)
            print(f"🔄 Found existing checkpoint, resuming from: {get_relative_path(resume_from_checkpoint)}")
        else:
            print("🆕 No checkpoint found, starting fresh training")
    else:
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

if __name__ == "__main__":
    # Example usage
    train_with_lightning(
        max_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name=experiment_name
    )
    
    print("\n🎯 To view training progress:")
    print("tensorboard --logdir lightning_logs")
    print("\n🔍 To load the best model:")
    print("model = ResnetLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')")
    
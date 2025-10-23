"""
PyTorch Lightning Module for training Imagenet1K with Resnet50

LightningModule encapsulates:
- Model architecture
- Training/validation/test logic
- Optimizer configuration
- Metrics computation
- Logging

Benefits:
- Clean separation of concerns
- Automatic logging and checkpointing
- Easy multi-GPU scaling
- Built-in best practices
"""

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from timm.data.mixup import Mixup
from config import scheduler_type
import copy
from torchinfo import summary
from lightning.pytorch.utilities import rank_zero_only
import timm

class ResnetLightningModule(L.LightningModule):
    """
    Lightning wrapper for ImageNet resnet model
    
    This class defines:
    - Forward pass
    - Training step
    - Validation step  
    - Optimizer configuration
    - Metrics tracking
    
    Note: Works with FFCV dataloaders for fast data loading
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        num_classes: int = 100,
        mixup_kwargs: dict = None
        ):
        super().__init__()
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Initialize MixUp/CutMix if enabled
        self.mixup_cutmix_fn = None
        if mixup_kwargs is not None and (mixup_kwargs.get('mixup_alpha', 0.0) > 0 or mixup_kwargs.get('cutmix_alpha', 0.0) > 0):
            self.mixup_cutmix_fn = Mixup(
                mixup_alpha=mixup_kwargs.get('mixup_alpha', 0.0),
                cutmix_alpha=mixup_kwargs.get('cutmix_alpha', 0.0),
                cutmix_minmax=mixup_kwargs.get('cutmix_minmax', None),
                prob=mixup_kwargs.get('prob', 1.0),
                switch_prob=mixup_kwargs.get('switch_prob', 0.5),
                mode=mixup_kwargs.get('mode', 'batch'),
                label_smoothing=mixup_kwargs.get('label_smoothing', 0.1),
                num_classes=num_classes
            )
            
            # Print what's enabled
            mixup_alpha = mixup_kwargs.get('mixup_alpha', 0.0)
            cutmix_alpha = mixup_kwargs.get('cutmix_alpha', 0.0)
            
            if mixup_alpha > 0 and cutmix_alpha > 0:
                print(f"✅ MixUp (α={mixup_alpha}) + CutMix (α={cutmix_alpha}) enabled")
            elif mixup_alpha > 0:
                print(f"✅ MixUp enabled with alpha={mixup_alpha}")
            elif cutmix_alpha > 0:
                print(f"✅ CutMix enabled with alpha={cutmix_alpha}")
        else:
            print("ℹ️  MixUp/CutMix disabled (set mixup_alpha > 0 or cutmix_alpha > 0 to enable)")
        
        # Save hyperparameters - Lightning logs these automatically
        self.save_hyperparameters()

        # Initialize model
        # self.model = torch.hub.load("pytorch/vision", "resnet50", weights=None)
        # Use antialiased_cnns for implementing Blur Pool
        # Use channels_last memory format for faster training
        self.model = timm.create_model('resnetblur50', pretrained=False, num_classes=self.num_classes)
        self.model = self.model.to(memory_format=torch.channels_last)

        # Initialize metrics for each stage
        # Why separate metrics? Each stage (train/val/test) needs independent tracking
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        
    def forward(self, x):
        """
        Forward pass - just call the model
        
        Note: FFCV outputs data in channels_last format by default (via ToTorchImage),
        which matches our model's memory format for optimal performance.
        """
        # Ensure input is in channels_last format for consistency
        x = x.to(memory_format=torch.channels_last)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step - called for each training batch
        
        Args:
            batch: (images, labels) tuple
            batch_idx: batch index
            
        Returns:
            loss: training loss (Lightning uses this for backprop)
        """
        images, labels = batch
        
        # Apply MixUp if enabled (before forward pass)
        if self.mixup_cutmix_fn is not None:
            images, labels = self.mixup_cutmix_fn(images, labels)
        
        # Forward pass
        logits = self(images) # Automatically calls self.forward(images)
        
        # Compute loss
        # Note: When MixUp is enabled, labels are soft (one-hot encoded)
        # F.cross_entropy handles both hard labels (class indices) and soft labels (probabilities)
        if self.mixup_cutmix_fn is not None:
            # Soft labels from MixUp - no label smoothing needed (already done by Mixup)
            loss = F.cross_entropy(logits, labels)
        else:
            # Hard labels - apply label smoothing
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        
        # Update metrics (convert soft labels back to hard labels for accuracy calculation)
        if self.mixup_cutmix_fn is not None:
            # For soft labels, take argmax to get predicted class
            labels_for_metrics = labels.argmax(dim=1)
        else:
            labels_for_metrics = labels
        
        self.train_accuracy(logits, labels_for_metrics)
        
        # Log metrics - Lightning handles the logging automatically
        # Note: sync_dist=True ensures metrics are properly aggregated across all GPUs in DDP mode
        # In single-GPU mode, sync_dist=True is a no-op (no performance penalty)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step - called for each validation batch
        
        Args:
            batch: (images, labels) tuple
            batch_idx: batch index
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        
        # Update metrics (pass logits directly for top-k accuracy)
        self.val_accuracy(logits, labels)
        
        # Log metrics - sync_dist=True aggregates metrics across GPUs
        # In single-GPU mode, sync_dist=True is a no-op (no performance penalty)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers
        
        Returns:
            optimizer or dict with optimizer and scheduler
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9
            )
        
        if scheduler_type == 'one_cycle_policy':
            # Calculate steps per epoch using Lightning's estimated_stepping_batches
            # This is available during configure_optimizers and accounts for all training settings
            total_steps = self.trainer.estimated_stepping_batches
            
            # Create OneCycle scheduler with EXACT parameters
            # This was setup in notebook by running set up ocp function
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,        
                steps=total_steps,
                pct_start=0.2,          
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=100.0,
                final_div_factor=1000.0
            )
            
            print(f"🔄 Recreated OneCycleLR Scheduler:")
            print(f"   Max LR: {self.learning_rate:.4e}")
            print(f"   Initial LR: {self.learning_rate/100.0:.4e}")
            print(f"   Total steps: {total_steps}")
            print(f"   Num devices: {self.trainer.num_devices}")
            print(f"   Strategy: {self.trainer.strategy.__class__.__name__}")
            print(f"   Pct start: {0.2}")
            print(f"   Div factor: {100.0}")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycle updates every step
                    "frequency": 1,
                    "name": "OneCycleLR"
                }
            }
        elif scheduler_type == 'cosine_annealing':
            # CosineAnnealingLR scheduler - gradually decreases learning rate following a cosine curve
            # Calculate total steps for step-based scheduling
            # Note: estimated_stepping_batches already accounts for all epochs
            total_steps = self.trainer.estimated_stepping_batches
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,  # Total number of training steps for one cosine cycle
                eta_min=1e-6  # Minimum learning rate (prevents going to zero)
            )
            
            print(f"📉 CosineAnnealingLR Scheduler:")
            print(f"   Initial LR: {self.learning_rate:.4e}")
            print(f"   Min LR (eta_min): {1e-6:.4e}")
            print(f"   T_max (steps): {total_steps}")
            print(f"   Num devices: {self.trainer.num_devices}")
            print(f"   Strategy: {self.trainer.strategy.__class__.__name__}")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # CosineAnnealing updates every step
                }
            }
        else:
            # Optional: Add learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        mode='max',           # Monitor accuracy (maximize)
                        factor=0.3,           # More aggressive reduction: LR *= 0.2 (instead of 0.5)
                        patience=3,           # Reduce patience: wait only 2 epochs (instead of 3)
                        threshold=0.001,      # Only reduce if improvement < 0.1%
                        min_lr=1e-6,          # Prevent LR from going too low
                        cooldown=1            # Wait n epoch after reduction before monitoring again
                    )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/accuracy",  # Metric to monitor
                "frequency": 1,
                "interval": "epoch"
            }
        }

    @rank_zero_only
    def on_train_start(self):
        """Called at the start of training - log model graph (only on rank 0 to avoid duplication in DDP)"""
        # Log model architecture to TensorBoard
        if self.logger is None:
            return
        
        try:
            # 1. Log model graph to TensorBoard
            sample_input = torch.randn(1, 3, 224, 224).to(self.device)
            self.logger.experiment.add_graph(self, sample_input)
            print("✅ Model graph logged to TensorBoard")
            
            # 2. Log model summary as text
            try:
                # Capture torchsummary output
                arch_summary = summary(copy.deepcopy(self.model), input_size=(1, 3, 224, 224), verbose=0, device=self.device)
                # Log to TensorBoard TEXT tab
                self.logger.experiment.add_text(
                    "Model/Architecture_Summary", 
                    f"```\n{arch_summary}\n```",
                    0
                )
                print("✅ Model summary logged to TensorBoard TEXT tab")
                
            except ImportError:
                print("⚠️  ModelSummary not found. Install with: pip install torchinfo")

            # 3. Log model parameters count
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            params_info = f"""
            **Model Parameters:**
            - Total Parameters: {total_params:,}
            - Trainable Parameters: {trainable_params:,}
            - Non-trainable Parameters: {total_params - trainable_params:,}
            
            **Model Architecture:**
            {str(self.model)}
            """
            
            self.logger.experiment.add_text(
                "Model/Parameters_Info", 
                params_info, 
                0
            )
            print("✅ Model parameters info logged to TensorBoard")
            
        except Exception as e:
            print(f"❌ Error logging model info: {e}")

    def on_train_epoch_start(self):
        """Called at the start of each training epoch - log current learning rate"""
        # Get current learning rate from optimizer
        try:
            # Get all optimizers (usually just one)
            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            
            # Get LR from first optimizer, first param group
            current_lr = optimizers[0].param_groups[0]['lr']
            
            # Log to console (only rank 0 to avoid spam in DDP)
            if self.trainer.is_global_zero:
                print(f"\n📚 Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | Learning Rate: {current_lr:.6e}")
            
            # Log to TensorBoard (sync_dist not needed for LR as it's the same across all ranks)
            self.log('train/learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=False)
            
        except Exception as e:
            # If we can't get LR (e.g., before optimizer is created), skip silently
            pass

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Get current metrics
        train_acc = self.train_accuracy.compute()
        
        if self.logger is not None:
            self.logger.experiment.add_text(
                "Training/Epoch_Results", 
                f"🚀 Epoch {self.current_epoch}: Train Acc: {train_acc:.3f}",
                self.current_epoch
            )

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        # Get current metrics
        val_acc = self.val_accuracy.compute()
        
        if self.logger is not None:
            self.logger.experiment.add_text(
                "Validation/Epoch_Results", 
                f"📊 Epoch {self.current_epoch}: Val Acc: {val_acc:.3f}",
                self.current_epoch
            )

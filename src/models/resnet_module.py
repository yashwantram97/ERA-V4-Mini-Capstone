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

from src.utils.utils import serialize_transforms
from pytorch_optimizer import SAM
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import antialiased_cnns
# import timm
from config import scheduler_type
import copy
from torchinfo import summary
from lightning.pytorch.utilities import rank_zero_only

class ResnetLightningModule(L.LightningModule):
    """
    Lightning wrapper for Cifar100 resnet model
    
    This class defines:
    - Forward pass
    - Training step
    - Validation step  
    - Optimizer configuration
    - Metrics tracking
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        num_classes: int = 100,
        train_transforms = None,
        # total_steps: int = None,
        use_sam: bool = False
        ):
        super().__init__()
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        # self.total_steps = total_steps
        # Store transforms for hyperparameter logging
        self.train_transforms = train_transforms

        self.use_sam = use_sam

        # We'll create a custom dict to include serialized transforms
        hparams_dict = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_classes": num_classes,
        }

        # Add serialized transforms if available
        if train_transforms is not None:
            hparams_dict["train_transforms"] = serialize_transforms(train_transforms)
        
        # Save hyperparameters - Lightning logs these automatically
        self.save_hyperparameters()

        # Initialize model
        # self.model = torch.hub.load("pytorch/vision", "resnet50", weights=None)
        # Use antialiased_cnns for implementing Blur Pool
        if self.num_classes == 100:
            self.model = antialiased_cnns.resnet50(pretrained=False)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model = antialiased_cnns.resnet50(pretrained=False)
        # Use channels_last memory format for faster training
        self.model = self.model.to(memory_format=torch.channels_last)

        # Initialize metrics for each stage
        # Why separate metrics? Each stage (train/val/test) needs independent tracking
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)            
    def forward(self, x):
        """Forward pass - just call the model"""
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
        
        # Forward pass
        logits = self(images) # Automatically calls self.forward(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1) # Added Label smoothing
        
        # Update metrics (pass logits directly for top-k accuracy)
        self.train_accuracy(logits, labels)
        
        # Log metrics - Lightning handles the logging automatically
        # Note: sync_dist=True ensures metrics are properly aggregated across all GPUs in DDP mode
        # In single-GPU mode, sync_dist=True is a no-op (no performance penalty)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers
        
        Returns:
            optimizer or dict with optimizer and scheduler
        """

        # if self.use_sam:
        #     # Use SAM optimizer
        #     base_optimizer = torch.optim.SGD
        #     optimizer = SAM(
        #         self.model.parameters(),
        #         base_optimizer=base_optimizer,
        #         lr=self.learning_rate,
        #         weight_decay=self.weight_decay,
        #         momentum=0.9
        #     )
        # else:
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
            steps_per_epoch = total_steps // self.trainer.max_epochs
            
            # Create OneCycle scheduler with EXACT parameters
            # This was setup in notebook by running set up ocp function
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,        
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,          
                anneal_strategy='cos',  
                cycle_momentum=True,    
                base_momentum=0.85,     
                max_momentum=0.95,      
                div_factor=100.0,        # Calculated: max_lr/base_lr = 2.35e-04/2.35e-05
                final_div_factor=1000.0  
            )
            
            print(f"🔄 Recreated OneCycleLR Scheduler:")
            print(f"   Max LR: {self.learning_rate:.4e}")
            print(f"   Initial LR: {self.learning_rate/100.0:.4e}")
            print(f"   Total steps: {total_steps}")
            print(f"   Steps per epoch: {steps_per_epoch}")
            print(f"   Total epochs: {self.trainer.max_epochs}")
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override optimizer step to support SAM's two-step optimization.
        
        This method is called by Lightning instead of optimizer.step().
        For SAM, we need to:
        1. First step: compute gradients and take ascent step
        2. Second step: recompute gradients and take descent step
        """
        # Lightning wraps optimizers in LightningOptimizer
        # Access the actual optimizer
        actual_optimizer = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer

        # Check if we're using SAM optimizer
        if isinstance(actual_optimizer, SAM):
            # SAM's two-step optimization
            # Step 1: Ascent step (move to adversarial parameters)
            actual_optimizer.first_step(zero_grad=True)
            
            # Step 2: Re-compute loss and gradients at adversarial parameters
            # Lightning will handle the closure call for us
            optimizer_closure()
            
            # Step 3: Descent step (update actual parameters)
            actual_optimizer.second_step(zero_grad=True)
        else:
            # Standard optimizer step for non-SAM optimizers
            optimizer.step(closure=optimizer_closure)

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

"""
Text Logging Callback for PyTorch Lightning
Adapted from original PyTorch logging setup to provide comprehensive logging

Features:
- Detailed timestamps with function/line info
- Structured logging
- Multiple log files (training.log, metrics.json, model_info.txt)
- Proper logging levels and formatters
- DDP-safe with rank_zero_only decorators for file operations
"""

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import torch
from torchinfo import summary
# Lazy imports to avoid circular dependencies:
# - get_relative_path: imported in methods that need it
# - input_size: imported in _log_model_info

class TextLoggingCallback(Callback):
    """
    text logging callback that Creates structured logs with detailed
    formatting and multiple output files
    """
    def __init__(self, log_dir: Path, experiment_name: str = "lightning_experiment", log_level: str = "INFO"):
        super().__init__()

        # Create experiment directory structure
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log files
        self.training_log_file = self.log_dir / "training.log"
        self.metrics_json_file = self.log_dir / "metrics.json"
        self.model_info_file = self.log_dir / "model_info.txt"

        # Initialize metrics storage
        self.metrics_history = []
        self.start_time = None
        self.experiment_start_time = None

        # Setup structured logger
        self.logger = self._setup_logger(log_level)

        from src.utils.utils import get_relative_path
        print(f"📝 text logs will be saved to: {get_relative_path(self.log_dir)}")

    def _setup_logger(self, log_level: str = "INFO"):
        """Setup comprehensive logging"""
        logger_name = f'lightning_training_{self.experiment_name}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # File handler for detailed logs
        file_handler = logging.FileHandler(self.training_log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _format_metric(self, value, format_spec=".4f", default="N/A"):
        """Safely format a metric value"""
        if isinstance(value, (int, float, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.item()
            return f"{value:{format_spec}}"
        return default

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Log training start with detailed information (only on rank 0 to avoid log duplication)"""
        from src.utils.utils import get_relative_path
        
        self.start_time = datetime.now()
        self.experiment_start_time = time.time()

        # Log experiment start
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 Starting Lightning experiment: {self.experiment_name}")
        self.logger.info(f"📁 Experiment directory: {get_relative_path(self.log_dir)}")
        self.logger.info(f"💾 Log file: {get_relative_path(self.training_log_file)}")
        self.logger.info("=" * 80)

        # Log model information
        self._log_model_info(pl_module)

        # Log training configuration
        self._log_training_config(trainer, pl_module)

        # Log dataset information if available
        if hasattr(trainer, 'datamodule') and trainer.datamodule:
            self._log_dataset_info(trainer.datamodule)

    def _log_model_info(self, pl_module):
        """Log model architecture and parameters, Also added in on_train_star in lit_module check if that can be removed"""
        from config import input_size
        
        self.logger.info("📋 Detailed Model Summary:")
        self.logger.info("-" * 60)

        # Log parameter counts
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        self.logger.info(f"   Total Parameters: {total_params:,}")
        self.logger.info(f"   Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"   Non-trainable Parameters: {total_params - trainable_params:,}")

        try:
            # Capture torchsummary output
            arch_summary = summary(pl_module.model, input_size=input_size, verbose=0, device=pl_module.device)
            self.logger.info(f"```\n{arch_summary}\n```",)
        except Exception as e:
            self.logger.warning(f"Could not generate model summary: {e}")

        # Also save detailed model info to separate file
        self._save_model_info_to_file(pl_module, total_params, trainable_params)

    @rank_zero_only
    def _save_model_info_to_file(self, pl_module, total_params, trainable_params):
        """Save detailed model info to separate file (only on rank 0 to avoid conflicts in DDP)"""
        with open(self.model_info_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LIGHTNING MODEL INFORMATION\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("PARAMETERS SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n\n")
            
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-"*40 + "\n")
            f.write(str(pl_module.model) + "\n\n")

    def _log_training_config(self, trainer, pl_module):
        """Log training configuration (matching original style)"""
        self.logger.info("⚙️ Training Configuration:")
        self.logger.info(f"   Device: {pl_module.device}")
        self.logger.info(f"   Accelerator: {trainer.accelerator}")
        self.logger.info(f"   Max Epochs: {trainer.max_epochs}")
        self.logger.info(f"   Learning Rate: {pl_module.learning_rate}")
        self.logger.info(f"   Weight Decay: {pl_module.weight_decay}")

        # Log data transforms
        if hasattr(pl_module, 'train_transforms') and pl_module.train_transforms:
            self.logger.info("📊 Data Transforms:")
            self.logger.info("   Training Transforms:")
            for i, transform in enumerate(pl_module.train_transforms.transforms):
                self.logger.info(f"     {i+1}. {transform}")

        # Log optimizer and scheduler info if available
        if hasattr(pl_module, 'configure_optimizers'):
            try:
                optimizer_config = pl_module.configure_optimizers()
                
                # Handle different return formats from configure_optimizers
                if isinstance(optimizer_config, dict):
                    optimizer = optimizer_config.get('optimizer')
                    lr_scheduler_config = optimizer_config.get('lr_scheduler')
                elif isinstance(optimizer_config, (list, tuple)):
                    # Handle case where multiple optimizers/schedulers are returned
                    if len(optimizer_config) >= 1:
                        optimizer = optimizer_config[0]
                        lr_scheduler_config = optimizer_config[1] if len(optimizer_config) > 1 else None
                    else:
                        optimizer = None
                        lr_scheduler_config = None
                else:
                    # Single optimizer returned
                    optimizer = optimizer_config
                    lr_scheduler_config = None
                
                # Log optimizer details
                if optimizer:
                    self.logger.info("🔧 Optimizer Configuration:")
                    self.logger.info(f"   Optimizer: {optimizer.__class__.__name__}")
                    
                    # Log optimizer parameters
                    for group_idx, param_group in enumerate(optimizer.param_groups):
                        self.logger.info(f"   Optimizer group {group_idx}:")
                        for key, value in param_group.items():
                            if key != 'params':
                                self.logger.info(f"     {key}: {value}")
                
                # Log scheduler details
                if lr_scheduler_config:
                    self.logger.info("📅 Learning Rate Scheduler Configuration:")
                    
                    if isinstance(lr_scheduler_config, dict):
                        # Scheduler config is a dictionary
                        scheduler = lr_scheduler_config.get('scheduler')
                        monitor = lr_scheduler_config.get('monitor', 'N/A')
                        frequency = lr_scheduler_config.get('frequency', 1)
                        interval = lr_scheduler_config.get('interval', 'epoch')
                        
                        if scheduler:
                            self.logger.info(f"   Scheduler: {scheduler.__class__.__name__}")
                            self.logger.info(f"   Monitor: {monitor}")
                            self.logger.info(f"   Frequency: {frequency}")
                            self.logger.info(f"   Interval: {interval}")
                            
                            # Log scheduler-specific parameters
                            scheduler_params = {}
                            if hasattr(scheduler, '__dict__'):
                                for key, value in scheduler.__dict__.items():
                                    if not key.startswith('_') and key not in ['optimizer']:
                                        # Convert non-serializable types to strings
                                        if isinstance(value, (int, float, str, bool, list, tuple)):
                                            scheduler_params[key] = value
                                        elif value is None:
                                            scheduler_params[key] = None
                                        else:
                                            scheduler_params[key] = str(value)
                            
                            if scheduler_params:
                                self.logger.info("   Scheduler Parameters:")
                                for key, value in scheduler_params.items():
                                    self.logger.info(f"     {key}: {value}")
                    else:
                        # Direct scheduler object
                        self.logger.info(f"   Scheduler: {lr_scheduler_config.__class__.__name__}")
                        
                        # Log scheduler parameters
                        scheduler_params = {}
                        if hasattr(lr_scheduler_config, '__dict__'):
                            for key, value in lr_scheduler_config.__dict__.items():
                                if not key.startswith('_') and key not in ['optimizer']:
                                    if isinstance(value, (int, float, str, bool, list, tuple)):
                                        scheduler_params[key] = value
                                    elif value is None:
                                        scheduler_params[key] = None
                                    else:
                                        scheduler_params[key] = str(value)
                        
                        if scheduler_params:
                            self.logger.info("   Scheduler Parameters:")
                            for key, value in scheduler_params.items():
                                self.logger.info(f"     {key}: {value}")
                else:
                    self.logger.info("📅 Learning Rate Scheduler: None")
                    
            except Exception as e:
                self.logger.debug(f"Could not log optimizer/scheduler info: {e}")

    def _log_dataset_info(self, datamodule):
        """Log dataset information (matching original style)"""
        self.logger.info("📊 Dataset Information:")
        
        try:
            # Setup datamodule to get loaders
            datamodule.setup('fit')
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            self.logger.info(f"   Training batches: {len(train_loader)}")
            self.logger.info(f"   Validation batches: {len(val_loader)}")
            self.logger.info(f"   Training samples: {len(train_loader.dataset):,}")
            self.logger.info(f"   Validation samples: {len(val_loader.dataset):,}")
            self.logger.info(f"   Batch size: {train_loader.batch_size}")
            
        except Exception as e:
            self.logger.debug(f"Could not log dataset info: {e}")

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        """Log epoch start (only on rank 0 to avoid log duplication)"""
        self.logger.info(f"🔄 EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs} - Starting...")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training epoch results with detailed formatting (only on rank 0 to avoid log duplication)"""
        train_metrics = self._extract_metrics(trainer, pl_module, 'train')
        
        # Format metrics nicely
        loss_str = self._format_metric(train_metrics.get('loss'), ".4f")
        # Display accuracy as percentage
        acc = train_metrics.get('accuracy')
        if isinstance(acc, (int, float, torch.Tensor)):
            if isinstance(acc, torch.Tensor):
                acc = acc.item()
            acc_str = f"{acc * 100:.2f}%"
        else:
            acc_str = "N/A"
        f1_str = self._format_metric(train_metrics.get('f1_score'), ".3f")
        
        self.logger.info(f"📈 EPOCH {trainer.current_epoch + 1} TRAIN - Loss: {loss_str}, Acc: {acc_str}, F1: {f1_str}")
        
        # Store metrics for JSON export
        self.metrics_history.append({
            'epoch': trainer.current_epoch + 1,
            'stage': 'train',
            'metrics': train_metrics,
            'timestamp': datetime.now().isoformat()
        })

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation epoch results (only on rank 0 to avoid log duplication)"""
        val_metrics = self._extract_metrics(trainer, pl_module, 'val')
        
        # Format metrics nicely
        loss_str = self._format_metric(val_metrics.get('loss'), ".4f")
        # Display accuracy as percentage
        acc = val_metrics.get('accuracy')
        if isinstance(acc, (int, float, torch.Tensor)):
            if isinstance(acc, torch.Tensor):
                acc = acc.item()
            acc_str = f"{acc * 100:.2f}%"
        else:
            acc_str = "N/A"
        f1_str = self._format_metric(val_metrics.get('f1_score'), ".3f")
        
        self.logger.info(f"📊 EPOCH {trainer.current_epoch + 1} VAL   - Loss: {loss_str}, Acc: {acc_str}, F1: {f1_str}")
        
        # Store metrics for JSON export
        self.metrics_history.append({
            'epoch': trainer.current_epoch + 1,
            'stage': 'val',
            'metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        })

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        """Log test results (only on rank 0 to avoid log duplication)"""
        test_metrics = self._extract_metrics(trainer, pl_module, 'test')
        
        self.logger.info("=" * 50)
        self.logger.info("🧪 FINAL TEST RESULTS:")
        self.logger.info("-" * 30)
        
        # Log all test metrics
        for metric_name, value in test_metrics.items():
            # Display accuracy as percentage
            if 'accuracy' in metric_name.lower():
                if isinstance(value, (int, float, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    formatted_value = f"{value * 100:.2f}%"
                else:
                    formatted_value = "N/A"
            else:
                formatted_value = self._format_metric(value, ".4f" if 'loss' in metric_name else ".3f")
            self.logger.info(f"   Test {metric_name.title()}: {formatted_value}")
        
        self.logger.info("=" * 50)
        
        # Store test metrics
        self.metrics_history.append({
            'epoch': trainer.current_epoch + 1,
            'stage': 'test',
            'metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Log training completion and save metrics (only on rank 0 to avoid log duplication)"""
        from src.utils.utils import get_relative_path
        
        end_time = datetime.now()
        duration = time.time() - self.experiment_start_time if self.experiment_start_time else 0
        
        self.logger.info("=" * 80)
        self.logger.info("✅ TRAINING COMPLETED")
        self.logger.info(f"⏱️  Total Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        self.logger.info(f"📅 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"🏁 Final Epoch: {trainer.current_epoch + 1}")
        
        # Log best checkpoint info if available
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                self.logger.info(f"🏆 Best Model: {get_relative_path(callback.best_model_path)}")
                self.logger.info(f"📊 Best Score: {callback.best_model_score}")
                break
        
        self.logger.info("=" * 80)
        
        # Save metrics to JSON (matching your save_metrics_to_json function)
        self._save_metrics_to_json(duration)
        
        # Final summary
        self.logger.info(f"💾 All logs saved to: {get_relative_path(self.log_dir)}")
        self.logger.info(f"📄 Training log: {get_relative_path(self.training_log_file)}")
        self.logger.info(f"📊 Metrics JSON: {get_relative_path(self.metrics_json_file)}")
        self.logger.info(f"🔍 Model info: {get_relative_path(self.model_info_file)}")
    
    @rank_zero_only
    def _save_metrics_to_json(self, duration_seconds):
        """Save training metrics to JSON file (only on rank 0 to avoid conflicts in DDP)"""
        from src.utils.utils import get_relative_path
        
        # Prepare final metrics structure
        final_metrics = {}
        for entry in self.metrics_history:
            stage = entry['stage']
            epoch = entry['epoch']
            
            if stage not in final_metrics:
                final_metrics[stage] = {}
            
            final_metrics[stage][f'epoch_{epoch}'] = entry['metrics']
        
        # Add experiment metadata (matching your original structure)
        metrics_with_meta = {
            "experiment_info": {
                "experiment_name": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60,
                "total_epochs": max([entry['epoch'] for entry in self.metrics_history]) if self.metrics_history else 0,
            },
            "metrics": final_metrics,
            "raw_history": self.metrics_history
        }
        
        with open(self.metrics_json_file, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        self.logger.info(f"💾 Metrics saved to: {get_relative_path(self.metrics_json_file)}")

    def _extract_metrics(self, trainer, pl_module, stage):
        """Extract metrics from the trainer"""
        metrics = {}
        
        # First try: logged_metrics
        if hasattr(trainer, 'logged_metrics'):
            for key, value in trainer.logged_metrics.items():
                if key.startswith(f"{stage}/"):
                    metric_name = key.split('/')[-1]
                    if isinstance(value, torch.Tensor):
                        metrics[metric_name] = value.item()
                    else:
                        metrics[metric_name] = value
        
        # Second try: callback_metrics (Lightning's other metric storage)
        if hasattr(trainer, 'callback_metrics'):
            for key, value in trainer.callback_metrics.items():
                if key.startswith(f"{stage}/"):
                    metric_name = key.split('/')[-1]
                    if isinstance(value, torch.Tensor):
                        metrics[metric_name] = value.item()
                    else:
                        metrics[metric_name] = value
        
        # Third try: Direct access to metric objects
        if stage == 'train':
            try:
                if hasattr(pl_module, 'train_accuracy'):
                    metrics['accuracy'] = pl_module.train_accuracy.compute().item()
                if hasattr(pl_module, 'train_f1'):
                    metrics['f1_score'] = pl_module.train_f1.compute().item()
            except:
                pass
        elif stage == 'val':
            try:
                if hasattr(pl_module, 'val_accuracy'):
                    metrics['accuracy'] = pl_module.val_accuracy.compute().item()
                if hasattr(pl_module, 'val_f1'):
                    metrics['f1_score'] = pl_module.val_f1.compute().item()
            except:
                pass
        elif stage == 'test':
            try:
                if hasattr(pl_module, 'test_accuracy'):
                    metrics['accuracy'] = pl_module.test_accuracy.compute().item()
                if hasattr(pl_module, 'test_f1'):
                    metrics['f1_score'] = pl_module.test_f1.compute().item()
            except:
                pass
        
        return metrics
    
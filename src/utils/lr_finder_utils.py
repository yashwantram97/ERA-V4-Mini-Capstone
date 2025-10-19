import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from config import device, logs_dir, experiment_name
import os

def run_lr_finder(model: nn.Module, 
                  train_loader: DataLoader, 
                  loss_fn: nn.Module,
                  optimizer: type,
                  start_lr: float = 1e-7,
                  end_lr: float = 10,
                  num_iter: int = 500,
                  step_mode: str = 'exp',
                  smooth_f: float = 0.05,
                  save_path: Optional[Path] = None,
                  logger = None) -> Tuple[float, float]:
    """
    Run LR Finder to find optimal learning rate
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer_class: Optimizer class (default: SGD)
        optimizer_kwargs: Additional optimizer parameters
        start_lr: Starting learning rate for search
        end_lr: Ending learning rate for search
        num_iter: Number of iterations for LR search
        step_mode: How to step between start_lr and end_lr ('exp' or 'linear')
        smooth_f: Smoothing factor for loss curve
        save_path: Path to save the LR finder plot
        logger: Logger instance for logging
        
    Returns:
        suggested_lr: double
    """
    if not save_path:
        save_path = logs_dir / experiment_name / "lr_finder.png"
    
    # Initialize LR Finder
    lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    
    if logger:
        logger.info("🔍 Starting Learning Rate Finder...")
        logger.info(f"   Search range: {start_lr:.2e} to {end_lr:.2e}")
        logger.info(f"   Number of iterations: {num_iter}")
        logger.info(f"   Step mode: {step_mode}")
    else:
        print("🔍 Starting Learning Rate Finder...")
        print(f"   Search range: {start_lr:.2e} to {end_lr:.2e}")
        print(f"   Number of iterations: {num_iter}")
        print(f"   Step mode: {step_mode}")
    
    # Run the LR range test
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        smooth_f=smooth_f
    )
    
    # Get suggested learning rate using the plot method
    # The plot method returns (ax, suggested_lr)
    if save_path:
        plt.figure(figsize=(10, 6))
        ax, suggested_lr = lr_finder.plot(skip_start=10, skip_end=5, suggest_lr=True, ax=plt.gca())
        plt.title('Learning Rate Finder', fontsize=16, fontweight='bold')
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotation for suggested learning rate
        plt.axvline(x=suggested_lr, color='red', linestyle='--', alpha=0.7, 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"📊 LR Finder plot saved to: {save_path}")
    else:
        print(f"📊 LR Finder plot saved to: {save_path}")
        # If no save path, just get the suggested LR without plotting
        _, suggested_lr = lr_finder.plot(skip_start=10, skip_end=5, suggest_lr=True)
    
    if logger:
        logger.info(f"📊 LR Finder Results:")
        logger.info(f"   Suggested LR: {suggested_lr:.2e}")
    else:
        print(f"📊 LR Finder Results:")
        print(f"   Suggested LR: {suggested_lr:.2e}")
    # Reset the model and optimizer to original state
    lr_finder.reset()
    
    return suggested_lr
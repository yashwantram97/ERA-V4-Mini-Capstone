"""
Robust Learning Rate Finder - Run Multiple Times

This script runs the LR finder multiple times and provides statistics
to help you choose a more reliable learning rate.

Usage:
    python find_lr_robust.py --config local --runs 5
    python find_lr_robust.py --config g5 --runs 3
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.lr_finder_utils import run_lr_finder
from src.utils.utils import get_transforms
from src.data_modules.imagenet_datamodule import ImageNetDataModule
from src.models.resnet_module import ResnetLightningModule
from configs import get_config
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run LR finder multiple times for robust results"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='local',
        choices=['local', 'g5', 'p3'],
        help='Hardware configuration profile (default: local)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of times to run LR finder (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n🔧 Loading configuration: {args.config}")
    config = get_config(args.config)
    print(config)
    
    # Setup
    train_transforms = get_transforms(transform_type="train", mean=config.mean, std=config.std)
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
    
    # Run multiple times
    suggested_lrs = []
    
    print("\n" + "="*80)
    print(f"🚀 RUNNING LR FINDER {args.runs} TIMES FOR ROBUST RESULTS")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"   Range: {config.lr_finder_kwargs['start_lr']:.2e} to {config.lr_finder_kwargs['end_lr']:.2e}")
    print(f"   Iterations per run: {config.lr_finder_kwargs['num_iter']}")
    print(f"   Number of runs: {args.runs}")
    print(f"   Total iterations: {config.lr_finder_kwargs['num_iter'] * args.runs}")
    
    for run_num in range(1, args.runs + 1):
        print(f"\n{'='*80}")
        print(f"🔄 RUN {run_num}/{args.runs}")
        print(f"{'='*80}")
        
        # Create fresh model and optimizer for each run
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
        
        # Run LR finder
        save_path = experiment_dir / f"lr_finder_run{run_num}.png"
        suggested_lr = run_lr_finder(
            model,
            train_loader,
            loss_fn,
            optimizer,
            **config.lr_finder_kwargs,
            save_path=save_path,
            logger=None
        )
        
        suggested_lrs.append(suggested_lr)
        print(f"✅ Run {run_num} suggested LR: {suggested_lr:.2e}")
        
        # Clean up
        del model, optimizer, lit_module
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate statistics
    suggested_lrs = np.array(suggested_lrs)
    
    # Compute statistics
    mean_lr = np.mean(suggested_lrs)
    median_lr = np.median(suggested_lrs)
    std_lr = np.std(suggested_lrs)
    min_lr = np.min(suggested_lrs)
    max_lr = np.max(suggested_lrs)
    geom_mean_lr = np.exp(np.mean(np.log(suggested_lrs)))
    
    # Coefficient of variation (relative std dev)
    cv = std_lr / mean_lr
    
    # Print results
    print("\n" + "="*80)
    print("📊 STATISTICAL ANALYSIS OF LR FINDER RESULTS")
    print("="*80)
    
    print(f"\n📈 All Suggested Learning Rates:")
    for i, lr in enumerate(suggested_lrs, 1):
        print(f"   Run {i}: {lr:.4e}")
    
    print(f"\n📊 Statistics:")
    print(f"   Mean (arithmetic):     {mean_lr:.4e}")
    print(f"   Median:                {median_lr:.4e}")
    print(f"   Geometric Mean:        {geom_mean_lr:.4e}  ⭐ RECOMMENDED")
    print(f"   Std Deviation:         {std_lr:.4e}")
    print(f"   Coefficient of Var:    {cv:.2%}")
    print(f"   Min:                   {min_lr:.4e}")
    print(f"   Max:                   {max_lr:.4e}")
    print(f"   Range (max/min):       {max_lr/min_lr:.2f}x")
    
    # Interpretation
    print(f"\n🎯 Interpretation:")
    if cv < 0.3:
        print(f"   ✅ Low variation ({cv:.1%}) - Results are very consistent")
        confidence = "HIGH"
    elif cv < 0.5:
        print(f"   ✅ Moderate variation ({cv:.1%}) - Results are reasonably consistent")
        confidence = "MEDIUM"
    else:
        print(f"   ⚠️  High variation ({cv:.1%}) - Consider running more iterations")
        confidence = "LOW"
    
    print(f"   Confidence Level: {confidence}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   ")
    print(f"   1. 🥇 Best Choice (Geometric Mean):")
    print(f"      LEARNING_RATE = {geom_mean_lr:.4e}")
    print(f"      → Accounts for exponential nature of learning rates")
    print(f"      → Less sensitive to outliers")
    print(f"   ")
    print(f"   2. 🥈 Conservative Choice (Median):")
    print(f"      LEARNING_RATE = {median_lr:.4e}")
    print(f"      → Middle value, robust to outliers")
    print(f"      → Safer for training stability")
    print(f"   ")
    print(f"   3. 🥉 Aggressive Choice (75th percentile):")
    percentile_75 = np.percentile(suggested_lrs, 75)
    print(f"      LEARNING_RATE = {percentile_75:.4e}")
    print(f"      → Higher LR for faster convergence")
    print(f"      → Monitor for instability")
    print(f"   ")
    print(f"   4. 🛡️  Safe Range:")
    print(f"      MIN: {median_lr * 0.5:.4e}  (50% of median)")
    print(f"      MAX: {median_lr * 2.0:.4e}  (200% of median)")
    
    # Create visualization
    create_summary_plot(suggested_lrs, geom_mean_lr, median_lr, experiment_dir)
    
    print(f"\n📊 Individual run plots saved to:")
    for i in range(1, args.runs + 1):
        print(f"   {experiment_dir / f'lr_finder_run{i}.png'}")
    print(f"\n📊 Summary plot saved to:")
    print(f"   {experiment_dir / 'lr_finder_summary.png'}")
    
    print("\n" + "="*80)
    print("✅ ROBUST LR FINDER COMPLETE!")
    print("="*80)
    print(f"\n🎯 RECOMMENDED LEARNING RATE: {geom_mean_lr:.4e}")
    print(f"\n💡 Update your config file:")
    print(f"   LEARNING_RATE = {geom_mean_lr:.4e}")
    print("="*80)


def create_summary_plot(suggested_lrs, geom_mean, median, output_dir):
    """Create a summary visualization of all runs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Distribution
    ax1 = axes[0]
    runs = np.arange(1, len(suggested_lrs) + 1)
    ax1.plot(runs, suggested_lrs, 'o-', markersize=10, linewidth=2, color='#3A86FF', label='Suggested LR')
    ax1.axhline(geom_mean, color='red', linestyle='--', linewidth=2, label=f'Geometric Mean: {geom_mean:.2e}')
    ax1.axhline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2e}')
    
    ax1.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Suggested Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_title('LR Finder Results Across Multiple Runs', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xticks(runs)
    
    # Plot 2: Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([suggested_lrs], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][0].set_alpha(0.7)
    
    # Add scatter points
    ax2.scatter(np.ones(len(suggested_lrs)), suggested_lrs, 
               color='darkblue', s=100, alpha=0.6, zorder=3)
    
    # Add mean and median lines
    ax2.axhline(geom_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Geo Mean: {geom_mean:.2e}')
    ax2.axhline(median, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median:.2e}')
    
    ax2.set_ylabel('Suggested Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Suggested LRs', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['All Runs'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best', fontsize=10)
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    • Mean: {np.mean(suggested_lrs):.2e}
    • Median: {median:.2e}
    • Geo Mean: {geom_mean:.2e}
    • Std: {np.std(suggested_lrs):.2e}
    • CV: {np.std(suggested_lrs)/np.mean(suggested_lrs):.1%}
    • Range: {np.max(suggested_lrs)/np.min(suggested_lrs):.2f}x
    """
    
    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_finder_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()


"""
FFCV Dataset Writer for ImageNet

This script converts PyTorch ImageFolder datasets to FFCV format for faster data loading.
Uses the repository's configuration system for consistent paths and settings.

FFCV (Fast Forward Computer Vision) is a library for creating highly optimized data loaders.
It can significantly speed up training by:
- Pre-processing images and storing them in a binary format
- Reducing I/O overhead during training
- Enabling efficient data augmentation pipelines

Usage:
    # Write training set using config paths
    python write_data.py --config local --split train
    
    # Write validation set using config paths
    python write_data.py --config local --split val
    
    # Custom output path and settings
    python write_data.py --config local --split train \\
        --write-path /path/to/output.beton \\
        --max-resolution 512 \\
        --jpeg-quality 95
    
    # Write subset for testing
    python write_data.py --config local --split train --subset 1000
"""

import argparse
from pathlib import Path
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

# Import repository's config system
from configs import get_config, list_configs


def write_ffcv_dataset(
    data_dir: Path,
    write_path: Path,
    split: str,
    max_resolution: int = 512,
    write_mode: str = 'smart',
    jpeg_quality: float = 90,
    num_workers: int = 16,
    chunk_size: int = 100,
    subset: int = -1,
    compress_probability: float = None
):
    """
    Convert ImageFolder dataset to FFCV format.
    
    Args:
        data_dir: Path to ImageFolder dataset (contains class subdirectories)
        write_path: Where to write the FFCV dataset (.beton file)
        split: Dataset split name ('train' or 'val') - for logging purposes
        max_resolution: Maximum image side length (larger images will be downsampled)
        write_mode: FFCV write mode ('raw', 'smart', or 'jpg')
            - 'raw': Store raw pixel values (largest, fastest to load)
            - 'smart': Automatically choose best format per image
            - 'jpg': Store as JPEG (smallest, some quality loss)
        jpeg_quality: JPEG compression quality (0-100)
        num_workers: Number of parallel workers for writing
        chunk_size: Number of samples per chunk (affects memory usage)
        subset: Number of images to use (-1 for all)
        compress_probability: Probability of compressing each image (None = auto)
    """
    print("=" * 70)
    print(f"🔄 Converting {split} dataset to FFCV format")
    print("=" * 70)
    print(f"📁 Input:  {data_dir}")
    print(f"💾 Output: {write_path}")
    print(f"⚙️  Settings:")
    print(f"   • Max resolution: {max_resolution}px")
    print(f"   • Write mode: {write_mode}")
    print(f"   • JPEG quality: {jpeg_quality}")
    print(f"   • Workers: {num_workers}")
    print(f"   • Chunk size: {chunk_size}")
    print("=" * 70)
    
    # Load dataset using torchvision's ImageFolder
    print(f"📊 Loading dataset from {data_dir}...")
    dataset = ImageFolder(root=str(data_dir))
    
    # Use subset if specified
    if subset > 0:
        print(f"⚠️  Using subset: {subset:,} / {len(dataset):,} images")
        dataset = Subset(dataset, range(subset))
    else:
        print(f"✅ Using full dataset: {len(dataset):,} images")
    
    # Create output directory if needed
    write_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure FFCV writer
    print("🔧 Configuring FFCV writer...")
    writer = DatasetWriter(
        fname=str(write_path),
        fields={
            'image': RGBImageField(
                write_mode=write_mode,
                compress_probability=compress_probability,
                jpeg_quality=jpeg_quality
            ),
            'label': IntField(),
        },
        num_workers=num_workers
    )
    
    # Write dataset
    print("✍️  Writing dataset to FFCV format...")
    print("   (This may take several minutes...)")
    writer.from_indexed_dataset(dataset, chunksize=chunk_size)
    
    # Summary
    file_size_mb = write_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print("✅ Dataset conversion complete!")
    print(f"📊 Output file: {write_path}")
    print(f"💾 File size: {file_size_mb:.1f} MB")
    print(f"🖼️  Images: {len(dataset):,}")
    print("=" * 70)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert ImageNet dataset to FFCV format for faster training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config to get dataset paths
  python write_data.py --config local --split train
  python write_data.py --config local --split val
  
  # Custom output path
  python write_data.py --config local --split train --write-path /path/to/train.beton
  
  # High quality settings for final training
  python write_data.py --config local --split train --max-resolution 512 --jpeg-quality 95
  
  # Quick test with subset
  python write_data.py --config local --split train --subset 1000
  
  # List available configs
  python write_data.py --list-configs
        """
    )
    
    # Configuration selection
    parser.add_argument(
        '--config',
        type=str,
        default='local',
        choices=['local', 'g5', 'p3'],
        help='Hardware configuration profile to use for dataset paths (default: local)'
    )
    
    # Dataset split
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val'],
        help='Dataset split to convert (train or val)'
    )
    
    # Output path (optional - defaults to config path + .beton)
    parser.add_argument(
        '--write-path',
        type=str,
        default=None,
        help='Output path for FFCV dataset (.beton file). If not specified, uses config directory + /ffcv/{split}.beton'
    )
    
    # FFCV settings
    parser.add_argument(
        '--max-resolution',
        type=int,
        default=512,
        help='Maximum image side length (default: 512)'
    )
    
    parser.add_argument(
        '--write-mode',
        type=str,
        default='smart',
        choices=['raw', 'smart', 'jpg'],
        help='FFCV write mode: raw (largest/fastest), smart (automatic), jpg (smallest) (default: smart)'
    )
    
    parser.add_argument(
        '--jpeg-quality',
        type=float,
        default=90,
        help='JPEG compression quality 0-100 (default: 90)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: 16)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of samples per chunk (default: 100)'
    )
    
    parser.add_argument(
        '--subset',
        type=int,
        default=-1,
        help='Number of images to use, -1 for all (default: -1)'
    )
    
    parser.add_argument(
        '--compress-probability',
        type=float,
        default=None,
        help='Probability of compressing each image (default: None = auto)'
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
    
    # Determine input data directory based on split
    if args.split == 'train':
        data_dir = config.train_img_dir
    else:  # val
        data_dir = config.val_img_dir
    
    # Determine output path
    if args.write_path:
        write_path = Path(args.write_path)
    else:
        # Default: store in config's project root under ffcv directory
        ffcv_dir = config.project_root / 'dataset' / 'ffcv'
        write_path = ffcv_dir / f'imagenet_{args.split}.beton'
    
    # Convert dataset
    write_ffcv_dataset(
        data_dir=data_dir,
        write_path=write_path,
        split=args.split,
        max_resolution=args.max_resolution,
        write_mode=args.write_mode,
        jpeg_quality=args.jpeg_quality,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        subset=args.subset,
        compress_probability=args.compress_probability
    )
    
    print(f"\n🎯 To use this FFCV dataset:")
    print("   1. Update your dataloader to use FFCV Loader instead of PyTorch DataLoader")
    print("   2. See FFCV documentation: https://ffcv.io/")
    print(f"   3. Dataset path: {write_path}")


if __name__ == '__main__':
    main()
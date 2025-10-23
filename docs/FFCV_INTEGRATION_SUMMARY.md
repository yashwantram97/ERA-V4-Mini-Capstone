# FFCV Integration Summary

This document summarizes the changes made to integrate FFCV (Fast Forward Computer Vision) with PyTorch Lightning for high-performance data loading.

## References
- [Lightning FFCV Documentation](https://lightning.ai/docs/pytorch/stable/data/alternatives.html#ffcv)
- [FFCV Documentation](https://docs.ffcv.io/making_dataloaders.html)
- [FFCV ImageNet Example](https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py)

## Key Changes

### 1. `src/utils/utils.py`

#### What Changed:
- **Updated `get_transforms()` function** to return FFCV pipelines instead of just torchvision transforms
- Returns tuple of `(image_pipeline, label_pipeline)` instead of single transform

#### Key Features:
- **FFCV Decoder**: Added `SimpleRGBImageDecoder()` at the start of the pipeline (required by FFCV)
- **Hybrid Approach**: Uses FFCV decoders + torchvision transforms
- **Individual Transforms**: Unpacks torchvision transforms from `Compose` since FFCV auto-converts `nn.Module` subclasses
- **Label Pipeline**: Properly configured with `IntDecoder()`, `ToTensor()`, and `Squeeze()`

#### Code Structure:
```python
def get_transforms(transform_type="train", mean=None, std=None, resolution=224):
    # Training transforms
    if transform_type == "train":
        transforms = [
            T.RandomResizedCrop(...),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),  # These are nn.Module subclasses
            T.ToTensor(),
            T.Normalize(...),
            T.RandomErasing(...)
        ]
    
    # FFCV pipelines
    image_pipeline = [SimpleRGBImageDecoder()] + transforms
    label_pipeline = [IntDecoder(), ToTensor(), Squeeze()]
    
    return image_pipeline, label_pipeline
```

### 2. `src/data_modules/imagenet_datamodule.py`

#### What Changed:
- **Replaced PyTorch DataLoader** with `ffcv.loader.Loader`
- **Removed ImageNetDataset** - FFCV uses `.beton` files directly
- **Updated parameter names**: `train_img_dir` → `train_beton_path`, `val_img_dir` → `val_beton_path`
- **Added FFCV-specific parameters**: `os_cache`, `quasi_random`, `drop_last`

#### Key Features:
- **FFCV Loader**: Uses `Loader` with pipelines instead of PyTorch `DataLoader`
- **Distributed Training**: Set `distributed=True` - FFCV handles DDP automatically
- **Ordering Options**:
  - Training: `RANDOM` or `QUASI_RANDOM` (for large datasets that don't fit in RAM)
  - Validation: `SEQUENTIAL` (no shuffling needed)
- **Pipeline Integration**: Passes `image_pipeline` and `label_pipeline` to Loader

#### Code Structure:
```python
def setup(self, stage: str = None):
    # Get FFCV pipelines
    train_image_pipeline, train_label_pipeline = get_transforms(...)
    valid_image_pipeline, valid_label_pipeline = get_transforms(...)
    
    # Create FFCV Loaders
    self.train_loader = Loader(
        self.train_beton_path,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        order=OrderOption.RANDOM,
        os_cache=self.os_cache,
        drop_last=self.drop_last,
        pipelines={
            'image': train_image_pipeline,
            'label': train_label_pipeline
        },
        distributed=True  # FFCV handles DDP
    )

def train_dataloader(self):
    return self.train_loader  # Return FFCV Loader directly
```

### 3. `src/models/resnet_module.py`

#### What Changed:
- **Removed `train_transforms` parameter** - no longer needed since transforms are in the dataloader
- **Removed `serialize_transforms` call** - FFCV pipelines can't be serialized the same way
- **Simplified `__init__`** - cleaner initialization without transform handling
- **Updated `forward()` method** - ensures channels_last format for FFCV compatibility

#### Key Features:
- **Channels-Last Format**: Explicitly converts input to `channels_last` in forward pass
- **FFCV Compatibility**: Works seamlessly with FFCV's data format
- **Cleaner Code**: Removed unnecessary transform serialization logic

#### Code Structure:
```python
def __init__(
    self,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    num_classes: int = 100,
    mixup_kwargs: dict = None  # Removed train_transforms
):
    # Simplified initialization
    ...

def forward(self, x):
    # Ensure channels_last format for FFCV compatibility
    x = x.to(memory_format=torch.channels_last)
    return self.model(x)
```

## How FFCV Works with Lightning

### Data Flow:
1. **Data Preparation**: Convert ImageNet to `.beton` format using `write_data.py`
2. **FFCV Loader**: Loads from `.beton` files with custom pipelines
3. **Pipelines**: 
   - Decode images with `SimpleRGBImageDecoder`
   - Apply torchvision transforms (auto-converted by FFCV)
   - Decode labels with `IntDecoder`
4. **Lightning Integration**: Pass FFCV Loader directly to Lightning Trainer

### Benefits:
- ✅ **Fast Data Loading**: FFCV is optimized for speed
- ✅ **GPU Decoding**: Can decode on GPU (not implemented yet, but possible)
- ✅ **Hybrid Transforms**: Can mix FFCV and torchvision transforms
- ✅ **DDP Support**: FFCV handles distributed training automatically
- ✅ **Memory Efficient**: Options for large datasets (QUASI_RANDOM, os_cache)

## Usage Example

```python
from src.data_modules.imagenet_datamodule import ImageNetDataModule
from src.models.resnet_module import ResnetLightningModule
import lightning as L

# Create DataModule with FFCV
data_module = ImageNetDataModule(
    train_beton_path="/path/to/train.beton",
    val_beton_path="/path/to/val.beton",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    batch_size=256,
    num_workers=8,
    initial_resolution=224,
    use_train_augs=True,
    os_cache=True,
    quasi_random=False
)

# Create model
model = ResnetLightningModule(
    learning_rate=0.1,
    weight_decay=1e-4,
    num_classes=100,
    mixup_kwargs={'mixup_alpha': 0.2, 'cutmix_alpha': 1.0}
)

# Train with Lightning
trainer = L.Trainer(
    max_epochs=100,
    devices=4,
    strategy='ddp',
    precision='16-mixed'
)

trainer.fit(model, data_module)
```

## Important Notes

### Lightning + FFCV Considerations:
1. **No `use_distributed_sampler`**: FFCV handles distributed training internally
2. **Channels-Last Format**: FFCV outputs in channels_last by default (optimal for performance)
3. **`.beton` Files Required**: Must pre-process data into FFCV format
4. **Device Placement**: Don't hardcode `ToDevice(0)` - let Lightning handle it or use `trainer.local_rank`

### FFCV Pipeline Rules:
1. **Must start with decoder**: Every pipeline needs a decoder first (e.g., `SimpleRGBImageDecoder()`)
2. **Individual transforms**: Don't use `T.Compose` - pass individual transforms in a list
3. **Auto-conversion**: Any `nn.Module` subclass is automatically converted by FFCV
4. **Label pipeline**: Must include `IntDecoder()` for integer labels

## Performance Tips

### For Maximum Speed:
1. **Use pure FFCV transforms** instead of torchvision (faster, GPU-accelerated)
2. **Enable GPU decoding** with `ToDevice(trainer.local_rank)` in pipelines
3. **Use `os_cache=True`** to let OS manage caching
4. **Use `QUASI_RANDOM`** for datasets that don't fit in RAM
5. **Increase `num_workers`** to match CPU cores

### Trade-offs:
- **Pure FFCV**: Fastest, but limited augmentation options
- **Hybrid FFCV + Torchvision**: More augmentations (TrivialAugmentWide), slightly slower
- **Current Implementation**: Uses hybrid approach for best of both worlds

## Next Steps

### Optional Optimizations:
1. **GPU Decoding**: Add `ToDevice(trainer.local_rank)` to pipelines for GPU-accelerated decoding
2. **Pure FFCV Transforms**: Replace torchvision transforms with FFCV equivalents for maximum speed
3. **Custom Transforms**: Implement custom FFCV transforms for specialized augmentations
4. **Profiling**: Use FFCV's bottleneck doctor to identify performance bottlenecks

### To Enable GPU Decoding:
```python
# In get_transforms(), add device parameter
image_pipeline = [
    SimpleRGBImageDecoder(),
    # ... transforms ...
    ToDevice(device, non_blocking=True)  # Move to GPU
]

# In DataModule, pass trainer.local_rank
train_image_pipeline, train_label_pipeline = get_transforms(
    transform_type="train",
    device=self.trainer.local_rank if hasattr(self, 'trainer') else None
)
```

## Conclusion

The codebase now uses FFCV for fast data loading while maintaining full compatibility with PyTorch Lightning. The hybrid approach (FFCV decoders + torchvision transforms) provides a good balance between speed and augmentation flexibility.

Key benefits:
- ✅ Faster data loading than PyTorch DataLoader
- ✅ Full Lightning integration (DDP, callbacks, logging)
- ✅ Flexible augmentation pipeline
- ✅ Memory-efficient options for large datasets
- ✅ Easy to switch between pure FFCV and hybrid approaches


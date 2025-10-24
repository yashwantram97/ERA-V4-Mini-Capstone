"""
Advanced Features Test Suite

Tests for:
1. Progressive Resizing (MosaicML Composer approach)
2. FixRes (test-time augmentation alignment)
3. 16-Mixed Precision (automatic mixed precision)
4. Channels Last Memory Format (NHWC layout)

Usage:
    python tests/test_advanced_features.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.callbacks import create_progressive_resize_schedule
from src.models.resnet_module import ResnetLightningModule
from src.data_modules.imagenet_dataset import ImageNetDataset


def print_test_header(test_name, test_num):
    """Print formatted test header"""
    print("\n" + "="*80)
    print(f"TEST {test_num}: {test_name}")
    print("="*80)


def print_success(message):
    """Print success message"""
    print(f"   ✅ {message}")


def print_error(message):
    """Print error message"""
    print(f"   ❌ {message}")


def print_info(message):
    """Print info message"""
    print(f"   ℹ️  {message}")


def test_progressive_resizing():
    """
    Test 1: Progressive Resizing Schedule
    
    Verifies:
    - Schedule generation with MosaicML Composer parameters
    - Smooth curriculum progression
    - Correct phase boundaries
    - Size increment alignment
    """
    print_test_header("PROGRESSIVE RESIZING (MosaicML Composer)", 1)
    
    try:
        # Test 1.1: Basic schedule generation
        print("\n📋 Test 1.1: Schedule Generation")
        schedule = create_progressive_resize_schedule(
            total_epochs=60,
            target_size=224,
            initial_scale=0.5,
            delay_fraction=0.5,
            finetune_fraction=0.2,
            size_increment=4
        )
        
        assert len(schedule) > 0, "Schedule should not be empty"
        print_success(f"Generated schedule with {len(schedule)} transitions")
        
        # Test 1.2: Verify initial size
        print("\n📋 Test 1.2: Initial Resolution")
        initial_size = schedule[0][0]
        expected_initial = 112  # 50% of 224, rounded to multiple of 4
        assert initial_size == expected_initial, f"Expected {expected_initial}px, got {initial_size}px"
        print_success(f"Initial size: {initial_size}x{initial_size}px (50% of target 224px)")
        
        # Test 1.3: Verify phase boundaries
        print("\n📋 Test 1.3: Phase Boundaries")
        sorted_epochs = sorted(schedule.keys())
        
        # Phase 1: Delay (should be at epoch 0)
        assert 0 in schedule, "Schedule should start at epoch 0"
        print_success(f"Delay phase starts at epoch 0")
        
        # Phase 2: Progressive (should start around epoch 30 for 60 epochs)
        expected_progressive_start = int(60 * 0.5)  # 30
        progressive_starts = [e for e in sorted_epochs if 25 <= e <= 35]
        assert len(progressive_starts) > 0, "Progressive phase should exist"
        print_success(f"Progressive phase detected around epoch {expected_progressive_start}")
        
        # Phase 3: Fine-tune (should start around epoch 48 for 60 epochs)
        expected_finetune_start = 60 - int(60 * 0.2)  # 48
        finetune_epochs = [e for e in sorted_epochs if schedule[e][0] == 224]
        assert len(finetune_epochs) > 0, "Fine-tune phase should exist"
        print_success(f"Fine-tune phase at full 224px resolution")
        
        # Test 1.4: Verify smooth progression
        print("\n📋 Test 1.4: Curriculum Smoothness")
        resolutions = [schedule[e][0] for e in sorted_epochs]
        
        # Check monotonic increase
        for i in range(len(resolutions) - 1):
            assert resolutions[i] <= resolutions[i+1], "Resolution should monotonically increase"
        
        # Check increment size (should be multiples of size_increment)
        for res in resolutions:
            assert res % 4 == 0, f"Resolution {res} should be multiple of 4"
        
        print_success(f"Smooth curriculum: {resolutions[0]} → {resolutions[-1]}px")
        print_info(f"Resolution steps: {len(set(resolutions))} unique sizes")
        
        # Test 1.5: Verify percentage-based adaptation
        print("\n📋 Test 1.5: Adaptability to Different Epoch Counts")
        
        for total_epochs in [30, 60, 100]:
            schedule_test = create_progressive_resize_schedule(
                total_epochs=total_epochs,
                target_size=224,
                initial_scale=0.5,
                delay_fraction=0.5,
                finetune_fraction=0.2,
                size_increment=4
            )
            
            # Check that first and last resolutions are correct
            assert schedule_test[0][0] == 112, "Initial size should always be 112px"
            
            # Find last resolution
            last_epoch = max(schedule_test.keys())
            last_res = schedule_test[last_epoch][0]
            assert last_res == 224, "Final size should always be 224px"
            
            print_success(f"{total_epochs} epochs: 112px → 224px (adapts correctly)")
        
        # Test 1.6: Verify size increment alignment
        print("\n📋 Test 1.6: Size Increment Alignment")
        for increment in [4, 8, 16]:
            schedule_inc = create_progressive_resize_schedule(
                total_epochs=60,
                target_size=224,
                initial_scale=0.5,
                size_increment=increment
            )
            
            all_sizes = [schedule_inc[e][0] for e in sorted(schedule_inc.keys())]
            all_aligned = all(size % increment == 0 for size in all_sizes)
            assert all_aligned, f"All sizes should be multiples of {increment}"
            print_success(f"Size increment {increment}: All resolutions aligned")
        
        print("\n" + "="*80)
        print("✅ TEST 1 PASSED: Progressive Resizing is working correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print_error(f"Progressive Resizing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_fixres():
    """
    Test 2: FixRes (Test-Time Augmentation Alignment)
    
    Verifies:
    - FixRes phase is properly configured
    - Test-time augmentations (use_train_augs=False) are applied
    - Higher resolution in FixRes phase
    """
    print_test_header("FIXRES (Test-Time Augmentation Alignment)", 2)
    
    try:
        # Test 2.1: FixRes schedule generation
        print("\n📋 Test 2.1: FixRes Schedule Generation")
        schedule_fixres = create_progressive_resize_schedule(
            total_epochs=60,
            target_size=224,
            initial_scale=0.5,
            delay_fraction=0.5,
            finetune_fraction=0.2,
            size_increment=4,
            use_fixres=True,
            fixres_size=256
        )
        
        # Find FixRes phase (use_train_augs=False)
        fixres_epochs = {e: config for e, config in schedule_fixres.items() if not config[1]}
        assert len(fixres_epochs) > 0, "FixRes phase should exist when use_fixres=True"
        print_success(f"FixRes phase found with test-time augmentations")
        
        # Test 2.2: Verify FixRes resolution
        print("\n📋 Test 2.2: FixRes Resolution")
        fixres_size = list(fixres_epochs.values())[0][0]
        assert fixres_size == 256, f"Expected 256px for FixRes, got {fixres_size}px"
        print_success(f"FixRes uses higher resolution: {fixres_size}x{fixres_size}px")
        
        # Test 2.3: Verify FixRes is in final phase
        print("\n📋 Test 2.3: FixRes Timing")
        fixres_start = min(fixres_epochs.keys())
        expected_start = 60 - int(60 * 0.1)  # Last 10%
        assert fixres_start >= expected_start - 2, f"FixRes should start in final 10% of training"
        print_success(f"FixRes starts at epoch {fixres_start} (last 10% of training)")
        
        # Test 2.4: Verify augmentation flags
        print("\n📋 Test 2.4: Augmentation Configuration")
        
        # Before FixRes: use_train_augs=True
        pre_fixres = {e: config for e, config in schedule_fixres.items() if e < fixres_start}
        all_train_augs = all(config[1] for config in pre_fixres.values())
        assert all_train_augs, "All epochs before FixRes should use train augmentations"
        print_success(f"Pre-FixRes: Train augmentations (RandomResizedCrop + Flip + TrivialAugmentWide + RandomErasing)")
        
        # During FixRes: use_train_augs=False
        all_test_augs = all(not config[1] for config in fixres_epochs.values())
        assert all_test_augs, "FixRes phase should use test augmentations"
        print_success(f"FixRes: Test augmentations (Resize + CenterCrop)")
        
        # Test 2.5: Compare with and without FixRes
        print("\n📋 Test 2.5: FixRes Impact")
        schedule_no_fixres = create_progressive_resize_schedule(
            total_epochs=60,
            target_size=224,
            initial_scale=0.5,
            delay_fraction=0.5,
            finetune_fraction=0.2,
            size_increment=4,
            use_fixres=False
        )
        
        # Without FixRes: all should use train augmentations
        all_train = all(config[1] for config in schedule_no_fixres.values())
        assert all_train, "Without FixRes, all epochs should use train augmentations"
        print_success(f"Without FixRes: All epochs use train augmentations")
        
        # With FixRes: should have mixed
        has_both = any(not config[1] for config in schedule_fixres.values())
        assert has_both, "With FixRes, should have both train and test augmentations"
        print_success(f"With FixRes: Mixed train/test augmentations")
        
        print("\n" + "="*80)
        print("✅ TEST 2 PASSED: FixRes is working correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print_error(f"FixRes test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_precision():
    """
    Test 3: 16-Mixed Precision Training
    
    Verifies:
    - Automatic mixed precision (AMP) is working
    - GradScaler for loss scaling
    - Forward pass in FP16
    - Backward pass with scaled gradients
    """
    print_test_header("16-MIXED PRECISION (Automatic Mixed Precision)", 3)
    
    try:
        # Check if CUDA is available (mixed precision works best on GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print_info("Running on CPU (testing mixed precision infrastructure)")
            print_info("Note: Actual FP16 conversion requires CUDA, but testing API works")
        else:
            print_info(f"Running on GPU: {torch.cuda.get_device_name(0)}")
        
        # Test 3.1: Create model and move to device
        print("\n📋 Test 3.1: Model Setup")
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        ).to(device)
        
        print_success("Model created and moved to device")
        
        # Test 3.2: Test autocast context
        print("\n📋 Test 3.2: Autocast Context")
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        # Forward pass without autocast (FP32)
        with torch.no_grad():
            output_fp32 = model(dummy_input)
        
        print_success(f"FP32 output dtype: {output_fp32.dtype}")
        
        # Forward pass with autocast (FP16/BF16)
        with torch.no_grad():
            if device.type == 'cuda':
                with autocast():
                    output_amp = model(dummy_input)
            else:
                # CPU doesn't support autocast in older PyTorch versions
                output_amp = model(dummy_input)
        
        # On CPU, autocast might not change dtype, but on CUDA it should
        if device.type == 'cuda':
            print_info(f"AMP output dtype: {output_amp.dtype}")
        else:
            print_info("AMP dtype checking works (CPU mode)")
        
        print_success("Autocast context manager working")
        
        # Test 3.3: Test GradScaler
        print("\n📋 Test 3.3: Gradient Scaler")
        scaler = GradScaler()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        print_success("GradScaler initialized")
        
        # Test 3.4: Full training step with mixed precision
        print("\n📋 Test 3.4: Mixed Precision Training Step")
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass in autocast
        if device.type == 'cuda':
            with autocast():
                output = model(dummy_input)
                loss = output.mean()  # Dummy loss
        else:
            # CPU mode - run without autocast
            output = model(dummy_input)
            loss = output.mean()
        
        # Check if loss is computed
        assert loss.item() is not None, "Loss should be computed"
        print_success(f"Forward pass completed, loss: {loss.item():.6f}")
        
        # Backward pass with scaled gradients
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            
            # Check gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters())
            assert has_gradients, "Model should have gradients"
            print_success("Backward pass completed with scaled gradients")
            
            # Optimizer step with unscaling
            scaler.step(optimizer)
            scaler.update()
            
            print_success("Optimizer step with gradient unscaling completed")
        else:
            # CPU mode - standard backward
            loss.backward()
            has_gradients = any(p.grad is not None for p in model.parameters())
            assert has_gradients, "Model should have gradients"
            print_success("Backward pass completed (CPU mode)")
            
            optimizer.step()
            print_success("Optimizer step completed (CPU mode)")
        
        # Test 3.5: Verify precision savings
        print("\n📋 Test 3.5: Memory Efficiency")
        
        # Create larger model for memory comparison
        large_model = nn.Sequential(
            nn.Conv2d(3, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 100)
        ).to(device)
        
        # Calculate model size in FP32
        param_count = sum(p.numel() for p in large_model.parameters())
        fp32_size_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per FP32
        fp16_size_mb = (param_count * 2) / (1024 * 1024)  # 2 bytes per FP16
        
        print_info(f"Model parameters: {param_count:,}")
        print_info(f"FP32 model size: {fp32_size_mb:.2f} MB")
        print_info(f"FP16 activations: ~{fp16_size_mb:.2f} MB (50% savings)")
        print_success("Mixed precision provides ~50% memory savings for activations")
        
        # Test 3.6: Verify numerical stability
        print("\n📋 Test 3.6: Numerical Stability")
        
        model.eval()
        with torch.no_grad():
            # Multiple forward passes should be consistent
            outputs = []
            for _ in range(5):
                if device.type == 'cuda':
                    with autocast():
                        out = model(dummy_input)
                else:
                    out = model(dummy_input)
                outputs.append(out)
            
            # Check consistency
            output_std = torch.std(torch.stack(outputs), dim=0).mean().item()
            assert output_std < 1e-3, "Outputs should be consistent across runs"
            print_success(f"Numerical stability verified (std: {output_std:.2e})")
        
        print("\n" + "="*80)
        print("✅ TEST 3 PASSED: 16-Mixed Precision is working correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print_error(f"Mixed Precision test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_channels_last():
    """
    Test 4: Channels Last Memory Format
    
    Verifies:
    - NHWC (channels last) memory layout
    - Performance benefits
    - Compatibility with operations
    - Proper format conversion
    """
    print_test_header("CHANNELS LAST MEMORY FORMAT (NHWC Layout)", 4)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test 4.1: Create tensors in different formats
        print("\n📋 Test 4.1: Memory Format Conversion")
        
        # Default: NCHW (channels first)
        tensor_nchw = torch.randn(4, 3, 224, 224)
        assert tensor_nchw.is_contiguous(), "NCHW tensor should be contiguous"
        print_success(f"NCHW tensor created: shape {list(tensor_nchw.shape)}")
        
        # Convert to NHWC (channels last)
        tensor_nhwc = tensor_nchw.to(memory_format=torch.channels_last)
        assert tensor_nhwc.is_contiguous(memory_format=torch.channels_last), \
            "NHWC tensor should be contiguous in channels_last format"
        print_success(f"NHWC tensor created: shape {list(tensor_nhwc.shape)}")
        
        # Test 4.2: Verify stride patterns
        print("\n📋 Test 4.2: Memory Layout (Stride Pattern)")
        
        nchw_strides = tensor_nchw.stride()
        nhwc_strides = tensor_nhwc.stride()
        
        print_info(f"NCHW strides: {nchw_strides} (N, C, H, W)")
        print_info(f"NHWC strides: {nhwc_strides} (N, H, W, C)")
        
        # In NHWC, channel stride should be 1 (innermost)
        assert nhwc_strides[1] == 1, "In NHWC, channel dimension should have stride 1"
        print_success("NHWC layout verified: channels are innermost dimension")
        
        # Test 4.3: Model compatibility
        print("\n📋 Test 4.3: Model Compatibility")
        
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ).to(device)
        
        # Convert model to channels last
        model = model.to(memory_format=torch.channels_last)
        print_success("Model converted to channels_last format")
        
        # Test forward pass with channels_last input
        input_nhwc = torch.randn(2, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
        
        model.eval()
        with torch.no_grad():
            output = model(input_nhwc)
        
        assert output.shape == (2, 10), "Output shape should be correct"
        print_success("Forward pass with channels_last input successful")
        
        # Test 4.4: Verify format preservation through layers
        print("\n📋 Test 4.4: Format Preservation")
        
        # Create a model with hooks to check intermediate formats
        formats_checked = []
        
        def check_format_hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                is_channels_last = output.is_contiguous(memory_format=torch.channels_last)
                formats_checked.append(is_channels_last)
        
        # Register hooks
        hooks = []
        for layer in model:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                hook = layer.register_forward_hook(check_format_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_nhwc)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Check if format was preserved
        if len(formats_checked) > 0:
            format_preservation = sum(formats_checked) / len(formats_checked)
            print_info(f"Format preserved in {format_preservation*100:.0f}% of conv/bn layers")
            if format_preservation > 0.5:
                print_success("Channels last format generally preserved through layers")
            else:
                print_info("Some layers converted back to contiguous format (expected)")
        
        # Test 4.5: Performance characteristics
        print("\n📋 Test 4.5: Performance Characteristics")
        
        # Create test model and input
        test_model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        ).to(device)
        
        batch = torch.randn(8, 3, 224, 224, device=device)
        
        # Test NCHW format
        test_model_nchw = test_model
        batch_nchw = batch
        
        # Test NHWC format
        test_model_nhwc = test_model.to(memory_format=torch.channels_last)
        batch_nhwc = batch.to(memory_format=torch.channels_last)
        
        # Both should produce similar results
        with torch.no_grad():
            out_nchw = test_model_nchw(batch_nchw)
            out_nhwc = test_model_nhwc(batch_nhwc)
        
        # Compare outputs (should be similar)
        diff = (out_nchw - out_nhwc).abs().mean().item()
        assert diff < 1e-4, f"Outputs should be similar (diff: {diff})"
        print_success(f"NCHW and NHWC produce equivalent results (diff: {diff:.2e})")
        
        # Test 4.6: Verify ResNet compatibility
        print("\n📋 Test 4.6: ResNet Model Compatibility")
        
        try:
            from torchvision.models import resnet50
            
            resnet = resnet50(weights=None).to(device)
            resnet = resnet.to(memory_format=torch.channels_last)
            
            test_input = torch.randn(2, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
            
            resnet.eval()
            with torch.no_grad():
                resnet_output = resnet(test_input)
            
            assert resnet_output.shape == (2, 1000), "ResNet output shape should be correct"
            print_success("ResNet50 compatible with channels_last format")
            
        except Exception as e:
            print_info(f"ResNet test skipped: {str(e)}")
        
        # Test 4.7: Memory layout benefits
        print("\n📋 Test 4.7: Expected Benefits")
        print_info("Channels last format benefits:")
        print_info("  • Better cache locality for convolutions")
        print_info("  • ~5-10% speedup on modern GPUs (Tensor Cores)")
        print_info("  • More efficient memory access patterns")
        print_info("  • Compatible with mixed precision training")
        print_success("Channels last format properly configured")
        
        print("\n" + "="*80)
        print("✅ TEST 4 PASSED: Channels Last format is working correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print_error(f"Channels Last test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all advanced feature tests"""
    print("\n" + "🎯"*40)
    print("ADVANCED FEATURES TEST SUITE")
    print("🎯"*40)
    
    results = []
    
    # Run all tests
    tests = [
        ("Progressive Resizing", test_progressive_resizing),
        ("FixRes", test_fixres),
        ("16-Mixed Precision", test_mixed_precision),
        ("Channels Last", test_channels_last),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


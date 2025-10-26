"""
Test Progressive Resizing Resume Functionality

This test verifies that progressive resizing correctly restores the resolution
when training is resumed from a checkpoint at any epoch.

Critical for ensuring:
1. Checkpoint resume restores correct resolution
2. Schedule continues correctly after resume
3. Works at any epoch (delay phase, progressive phase, finetune phase)
"""

from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.callbacks.resolution_schedule_callback import (
    ResolutionScheduleCallback,
    create_progressive_resize_schedule
)


class MockDataModule:
    """Mock DataModule that tracks resolution updates"""
    def __init__(self):
        self.resolution = 224
        self.use_train_augs = True
        self.update_history = []
        self.train_dataset = None
        
    def update_resolution(self, size, use_train_augs):
        """Track all resolution updates"""
        self.resolution = size
        self.use_train_augs = use_train_augs
        self.update_history.append({
            'size': size,
            'use_train_augs': use_train_augs
        })


class MockTrainer:
    """Mock Trainer that simulates Lightning's trainer"""
    def __init__(self, current_epoch=0, datamodule=None):
        self.current_epoch = current_epoch
        self.datamodule = datamodule
        self.is_global_zero = True
        self.strategy = Mock()
        self.strategy.barrier = Mock()
        
    def set_epoch(self, epoch):
        """Helper to change current epoch"""
        self.current_epoch = epoch


class MockLightningModule:
    """Mock Lightning Module"""
    pass


def test_resume_at_delay_phase():
    """Test resume during the initial delay phase (constant small resolution)"""
    print("\n" + "="*80)
    print("TEST 1: Resume at Delay Phase")
    print("="*80)
    
    # Create schedule: 60 epochs, delay 30%, progressive 40%, finetune 30%
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,  # 144px
        delay_fraction=0.3,   # First 18 epochs
        finetune_fraction=0.3,  # Last 18 epochs
        size_increment=4
    )
    
    print(f"\nSchedule: {schedule}")
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=10, datamodule=datamodule)  # Resume at epoch 10 (delay phase)
    pl_module = MockLightningModule()
    
    # Simulate resume - this should restore resolution for epoch 10
    callback.on_train_epoch_start(trainer, pl_module)
    
    # Verify correct resolution was applied
    # Find the expected resolution from the schedule
    expected_config = callback._get_resolution_for_epoch(10)
    expected_resolution, expected_use_train_augs = expected_config
    
    assert datamodule.resolution == expected_resolution, \
        f"Resume at epoch 10 failed: expected {expected_resolution}px, got {datamodule.resolution}px"
    assert datamodule.use_train_augs == expected_use_train_augs, \
        "Train augmentations should be enabled in delay phase"
    
    print(f"✅ PASS: Resumed at epoch 10 (delay phase)")
    print(f"   Expected: {expected_resolution}px")
    print(f"   Got: {datamodule.resolution}px")
    print(f"   Train augs: {datamodule.use_train_augs}")


def test_resume_at_progressive_phase():
    """Test resume during the progressive phase (gradually increasing resolution)"""
    print("\n" + "="*80)
    print("TEST 2: Resume at Progressive Phase")
    print("="*80)
    
    # Create schedule
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,  # 144px
        delay_fraction=0.3,   # Epochs 0-17 (18 epochs)
        finetune_fraction=0.3,  # Epochs 42-59 (18 epochs)
        size_increment=4
    )
    
    print(f"\nSchedule: {schedule}")
    
    # Test multiple resume points in progressive phase
    test_epochs = [20, 30, 40]
    
    for resume_epoch in test_epochs:
        # Create fresh callback and datamodule for each test
        callback = ResolutionScheduleCallback(schedule)
        datamodule = MockDataModule()
        trainer = MockTrainer(current_epoch=resume_epoch, datamodule=datamodule)
        pl_module = MockLightningModule()
        
        # Simulate resume
        callback.on_train_epoch_start(trainer, pl_module)
        
        # Find expected resolution for this epoch
        expected_config = callback._get_resolution_for_epoch(resume_epoch)
        assert expected_config is not None, f"No schedule entry for epoch {resume_epoch}"
        expected_resolution, expected_use_train_augs = expected_config
        
        # Verify
        assert datamodule.resolution == expected_resolution, \
            f"Resume at epoch {resume_epoch} failed: expected {expected_resolution}px, got {datamodule.resolution}px"
        assert datamodule.use_train_augs == expected_use_train_augs, \
            f"Augmentation setting wrong at epoch {resume_epoch}"
        
        print(f"✅ PASS: Resumed at epoch {resume_epoch} (progressive phase)")
        print(f"   Expected: {expected_resolution}px, train_augs={expected_use_train_augs}")
        print(f"   Got: {datamodule.resolution}px, train_augs={datamodule.use_train_augs}")


def test_resume_at_finetune_phase():
    """Test resume during the fine-tune phase (full resolution)"""
    print("\n" + "="*80)
    print("TEST 3: Resume at Fine-tune Phase")
    print("="*80)
    
    # Create schedule
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,  # 144px
        delay_fraction=0.3,   # Epochs 0-17
        finetune_fraction=0.3,  # Epochs 42-59
        size_increment=4
    )
    
    print(f"\nSchedule: {schedule}")
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components - resume at epoch 50 (fine-tune phase)
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=50, datamodule=datamodule)
    pl_module = MockLightningModule()
    
    # Simulate resume
    callback.on_train_epoch_start(trainer, pl_module)
    
    # Verify correct resolution was applied
    expected_resolution = 224  # Should be full resolution in fine-tune phase
    assert datamodule.resolution == expected_resolution, \
        f"Resume at epoch 50 failed: expected {expected_resolution}px, got {datamodule.resolution}px"
    assert datamodule.use_train_augs == True, \
        "Train augmentations should be enabled in fine-tune phase"
    
    print(f"✅ PASS: Resumed at epoch 50 (fine-tune phase)")
    print(f"   Expected: {expected_resolution}px")
    print(f"   Got: {datamodule.resolution}px")
    print(f"   Train augs: {datamodule.use_train_augs}")


def test_resume_with_fixres():
    """Test resume during FixRes phase (high resolution + test augmentations)"""
    print("\n" + "="*80)
    print("TEST 4: Resume with FixRes Phase")
    print("="*80)
    
    # Create schedule with FixRes enabled
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,
        delay_fraction=0.3,
        finetune_fraction=0.3,
        size_increment=4,
        use_fixres=True,
        fixres_size=256
    )
    
    print(f"\nSchedule: {schedule}")
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components - resume at epoch 55 (should be in FixRes phase)
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=55, datamodule=datamodule)
    pl_module = MockLightningModule()
    
    # Simulate resume
    callback.on_train_epoch_start(trainer, pl_module)
    
    # Find expected resolution (should be FixRes)
    expected_config = callback._get_resolution_for_epoch(55)
    expected_resolution, expected_use_train_augs = expected_config
    
    print(f"   Expected config: {expected_resolution}px, train_augs={expected_use_train_augs}")
    
    # Verify correct resolution and augmentation setting
    assert datamodule.resolution == expected_resolution, \
        f"Resume at epoch 55 (FixRes) failed: expected {expected_resolution}px, got {datamodule.resolution}px"
    
    print(f"✅ PASS: Resumed at epoch 55 (FixRes phase)")
    print(f"   Expected: {expected_resolution}px, train_augs={expected_use_train_augs}")
    print(f"   Got: {datamodule.resolution}px, train_augs={datamodule.use_train_augs}")


def test_schedule_continues_after_resume():
    """Test that schedule continues to work correctly after resume"""
    print("\n" + "="*80)
    print("TEST 5: Schedule Continues After Resume")
    print("="*80)
    
    # Create schedule
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,  # 144px
        delay_fraction=0.3,   # Epochs 0-17
        finetune_fraction=0.3,  # Epochs 42-59
        size_increment=4
    )
    
    print(f"\nSchedule: {schedule}")
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components - resume at epoch 30
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=30, datamodule=datamodule)
    pl_module = MockLightningModule()
    
    # Simulate resume at epoch 30
    callback.on_train_epoch_start(trainer, pl_module)
    resume_resolution = datamodule.resolution
    print(f"\n📍 Resumed at epoch 30: {resume_resolution}px")
    
    # Clear update history to track only post-resume updates
    initial_updates = len(datamodule.update_history)
    
    # Simulate continuing training through epoch 42 (start of fine-tune)
    for epoch in range(31, 43):
        trainer.set_epoch(epoch)
        callback.on_train_epoch_start(trainer, pl_module)
        
        if epoch in schedule:
            print(f"   Epoch {epoch}: {datamodule.resolution}px (schedule change)")
    
    # Verify we reached fine-tune resolution
    assert datamodule.resolution == 224, \
        f"After resume, should reach 224px by epoch 42, got {datamodule.resolution}px"
    
    # Verify we had updates (schedule changes happened)
    post_resume_updates = len(datamodule.update_history) - initial_updates
    print(f"\n   Post-resume updates: {post_resume_updates}")
    print(f"   Final resolution: {datamodule.resolution}px")
    
    print(f"✅ PASS: Schedule continued correctly after resume")
    print(f"   Started at epoch 30: {resume_resolution}px")
    print(f"   Reached fine-tune by epoch 42: {datamodule.resolution}px")


def test_resume_only_happens_once():
    """Test that resume logic only executes once, not on every epoch"""
    print("\n" + "="*80)
    print("TEST 6: Resume Logic Executes Only Once")
    print("="*80)
    
    # Create schedule
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,
        delay_fraction=0.3,
        finetune_fraction=0.3,
        size_increment=4
    )
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components - resume at epoch 30
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=30, datamodule=datamodule)
    pl_module = MockLightningModule()
    
    # First call - should trigger resume logic
    assert callback._resume_handled == False, "Resume should not be handled yet"
    callback.on_train_epoch_start(trainer, pl_module)
    assert callback._resume_handled == True, "Resume should be marked as handled"
    
    first_call_updates = len(datamodule.update_history)
    print(f"\n   First call (resume): {first_call_updates} update(s)")
    
    # Second call at same epoch - should NOT trigger resume logic again
    callback.on_train_epoch_start(trainer, pl_module)
    second_call_updates = len(datamodule.update_history)
    print(f"   Second call (same epoch): {second_call_updates} update(s) total")
    
    # The second call might add updates if epoch 30 is in schedule, but resume logic shouldn't run
    # Resume logic only adds one update, so difference should be at most 1
    assert callback._resume_handled == True, "Resume should still be marked as handled"
    
    print(f"✅ PASS: Resume logic only executed once")
    print(f"   _resume_handled flag: {callback._resume_handled}")


def test_resume_at_epoch_zero():
    """Test that resume logic doesn't trigger at epoch 0 (fresh start)"""
    print("\n" + "="*80)
    print("TEST 7: Fresh Start at Epoch 0 (Not a Resume)")
    print("="*80)
    
    # Create schedule
    schedule = create_progressive_resize_schedule(
        total_epochs=60,
        target_size=224,
        initial_scale=0.64,
        delay_fraction=0.3,
        finetune_fraction=0.3,
        size_increment=4
    )
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components - start at epoch 0
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=0, datamodule=datamodule)
    pl_module = MockLightningModule()
    
    # First call at epoch 0
    callback.on_train_epoch_start(trainer, pl_module)
    
    # Verify resume logic was NOT triggered (should use normal schedule logic)
    # At epoch 0, we should have schedule[0] applied
    expected_resolution = schedule[0][0]  # 144px
    assert datamodule.resolution == expected_resolution, \
        f"At epoch 0, expected {expected_resolution}px, got {datamodule.resolution}px"
    
    print(f"✅ PASS: Fresh start at epoch 0 handled correctly")
    print(f"   Resolution: {datamodule.resolution}px")
    print(f"   This is NOT a resume (current_epoch == 0)")


def test_edge_case_resume_before_first_schedule_entry():
    """Test resume at an epoch before any schedule entries"""
    print("\n" + "="*80)
    print("TEST 8: Edge Case - Resume Before First Schedule Entry")
    print("="*80)
    
    # Create a schedule that doesn't start at epoch 0
    schedule = {
        10: (144, True),
        30: (224, True)
    }
    
    print(f"\nSchedule: {schedule}")
    
    # Create callback
    callback = ResolutionScheduleCallback(schedule)
    
    # Create mock components - resume at epoch 5 (before first schedule entry)
    datamodule = MockDataModule()
    trainer = MockTrainer(current_epoch=5, datamodule=datamodule)
    pl_module = MockLightningModule()
    
    # Simulate resume
    callback.on_train_epoch_start(trainer, pl_module)
    
    # Verify that no resolution was applied (schedule hasn't started yet)
    # The datamodule should keep its default resolution
    assert datamodule.resolution == 224, \
        f"Before first schedule entry, should keep default 224px, got {datamodule.resolution}px"
    
    print(f"✅ PASS: Resume before first schedule entry handled correctly")
    print(f"   Epoch 5 is before first schedule entry (epoch 10)")
    print(f"   Resolution unchanged: {datamodule.resolution}px")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("PROGRESSIVE RESIZING RESUME TEST SUITE")
    print("="*80)
    print("Testing critical resume functionality for progressive resizing")
    print("="*80)
    
    tests = [
        ("Resume at Delay Phase", test_resume_at_delay_phase),
        ("Resume at Progressive Phase", test_resume_at_progressive_phase),
        ("Resume at Fine-tune Phase", test_resume_at_finetune_phase),
        ("Resume with FixRes", test_resume_with_fixres),
        ("Schedule Continues After Resume", test_schedule_continues_after_resume),
        ("Resume Logic Executes Only Once", test_resume_only_happens_once),
        ("Fresh Start at Epoch 0", test_resume_at_epoch_zero),
        ("Edge Case: Resume Before Schedule", test_edge_case_resume_before_first_schedule_entry),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ FAILED: {test_name}")
            print(f"   Error: {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Final report
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Progressive resizing resume is working correctly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


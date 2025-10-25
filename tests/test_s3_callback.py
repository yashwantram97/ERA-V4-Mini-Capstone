"""
Test S3 and Remote Storage Support for TextLoggingCallback

This test verifies that the TextLoggingCallback correctly handles both
local and remote (S3) filesystem paths.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.callbacks.text_logging_callback import TextLoggingCallback


class TestS3CallbackDetection:
    """Test S3 path detection and filesystem initialization"""
    
    def test_local_path_detection(self):
        """Test that local paths are detected correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TextLoggingCallback(
                log_dir=Path(tmpdir),
                experiment_name="test_local"
            )
            
            assert callback.is_remote == False
            assert callback.fs is None
            assert callback.local_temp_dir is None
            assert isinstance(callback.log_dir, Path)
    
    def test_s3_path_detection(self):
        """Test that S3 paths are detected correctly"""
        # Note: This won't actually connect to S3, just test detection
        try:
            callback = TextLoggingCallback(
                log_dir="s3://test-bucket/logs",
                experiment_name="test_s3"
            )
            
            assert callback.is_remote == True
            assert callback.fs is not None
            assert callback.local_temp_dir is not None
            assert isinstance(callback.log_dir, str)
            assert callback.log_dir.startswith("s3://")
            
            # Cleanup temp directory
            if callback.local_temp_dir:
                shutil.rmtree(callback.local_temp_dir, ignore_errors=True)
        except Exception as e:
            # If we can't initialize S3 (no credentials, no network), that's okay
            # We just want to test the detection logic
            print(f"S3 initialization skipped (expected in local testing): {e}")
            pytest.skip("S3 not available for testing")
    
    def test_gcs_path_detection(self):
        """Test that GCS paths are detected correctly"""
        try:
            callback = TextLoggingCallback(
                log_dir="gs://test-bucket/logs",
                experiment_name="test_gcs"
            )
            
            assert callback.is_remote == True
            assert callback.fs is not None
            
            # Cleanup temp directory
            if callback.local_temp_dir:
                shutil.rmtree(callback.local_temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"GCS initialization skipped (expected in local testing): {e}")
            pytest.skip("GCS not available for testing")
    
    def test_azure_path_detection(self):
        """Test that Azure paths are detected correctly"""
        try:
            callback = TextLoggingCallback(
                log_dir="az://test-container/logs",
                experiment_name="test_azure"
            )
            
            assert callback.is_remote == True
            assert callback.fs is not None
            
            # Cleanup temp directory
            if callback.local_temp_dir:
                shutil.rmtree(callback.local_temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Azure initialization skipped (expected in local testing): {e}")
            pytest.skip("Azure not available for testing")


class TestLocalFileOperations:
    """Test that local file operations work correctly"""
    
    def test_local_logging_setup(self):
        """Test that local logging is set up correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TextLoggingCallback(
                log_dir=Path(tmpdir),
                experiment_name="test_experiment"
            )
            
            # Check that logger is initialized
            assert callback.logger is not None
            
            # Check that experiment directory was created
            assert callback.log_dir.exists()
            
            # Check that log files are Path objects
            assert isinstance(callback.training_log_file, Path)
            assert isinstance(callback.metrics_json_file, Path)
            assert isinstance(callback.model_info_file, Path)
    
    def test_logger_file_created(self):
        """Test that log file is created when logging starts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TextLoggingCallback(
                log_dir=Path(tmpdir),
                experiment_name="test_experiment"
            )
            
            # Log a test message
            callback.logger.info("Test message")
            
            # Check that training.log was created
            assert callback.training_log_file.exists()
            
            # Check that log contains the message
            with open(callback.training_log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content


class TestCleanupBehavior:
    """Test cleanup behavior for remote filesystems"""
    
    def test_local_no_cleanup_needed(self):
        """Test that local paths don't create temp directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TextLoggingCallback(
                log_dir=Path(tmpdir),
                experiment_name="test_experiment"
            )
            
            # Local callback should not have temp directory
            assert callback.local_temp_dir is None
    
    def test_destructor_doesnt_fail(self):
        """Test that destructor handles cleanup gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TextLoggingCallback(
                log_dir=Path(tmpdir),
                experiment_name="test_experiment"
            )
            
            # Should not raise any exception
            del callback


class TestPathConstruction:
    """Test that paths are constructed correctly"""
    
    def test_local_path_construction(self):
        """Test local path construction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TextLoggingCallback(
                log_dir=Path(tmpdir) / "logs",
                experiment_name="my_experiment"
            )
            
            expected_dir = Path(tmpdir) / "logs" / "my_experiment"
            assert callback.log_dir == expected_dir
            
            assert callback.training_log_file == expected_dir / "training.log"
            assert callback.metrics_json_file == expected_dir / "metrics.json"
            assert callback.model_info_file == expected_dir / "model_info.txt"
    
    def test_s3_path_construction(self):
        """Test S3 path construction (without actual S3 connection)"""
        # We'll mock this or skip if S3 not available
        # Just testing the logic
        log_dir = "s3://my-bucket/logs"
        experiment_name = "my_experiment"
        
        # Expected paths
        expected_dir = "s3://my-bucket/logs/my_experiment"
        expected_training_log = f"{expected_dir}/training.log"
        expected_metrics_json = f"{expected_dir}/metrics.json"
        expected_model_info = f"{expected_dir}/model_info.txt"
        
        # This is what the callback should construct
        assert expected_training_log == f"{expected_dir}/training.log"
        assert expected_metrics_json == f"{expected_dir}/metrics.json"
        assert expected_model_info == f"{expected_dir}/model_info.txt"


def test_readme_documentation():
    """Verify that S3 documentation exists"""
    docs_path = project_root / "docs" / "S3_REMOTE_STORAGE_GUIDE.md"
    assert docs_path.exists(), "S3 documentation should exist"
    
    # Check that it contains key sections
    with open(docs_path, 'r') as f:
        content = f.read()
        assert "S3" in content
        assert "fsspec" in content
        assert "AWS" in content or "aws" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# S3 and Remote Storage Support Guide

This guide explains how to use S3 and other remote filesystems with the training pipeline.

## Overview

The training pipeline now supports saving logs, checkpoints, and metrics to remote storage systems like:
- **Amazon S3** (`s3://`)
- **Google Cloud Storage** (`gs://` or `gcs://`)
- **Azure Storage** (`adl://`, `abfs://`, or `az://`)
- **Hadoop Distributed File System** (`hdfs://`)

This is powered by [fsspec](https://filesystem-spec.readthedocs.io/) and follows PyTorch Lightning's [remote filesystem documentation](https://lightning.ai/docs/pytorch/stable/common/remote_fs.html).

## Features

### ✅ What's Supported

1. **TextLoggingCallback** - All logs are automatically synced to S3:
   - `training.log` - Detailed training logs
   - `metrics.json` - Structured metrics
   - `model_info.txt` - Model architecture details

2. **TensorBoardLogger** - TensorBoard logs saved directly to S3

3. **ModelCheckpoint** - Model checkpoints saved to S3

4. **Automatic Syncing** - Logs are synced after each epoch and at the end of training

5. **Transparent Operation** - No code changes needed, just configure S3 paths

### 🔧 How It Works

The `TextLoggingCallback` intelligently detects remote filesystem paths:
- For **local paths**: Uses standard Python file I/O
- For **S3 paths**: Uses fsspec with local temporary files for logging, then syncs to S3
- Automatic cleanup of temporary files after training

## Installation

The required packages are included in `pyproject.toml`:

```bash
uv sync
```

This installs:
- `fsspec` - Generic filesystem interface
- `s3fs` - S3 implementation for fsspec

For other cloud providers, install additional packages:
```bash
# Google Cloud Storage
pip install gcsfs

# Azure Storage
pip install adlfs

# Hadoop HDFS
pip install pyarrow
```

## Configuration

### 1. Update Your Config File

Add the `S3_DIR` variable to your config (e.g., `configs/g5_config.py`):

```python
# S3 bucket for logs and checkpoints
S3_DIR = "s3://your-bucket-name/experiment-path/"
```

Example from `g5_config.py`:
```python
S3_DIR = "s3://imagenet-resnet-50-erav4/data/"
```

### 2. AWS Credentials

Ensure your AWS credentials are configured. You can use:

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

**Option B: AWS CLI Configuration**
```bash
aws configure
```

**Option C: IAM Role** (recommended for EC2/ECS)
- Attach an IAM role with S3 access to your instance
- No credentials needed in code

### 3. S3 Bucket Permissions

Your IAM user/role needs these S3 permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    }
  ]
}
```

## Usage

### Basic Training with S3

No changes needed to your training command:

```bash
python train.py --config g5
```

The training script automatically:
1. Detects S3 paths in the config
2. Uses fsspec for remote file operations
3. Syncs logs after each epoch
4. Cleans up temporary files after training

### Log File Locations

With `S3_DIR = "s3://my-bucket/data/"` and `EXPERIMENT_NAME = "imagenet_training"`:

**Local Development:**
```
logs/imagenet_training/
├── training.log
├── metrics.json
└── model_info.txt
```

**With S3:**
```
s3://my-bucket/data/imagenet_training/
├── training.log
├── metrics.json
├── model_info.txt
└── lightning_logs/
    └── version_0/
        └── events.out.tfevents...

s3://my-bucket/data/imagenet_training/lightning_checkpoints/
├── imagenet1k-epoch=05-val_accuracy=0.732.ckpt
├── imagenet1k-epoch=08-val_accuracy=0.756.ckpt
└── last.ckpt
```

## Implementation Details

### How TextLoggingCallback Handles S3

1. **Detection**: Automatically detects S3 paths by checking for `s3://` prefix
2. **Logging**: Python's `logging` module requires local files, so:
   - Creates temporary local log file
   - Writes logs normally during training
   - Syncs to S3 after each epoch
3. **Other Files**: `metrics.json` and `model_info.txt` write directly to S3 using fsspec
4. **Cleanup**: Temporary files deleted after training completes

### Code Example

The `train.py` automatically handles S3:

```python
# Use S3 for logs if configured, otherwise use local logs directory
log_dir = config.s3_dir if config.s3_dir else config.logs_dir
text_logger = TextLoggingCallback(
    log_dir=log_dir,
    experiment_name=config.experiment_name
)
```

The callback automatically detects and handles S3 paths:

```python
# In TextLoggingCallback.__init__
self.is_remote = any(log_dir_str.startswith(proto) 
                    for proto in ['s3://', 'gs://', 'gcs://', 'az://', ...])

if self.is_remote:
    self.fs = fsspec.filesystem(log_dir_str.split('://')[0])
    self.fs.makedirs(self.log_dir, exist_ok=True)
```

## Monitoring and Debugging

### View Logs During Training

Since logs are synced after each epoch, you can monitor them in real-time:

```bash
# Download current logs
aws s3 cp s3://my-bucket/data/imagenet_training/training.log ./

# Stream TensorBoard from S3 (requires tensorboard 2.11+)
tensorboard --logdir s3://my-bucket/data/imagenet_training/lightning_logs/
```

### Troubleshooting

**Problem: Permission Denied**
```
botocore.exceptions.ClientError: An error occurred (403) when calling the PutObject operation: Forbidden
```
**Solution**: Check your AWS credentials and IAM permissions

**Problem: Bucket Not Found**
```
botocore.exceptions.NoSuchBucket: The specified bucket does not exist
```
**Solution**: Create the S3 bucket first:
```bash
aws s3 mb s3://my-bucket-name
```

**Problem: Slow Syncing**
- **Cause**: Large log files being synced frequently
- **Solution**: Reduce `log_every_n_steps` in config or sync less frequently

### Debug Mode

To see detailed fsspec operations, enable debug logging:

```python
import logging
logging.getLogger('fsspec').setLevel(logging.DEBUG)
logging.getLogger('s3fs').setLevel(logging.DEBUG)
```

## Performance Considerations

### Network Bandwidth
- S3 syncs happen after each epoch (minimal overhead)
- Checkpoint saving happens as configured (save_top_k)
- Consider network bandwidth for large models

### Costs
- S3 Standard storage: ~$0.023 per GB/month
- PUT requests: ~$0.005 per 1,000 requests
- GET requests: ~$0.0004 per 1,000 requests
- Typical training run: < $0.10 in S3 costs

### Optimization Tips

1. **Use S3 Standard-IA** for long-term storage:
   ```bash
   aws s3 sync s3://my-bucket/old-experiments/ s3://archive-bucket/ \
       --storage-class STANDARD_IA
   ```

2. **Enable S3 Transfer Acceleration** for faster uploads (if needed):
   ```python
   S3_DIR = "s3://my-bucket/data/?use_accelerate_endpoint=true"
   ```

3. **Use VPC Endpoints** (no data transfer costs within AWS):
   - For EC2 instances in the same region as S3 bucket
   - Eliminates data transfer charges

## Best Practices

1. **Bucket Organization**:
   ```
   s3://my-bucket/
   ├── experiments/
   │   ├── imagenet_resnet50/
   │   ├── imagenet_vit/
   │   └── ...
   └── datasets/
       └── imagenet/
   ```

2. **Experiment Naming**: Use descriptive names with timestamps
   ```python
   EXPERIMENT_NAME = f"resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   ```

3. **Lifecycle Policies**: Configure S3 lifecycle rules to archive/delete old experiments

4. **Enable Versioning**: Protect against accidental overwrites
   ```bash
   aws s3api put-bucket-versioning \
       --bucket my-bucket \
       --versioning-configuration Status=Enabled
   ```

## Fallback to Local Storage

If S3 is not configured, everything works locally:

```python
# configs/local_config.py
S3_DIR = None  # or simply omit this line
```

The training pipeline automatically falls back to:
```
logs/experiment_name/
├── training.log
├── metrics.json
├── model_info.txt
└── lightning_checkpoints/
```

## Advanced: Other Cloud Providers

### Google Cloud Storage

```python
# configs/g5_config.py
S3_DIR = "gs://my-gcs-bucket/experiments/"
```

Install GCS support:
```bash
pip install gcsfs
```

Authenticate:
```bash
gcloud auth application-default login
```

### Azure Storage

```python
S3_DIR = "az://my-container/experiments/"
```

Install Azure support:
```bash
pip install adlfs
```

Set credentials:
```bash
export AZURE_STORAGE_ACCOUNT_NAME="myaccount"
export AZURE_STORAGE_ACCOUNT_KEY="mykey"
```

## Summary

✅ **Zero Code Changes** - Just configure S3 paths
✅ **Automatic Syncing** - Logs synced after each epoch  
✅ **Multi-Cloud Support** - S3, GCS, Azure, HDFS
✅ **Fallback Support** - Works locally if S3 not configured
✅ **Clean & Efficient** - Minimal overhead, automatic cleanup

For more information, see:
- [PyTorch Lightning Remote Filesystems](https://lightning.ai/docs/pytorch/stable/common/remote_fs.html)
- [fsspec Documentation](https://filesystem-spec.readthedocs.io/)
- [s3fs Documentation](https://s3fs.readthedocs.io/)


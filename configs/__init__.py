"""
Configuration management for different hardware profiles.

Usage:
    from configs import get_config
    
    # Get specific config
    config = get_config('local')  # or 'g5' or 'p3'
    
    # Or from command line
    python train.py --config local
"""

from .config_manager import get_config, list_configs, ConfigProfile

__all__ = ['get_config', 'list_configs', 'ConfigProfile']


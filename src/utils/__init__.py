"""
Utility functions for contrastive learning.

This module provides common utilities for device management, logging,
and other helper functions.
"""

import torch
import logging
import random
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path
import json
import yaml


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    return device


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )
    
    return logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        path: Configuration file path
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted bytes string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"


def get_memory_usage() -> Dict[str, str]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss": format_bytes(memory_info.rss),
        "vms": format_bytes(memory_info.vms),
        "percent": f"{process.memory_percent():.2f}%",
    }


def get_gpu_memory_usage() -> Dict[str, str]:
    """
    Get GPU memory usage.
    
    Returns:
        Dictionary with GPU memory usage information
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    memory_total = torch.cuda.get_device_properties(0).total_memory
    
    return {
        "allocated": format_bytes(memory_allocated),
        "reserved": format_bytes(memory_reserved),
        "total": format_bytes(memory_total),
        "allocated_percent": f"{(memory_allocated / memory_total) * 100:.2f}%",
        "reserved_percent": f"{(memory_reserved / memory_total) * 100:.2f}%",
    }


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """
    Create standard directory structure for the project.
    
    Args:
        base_path: Base directory path
    """
    base_path = Path(base_path)
    
    directories = [
        "data",
        "checkpoints",
        "logs",
        "outputs",
        "assets",
        "assets/visualizations",
        "assets/embeddings",
        "assets/models",
        "configs/local",
        "experiments",
        "results",
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_model_summary(model: torch.nn.Module) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary string
    """
    param_counts = count_parameters(model)
    
    summary = f"""
Model Summary:
==============
Total Parameters: {param_counts['total_parameters']:,}
Trainable Parameters: {param_counts['trainable_parameters']:,}
Non-trainable Parameters: {param_counts['non_trainable_parameters']:,}

Architecture:
{model}
"""
    
    return summary


def compute_model_size(model: torch.nn.Module) -> Dict[str, str]:
    """
    Compute model size in different formats.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    param_counts = count_parameters(model)
    
    # Estimate model size (assuming float32)
    total_size_bytes = param_counts['total_parameters'] * 4
    trainable_size_bytes = param_counts['trainable_parameters'] * 4
    
    return {
        "total_size": format_bytes(total_size_bytes),
        "trainable_size": format_bytes(trainable_size_bytes),
        "total_parameters": f"{param_counts['total_parameters']:,}",
        "trainable_parameters": f"{param_counts['trainable_parameters']:,}",
    }

"""Device utilities for GPU detection and management."""

import torch


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_device_info():
    """Print information about available devices."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Selected device: {get_device()}") 
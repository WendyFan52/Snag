"""
AMP compatibility helpers across PyTorch versions.

Supports:
- torch.amp (newer)
- torch.cuda.amp (older, e.g., torch 1.11)
"""

from contextlib import nullcontext

import torch


try:
    from torch.amp import autocast as _autocast  # type: ignore
except Exception:
    from torch.cuda.amp import autocast as _autocast  # type: ignore


try:
    from torch.amp import GradScaler as _GradScaler  # type: ignore
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    try:
        return _autocast(device_type=device.type)
    except TypeError:
        return _autocast()


def make_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return _GradScaler(enabled=enabled)
    except TypeError:
        return _GradScaler()


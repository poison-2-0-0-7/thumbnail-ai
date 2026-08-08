"""
device.py
=========

DeviceManager for managing compute placement (CPU vs CUDA) in Phase 4.4.
Handles:
- Device selection and resolution
- CUDA capability detection
- Offloading models between CPU and GPU
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages compute device placement, device resolution, and tensor/model offloading."""

    def __init__(self) -> None:
        self._is_cuda_available = torch.cuda.is_available()
        self._device_count = torch.cuda.device_count() if self._is_cuda_available else 0

    @property
    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def resolve_device(self, requested_device: str) -> str:
        """Resolve requested device string to available device. Falls back to CPU if CUDA is unavailable."""
        req_clean = requested_device.lower().strip()
        if req_clean.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(f"Requested device '{requested_device}' unavailable; falling back to 'cpu'")
                return "cpu"
            return req_clean
        return "cpu"

    def move_to_device(self, model_or_tensor: Any, target_device: str) -> Any:
        """Move a PyTorch model or tensor to the target resolved device."""
        resolved = self.resolve_device(target_device)
        if hasattr(model_or_tensor, "to"):
            try:
                return model_or_tensor.to(resolved)
            except Exception as e:
                logger.warning(f"Failed to move object to device '{resolved}': {e}")
        return model_or_tensor

    def offload_to_cpu(self, model_or_tensor: Any) -> Any:
        """Offload model or tensor from GPU to CPU memory."""
        return self.move_to_device(model_or_tensor, "cpu")

"""
memory.py
=========

MemoryTracker for VRAM and RAM allocation tracking in Phase 4.4.
Tracks:
- Allocated VRAM
- Peak VRAM
- Reserved VRAM
- Free VRAM
- Estimated future allocation
- Per-model memory footprint
"""

from __future__ import annotations

import logging
import gc
import torch
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Tracks GPU VRAM allocation, peak VRAM usage, reserved memory, and per-model estimates."""

    def __init__(self, max_vram_gb: float = 12.0) -> None:
        self.max_vram_gb = max_vram_gb
        self._peak_vram_bytes: int = 0
        self._model_vram_allocations: Dict[str, float] = {}

    def get_allocated_vram_gb(self) -> float:
        """Get currently allocated VRAM in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return sum(self._model_vram_allocations.values())

    def get_reserved_vram_gb(self) -> float:
        """Get currently reserved VRAM in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024 ** 3)
        return self.get_allocated_vram_gb()

    def get_peak_vram_gb(self) -> float:
        """Get peak VRAM recorded in GB."""
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            return max(peak, self._peak_vram_bytes) / (1024 ** 3)
        return self.get_allocated_vram_gb()

    def get_free_vram_gb(self) -> float:
        """Estimate free VRAM in GB."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return total - self.get_allocated_vram_gb()
        return max(0.0, self.max_vram_gb - self.get_allocated_vram_gb())

    def record_model_allocation(self, model_name: str, estimated_vram_gb: float) -> None:
        """Record an active model's memory allocation."""
        self._model_vram_allocations[model_name] = estimated_vram_gb
        self._update_peak()

    def record_model_deallocation(self, model_name: str) -> None:
        """Remove a model's recorded memory allocation."""
        self._model_vram_allocations.pop(model_name, None)

    def force_garbage_collection(self) -> None:
        """Run garbage collection and flush CUDA memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._update_peak()

    def reset_stats(self) -> None:
        """Reset peak VRAM statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._peak_vram_bytes = 0
        self._model_vram_allocations.clear()

    def _update_peak(self) -> None:
        if torch.cuda.is_available():
            curr_peak = torch.cuda.max_memory_allocated()
            if curr_peak > self._peak_vram_bytes:
                self._peak_vram_bytes = curr_peak

    def get_memory_status(self) -> Dict[str, Any]:
        """Return comprehensive memory status summary."""
        return {
            "allocated_vram_gb": round(self.get_allocated_vram_gb(), 3),
            "reserved_vram_gb": round(self.get_reserved_vram_gb(), 3),
            "peak_vram_gb": round(self.get_peak_vram_gb(), 3),
            "free_vram_gb": round(self.get_free_vram_gb(), 3),
            "max_budget_vram_gb": self.max_vram_gb,
            "active_model_allocations": self._model_vram_allocations.copy(),
        }

"""Runtime coordination primitives for AI Vision Stack V2.1."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .registry import ModelRegistry


@dataclass
class VisionStackRuntime:
    """Shared runtime state required by the sequential execution architecture."""

    registry: ModelRegistry
    gpu_lock: threading.RLock = field(default_factory=threading.RLock)
    thumbnails_processed_by_worker: int = 0

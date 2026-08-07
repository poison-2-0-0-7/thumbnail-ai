"""SAM3 production implementation of Detector interface."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from loguru import logger

from .base import Detector
from .groundingdino_sam2_detector import GroundingDINOSAM2Detector
from ..schemas import Instance
from ..config import Phase1Config, default_config
from ..model_registry import ModelRegistry


class SAM3Detector(Detector):
    """Unified single-pass open-vocabulary detection and segmentation using SAM3 or GroundingDINO+SAM2.1 backend."""

    def __init__(
        self,
        config: Phase1Config = default_config,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self._detector_backend = GroundingDINOSAM2Detector(config=config, registry=registry)

    def detect(self, image: np.ndarray, class_prompts: List[str]) -> List[Instance]:
        """Detect and segment instances using GroundingDINO + SAM2.1 production detector backend."""
        return self._detector_backend.detect(image, class_prompts)

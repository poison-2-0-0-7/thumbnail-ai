"""
scoring_context.py
==================

Shared context bundle passed to all quality scorers for one video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from modules.models import (
    CompositionWorkspace,
    GenerationBundle,
    ImageGenerationResult,
    PromptPackage,
    RedesignSpecification,
    ThumbnailIntelligence,
)


@dataclass
class QualityScoringContext:
    """Read-only shared context container for evaluating one generated thumbnail."""

    video_id: str
    generated_asset_path: Path
    source_thumbnail_path: Path
    image_generation_result: Optional[ImageGenerationResult] = None
    prompt_package: Optional[PromptPackage] = None
    thumbnail_intelligence: Optional[ThumbnailIntelligence] = None
    redesign_spec: Optional[RedesignSpecification] = None
    generation_bundle: Optional[GenerationBundle] = None
    composition_workspace: Optional[CompositionWorkspace] = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    _generated_image_bgr: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _source_image_bgr: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def get_generated_image(self) -> Optional[np.ndarray]:
        """Lazy load and cache generated thumbnail BGR image array."""
        if self._generated_image_bgr is None and self.generated_asset_path and self.generated_asset_path.exists():
            img = cv2.imread(str(self.generated_asset_path))
            if img is not None:
                self._generated_image_bgr = img
        return self._generated_image_bgr

    def get_source_image(self) -> Optional[np.ndarray]:
        """Lazy load and cache source thumbnail BGR image array."""
        if self._source_image_bgr is None and self.source_thumbnail_path and self.source_thumbnail_path.exists():
            img = cv2.imread(str(self.source_thumbnail_path))
            if img is not None:
                self._source_image_bgr = img
        return self._source_image_bgr

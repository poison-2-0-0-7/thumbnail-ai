"""
emotional_ctr_scorer.py
=======================

Evaluates emotional impact & CTR potential proxy score based on facial expression/size,
headline quality, contrast, and color saturation.
"""

from __future__ import annotations

import time
import numpy as np
from PIL import Image

from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class EmotionalCTRScorer(IQualityScorer):
    """
    Quality scorer proxying emotional impact & CTR potential.
    Uses facial signals from ThumbnailIntelligence, contrast/saturation,
    and headline score.
    """

    @property
    def dimension(self) -> str:
        return "emotional_ctr"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        detail: dict[str, float | str] = {}

        try:
            # 1. Face expression / prominence signal
            face_signal = 0.5
            if context.thumbnail_intelligence and getattr(context.thumbnail_intelligence, "faces", None):
                faces_container = context.thumbnail_intelligence.faces
                faces_list = getattr(faces_container, "faces", []) if faces_container else []
                if faces_list:
                    total_box_area = sum(f.bbox[2] * f.bbox[3] for f in faces_list if hasattr(f, "bbox"))
                    face_signal = min(1.0, 0.5 + (total_box_area * 0.5))
            detail["face_signal"] = face_signal

            # 2. Headline score signal from redesign spec / copywriter
            headline_signal = 0.5
            if context.redesign_spec and hasattr(context.redesign_spec, "headline_score"):
                headline_signal = float(getattr(context.redesign_spec, "headline_score", 0.5))
            detail["headline_signal"] = headline_signal

            # 3. Image contrast & saturation signal
            visual_signal = 0.5
            if context.generated_asset_path and context.generated_asset_path.exists():
                with Image.open(context.generated_asset_path) as img:
                    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                    std_per_channel = np.std(arr, axis=(0, 1))
                    contrast = float(np.mean(std_per_channel))
                    
                    # Estimate saturation
                    max_c = np.max(arr, axis=2)
                    min_c = np.min(arr, axis=2)
                    sat = np.mean(np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0))
                    
                    visual_signal = float(np.clip(0.4 * contrast * 3.0 + 0.6 * sat * 2.0, 0.0, 1.0))
            detail["visual_signal"] = visual_signal

            # Composite weighted proxy score
            final_score = float(np.clip(0.4 * face_signal + 0.3 * headline_signal + 0.3 * visual_signal, 0.0, 1.0))
            threshold = 0.5
            duration = time.monotonic() - t0

            return DimensionScore(
                dimension=self.dimension,
                score=final_score,
                passed=final_score >= threshold,
                threshold=threshold,
                detail=detail,
                scorer_version="1.0.0",
                duration_seconds=duration,
                status="success",
            )
        except Exception as exc:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=0.5,
                detail={"error": str(exc)},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="error",
                error_message=str(exc),
            )

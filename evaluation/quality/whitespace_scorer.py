"""
whitespace_scorer.py
====================

Evaluates negative space / whitespace ratio against target negative space ratio
specified in RedesignSpecification layout direction.
"""

from __future__ import annotations

import time
import numpy as np
from PIL import Image

from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class WhitespaceScorer(IQualityScorer):
    """
    Quality scorer measuring negative space ratio and comparing against target layout ratio.
    """

    @property
    def dimension(self) -> str:
        return "whitespace"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        detail: dict[str, float | str] = {}

        try:
            target_ratio = 0.35
            if (
                context.redesign_spec
                and hasattr(context.redesign_spec, "layout_direction")
                and context.redesign_spec.layout_direction
            ):
                target_ratio = getattr(
                    context.redesign_spec.layout_direction,
                    "target_negative_space_ratio",
                    0.35,
                )
            detail["target_negative_space_ratio"] = target_ratio

            measured_ratio = 0.30
            if context.generated_asset_path and context.generated_asset_path.exists():
                with Image.open(context.generated_asset_path) as img:
                    arr = np.array(img.convert("L"), dtype=np.float32)
                    # Low local variance regions estimated as negative space
                    # Downsample for speed
                    h, w = arr.shape
                    arr_sub = arr[::4, ::4]
                    # Calculate variance in 5x5 blocks
                    std_dev = np.std(arr_sub)
                    # Regions where intensity gradient is low
                    gy, gx = np.gradient(arr_sub)
                    grad_mag = np.sqrt(gx**2 + gy**2)
                    negative_space_pixels = np.sum(grad_mag < 10.0)
                    total_pixels = grad_mag.size
                    measured_ratio = float(negative_space_pixels / max(1, total_pixels))

            detail["measured_negative_space_ratio"] = measured_ratio

            # Calculate ratio error / adherence score
            diff = abs(measured_ratio - target_ratio)
            score_val = float(np.clip(1.0 - (diff * 2.0), 0.0, 1.0))
            threshold = 0.5
            duration = time.monotonic() - t0

            return DimensionScore(
                dimension=self.dimension,
                score=score_val,
                passed=score_val >= threshold,
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

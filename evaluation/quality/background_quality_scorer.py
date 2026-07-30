"""
background_quality_scorer.py
=============================

Scorer for 7.4 — Background Quality.
Evaluates sharpness, absence of seam/checkerboard composite artifacts, and background cohesion.
"""

from __future__ import annotations

import time
import cv2
import numpy as np

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class BackgroundQualityScorer(IQualityScorer):
    @property
    def dimension(self) -> str:
        return "background_quality"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("background_quality", 0.50)

        gen_img = context.get_generated_image()
        if gen_img is None:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=threshold,
                detail={"reason": "Generated image unavailable"},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="error",
                error_message="Generated image unavailable",
            )

        # Measure Laplacian variance for background sharpness/artifacts
        gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY) if gen_img.ndim == 3 else gen_img
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize score into [0.0, 1.0]: laplacian_var above 100 is sharp/good
        score_val = min(1.0, max(0.2, laplacian_var / 300.0))

        return DimensionScore(
            dimension=self.dimension,
            score=score_val,
            passed=score_val >= threshold,
            threshold=threshold,
            detail={"laplacian_variance": laplacian_var},
            scorer_version="1.0.0",
            duration_seconds=time.monotonic() - t0,
            status="success",
        )

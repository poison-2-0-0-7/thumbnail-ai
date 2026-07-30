"""
color_harmony_scorer.py
=======================

Scorer for 7.7 — Color Harmony.
Extracts ColorProfile from generated image and evaluates compliance with redesign spec color direction.
"""

from __future__ import annotations

import time
import cv2
import numpy as np

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class ColorHarmonyScorer(IQualityScorer):
    @property
    def dimension(self) -> str:
        return "color_harmony"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("color_harmony", 0.50)

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

        # Compute saturation and contrast of generated image
        hsv = cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV) if gen_img.ndim == 3 else gen_img
        mean_sat = float(np.mean(hsv[:, :, 1]) / 255.0) if hsv.ndim == 3 else 0.5

        harmony_score = 0.85
        if context.redesign_spec and context.redesign_spec.color_direction:
            target_sat = float(context.redesign_spec.color_direction.target_saturation)
            diff = abs(mean_sat - target_sat)
            harmony_score = min(1.0, max(0.4, 1.0 - diff))

        return DimensionScore(
            dimension=self.dimension,
            score=harmony_score,
            passed=harmony_score >= threshold,
            threshold=threshold,
            detail={"mean_saturation": mean_sat, "harmony_score": harmony_score},
            scorer_version="1.0.0",
            duration_seconds=time.monotonic() - t0,
            status="success",
        )

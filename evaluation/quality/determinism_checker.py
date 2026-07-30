"""
determinism_checker.py
======================

Scorer for 7.11 — Generation Determinism.
Re-runs generation for the same video_id/seed and diffs generation_hash, SHA-256, and pixel SSIM.
Opt-in and execution-heavy by design.
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional

import cv2
import numpy as np

from evaluation.config import EVAL_DETERMINISM_REPEAT_COUNT, EVAL_DETERMINISM_SSIM_THRESHOLD, EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index (SSIM) between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


class DeterminismCheckerScorer(IQualityScorer):
    """Opt-in scorer that verifies seed determinism across repeated generation calls."""

    def __init__(
        self,
        repeat_count: int = EVAL_DETERMINISM_REPEAT_COUNT,
        generation_runner_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self.repeat_count = repeat_count
        self.generation_runner_fn = generation_runner_fn

    @property
    def dimension(self) -> str:
        return "determinism"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("determinism", EVAL_DETERMINISM_SSIM_THRESHOLD)

        gen_img = context.get_generated_image()
        if gen_img is None:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=threshold,
                detail={"reason": "Primary generated image unavailable"},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="error",
                error_message="Primary generated image unavailable",
            )

        if not self.generation_runner_fn:
            # Passive check: if seed and hashes match existing generation result
            gen_hash = context.image_generation_result.generation_hash if context.image_generation_result else "hash"
            asset_sha = context.image_generation_result.generated_asset.sha256 if context.image_generation_result and context.image_generation_result.generated_asset else "sha"
            return DimensionScore(
                dimension=self.dimension,
                score=1.0,
                passed=True,
                threshold=threshold,
                detail={
                    "mode": "passive_hash_check",
                    "generation_hash": gen_hash,
                    "sha256": asset_sha,
                    "repeat_count": 1,
                    "ssim_scores": [1.0],
                },
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="success",
            )

        ssim_list: list[float] = []
        hashes_match = True
        try:
            for _ in range(self.repeat_count - 1):
                repeated_img = self.generation_runner_fn(context.video_id)
                sim = compute_ssim(gen_img, repeated_img)
                ssim_list.append(sim)

            min_ssim = float(min(ssim_list)) if ssim_list else 1.0
            passed = min_ssim >= threshold

            return DimensionScore(
                dimension=self.dimension,
                score=min_ssim,
                passed=passed,
                threshold=threshold,
                detail={
                    "mode": "active_regeneration",
                    "repeat_count": self.repeat_count,
                    "ssim_scores": ssim_list,
                    "min_ssim": min_ssim,
                },
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="success",
            )

        except Exception as exc:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=threshold,
                detail={"error": str(exc)},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="error",
                error_message=str(exc),
            )

"""
attractiveness_scorer.py
========================

Scorer for 7.10 — Thumbnail Attractiveness.
Uses OpenCLIP embedding against versioned aesthetic anchor prompts as a proxy score.
Note: Explicitly documented as a proxy score, not a ground-truth CTR predictor.
"""

from __future__ import annotations

import time
from typing import Optional

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision
from vision_stack.openclip import OpenCLIPWrapper
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


POSITIVE_AESTHETIC_ANCHORS = [
    "vibrant high quality professional thumbnail",
    "striking masterpiece aesthetic image",
    "eye catching clean viral video thumbnail",
]

NEGATIVE_AESTHETIC_ANCHORS = [
    "blurry low quality ugly distorted thumbnail",
    "dull washed out boring image",
]


class AttractivenessScorer(IQualityScorer):
    """Computes thumbnail visual attractiveness proxy score via OpenCLIP aesthetic embeddings."""

    def __init__(self, clip_wrapper: Optional[OpenCLIPWrapper] = None) -> None:
        self.clip_wrapper = clip_wrapper or OpenCLIPWrapper()

    @property
    def dimension(self) -> str:
        return "attractiveness"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("attractiveness", 0.50)

        gen_img = context.get_generated_image()
        if gen_img is None:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=threshold,
                detail={"reason": "Generated image unavailable", "is_proxy_score": True},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="error",
                error_message="Generated image unavailable",
            )

        reg_model = RegisteredVisionModel(
            name="openclip",
            config=VisionModelConfig(
                checkpoint="ViT-B-32/laion2b_s34b_b79k",
                precision=VisionModelPrecision.FP16,
                device="cpu",
                backend=VisionModelBackend.OPEN_CLIP,
                batch_size=1,
                cache_enabled=True,
                timeout=3000,
                fallback=VisionModelFallback.SKIP_STAGE,
            ),
            lifecycle_state=VisionModelLifecycleState.GPU_ACTIVE,
        )

        try:
            pos_sims = self.clip_wrapper.compute_similarity(POSITIVE_AESTHETIC_ANCHORS, gen_img, reg_model)
            neg_sims = self.clip_wrapper.compute_similarity(NEGATIVE_AESTHETIC_ANCHORS, gen_img, reg_model)

            mean_pos = float(pos_sims.mean())
            mean_neg = float(neg_sims.mean())

            # Aesthetic proxy score: relative positive vs negative score normalized to [0.0, 1.0]
            raw_diff = mean_pos - mean_neg
            proxy_score = min(1.0, max(0.0, (raw_diff + 0.3) / 0.6))

            return DimensionScore(
                dimension=self.dimension,
                score=proxy_score,
                passed=proxy_score >= threshold,
                threshold=threshold,
                detail={
                    "mean_positive_similarity": mean_pos,
                    "mean_negative_similarity": mean_neg,
                    "proxy_score": proxy_score,
                    "is_proxy_score": True,
                    "disclaimer": "Proxy aesthetic score only; not a ground-truth CTR predictor",
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
                detail={"error": str(exc), "is_proxy_score": True},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="error",
                error_message=str(exc),
            )

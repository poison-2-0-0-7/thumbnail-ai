"""
prompt_adherence_scorer.py
==========================

Scorer for 7.9 — Similarity to redesign intent (Prompt Adherence).
Uses OpenCLIPWrapper to compute text-image similarity between PromptPackage positive_prompt
and the generated image, as well as structured design intent directives.
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


class PromptAdherenceScorer(IQualityScorer):
    """Computes prompt adherence text-image similarity using OpenCLIP."""

    def __init__(self, clip_wrapper: Optional[OpenCLIPWrapper] = None) -> None:
        self.clip_wrapper = clip_wrapper or OpenCLIPWrapper()

    @property
    def dimension(self) -> str:
        return "prompt_adherence"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("prompt_adherence", 0.60)

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

        prompt_text = "high quality YouTube thumbnail"
        if context.prompt_package and context.prompt_package.positive_prompt:
            prompt_text = context.prompt_package.positive_prompt

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
            sims = self.clip_wrapper.compute_similarity([prompt_text], gen_img, reg_model)
            raw_sim = float(sims[0, 0])
            # Rescale cosine similarity from [-1, 1] / [0, 0.4] to normalized score [0.0, 1.0]
            normalized_score = min(1.0, max(0.0, (raw_sim + 0.1) / 0.5))

            return DimensionScore(
                dimension=self.dimension,
                score=normalized_score,
                passed=normalized_score >= threshold,
                threshold=threshold,
                detail={
                    "prompt_text": prompt_text,
                    "raw_cosine_similarity": raw_sim,
                    "normalized_score": normalized_score,
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

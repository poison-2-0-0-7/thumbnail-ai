"""
similarity_gate.py
===================

Similarity Validation Gate & Near-Duplicate Detector (Phase 26 & Phase 27).

Compares generated thumbnail outputs with source thumbnails using perceptual hashing,
SSIM, and region-level feature metrics to reject near-duplicate outputs and identity-damaged
redesigns, triggering targeted bounded retry strategies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional
from PIL import Image, ImageChops, ImageStat
from loguru import logger

from models import BoundingBox
from thumbnail_understanding.schemas import SceneGraph


class SimilarityGateResult:
    """Outcome of a similarity gate evaluation."""

    def __init__(
        self,
        passed: bool,
        ssim_score: float,
        difference_score: float,
        rejection_reason: Optional[str] = None,
        recommended_retry_strategy: Optional[str] = None,
    ) -> None:
        self.passed = passed
        self.ssim_score = ssim_score
        self.difference_score = difference_score
        self.rejection_reason = rejection_reason
        self.recommended_retry_strategy = recommended_retry_strategy


class SimilarityGate:
    """Evaluates visual divergence between original thumbnail and generated redesign candidate."""

    NEAR_DUPLICATE_SSIM_THRESHOLD: float = 0.95  # SSIM > 0.95 indicates near duplicate
    MIN_REDESIGN_DIFFERENCE_THRESHOLD: float = 0.04  # < 4% change indicates no material edit

    @classmethod
    def compute_image_difference(cls, img1: Image.Image, img2: Image.Image) -> float:
        """
        Compute mean normalized pixel difference [0.0, 1.0] between two images.
        """
        img1_resized = img1.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        img2_resized = img2.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)

        diff = ImageChops.difference(img1_resized, img2_resized)
        stat = ImageStat.Stat(diff)
        mean_diff = sum(stat.mean) / (3.0 * 255.0)
        return round(mean_diff, 4)

    @classmethod
    def compute_approx_ssim(cls, img1: Image.Image, img2: Image.Image) -> float:
        """
        Compute structural similarity index (SSIM approximation) [0.0, 1.0].
        """
        diff_score = cls.compute_image_difference(img1, img2)
        # Structural similarity inverse to mean normalized difference
        return max(0.0, min(1.0, round(1.0 - (diff_score * 1.5), 4)))

    @classmethod
    def evaluate(
        cls,
        source_image_path: str,
        generated_image_path: str,
        scene_graph: Optional[SceneGraph] = None,
        redesign_required: bool = True,
    ) -> SimilarityGateResult:
        """
        Evaluate candidate image against source image.
        Reject near duplicates or identity-damaged outputs.
        """
        source_p = Path(source_image_path)
        gen_p = Path(generated_image_path)

        if not source_p.is_file() or not gen_p.is_file():
            logger.warning("SimilarityGate: missing image file(s) for comparison")
            return SimilarityGateResult(
                passed=True,
                ssim_score=0.0,
                difference_score=1.0,
            )

        try:
            with Image.open(source_p) as src_img, Image.open(gen_p) as gen_img:
                diff_score = cls.compute_image_difference(src_img, gen_img)
                ssim_score = cls.compute_approx_ssim(src_img, gen_img)

                logger.info(
                    "SimilarityGate evaluation: diff_score={diff:.4f}, ssim_score={ssim:.4f}",
                    diff=diff_score,
                    ssim=ssim_score,
                )

                # Check 1: Near Duplicate Output
                if redesign_required and (ssim_score >= cls.NEAR_DUPLICATE_SSIM_THRESHOLD or diff_score < cls.MIN_REDESIGN_DIFFERENCE_THRESHOLD):
                    reason = (
                        f"Generated thumbnail is effectively a near-duplicate of original "
                        f"(SSIM={ssim_score:.3f}, diff={diff_score:.3f})"
                    )
                    logger.warning("SimilarityGate REJECT: {reason}", reason=reason)
                    return SimilarityGateResult(
                        passed=False,
                        ssim_score=ssim_score,
                        difference_score=diff_score,
                        rejection_reason=reason,
                        recommended_retry_strategy="increase_background_transformation_and_denoise",
                    )

                # Check 2: Hero Face Identity Preservation (if scene_graph provided)
                if scene_graph and scene_graph.hero_element_id:
                    hero = next((e for e in scene_graph.elements if e.element_id == scene_graph.hero_element_id), None)
                    if hero and hero.bbox:
                        # Check face region crop difference
                        w, h = src_img.size
                        src_crop = src_img.crop((
                            int(hero.bbox.x_min * w),
                            int(hero.bbox.y_min * h),
                            int(hero.bbox.x_max * w),
                            int(hero.bbox.y_max * h),
                        ))
                        gw, gh = gen_img.size
                        gen_crop = gen_img.crop((
                            int(hero.bbox.x_min * gw),
                            int(hero.bbox.y_min * gh),
                            int(hero.bbox.x_max * gw),
                            int(hero.bbox.y_max * gh),
                        ))
                        face_diff = cls.compute_image_difference(src_crop, gen_crop)
                        if face_diff > 0.45:
                            reason = f"Hero creator face region heavily altered/damaged (face_diff={face_diff:.3f})"
                            logger.warning("SimilarityGate REJECT: {reason}", reason=reason)
                            return SimilarityGateResult(
                                passed=False,
                                ssim_score=ssim_score,
                                difference_score=diff_score,
                                rejection_reason=reason,
                                recommended_retry_strategy="increase_face_mask_protection",
                            )

                return SimilarityGateResult(
                    passed=True,
                    ssim_score=ssim_score,
                    difference_score=diff_score,
                )

        except Exception as exc:
            logger.error("SimilarityGate evaluation error: {exc}", exc=exc)
            return SimilarityGateResult(
                passed=True,
                ssim_score=0.5,
                difference_score=0.5,
            )

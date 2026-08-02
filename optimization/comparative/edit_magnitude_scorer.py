"""
edit_magnitude_scorer.py
========================

Measures structural divergence and identity drift to flag over-edited thumbnails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np

from pydantic import BaseModel, ConfigDict
from evaluation.quality.determinism_checker import compute_ssim
from modules.models import QualityAssuranceReport
from optimization.config import (
    OPTIMIZATION_MAX_IDENTITY_DRIFT,
    OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY,
)


class EditMagnitudeScore(BaseModel):
    """Structural divergence and identity drift evaluation."""

    model_config = ConfigDict(frozen=True)

    structural_similarity: float
    identity_drift: float
    over_edited: bool


class EditMagnitudeScorer:
    """Evaluates edit magnitude and over-editing status."""

    def __init__(
        self,
        min_ssim: float = OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY,
        max_identity_drift: float = OPTIMIZATION_MAX_IDENTITY_DRIFT,
    ) -> None:
        self.min_ssim = min_ssim
        self.max_identity_drift = max_identity_drift

    def score(
        self,
        source_image_path: str | Path | None,
        candidate_image_path: str | Path | None,
        qa_report: QualityAssuranceReport,
    ) -> EditMagnitudeScore:
        """
        Compute structural similarity and identity drift.
        """
        ssim_val = 0.50
        if source_image_path and candidate_image_path:
            src_p = Path(source_image_path)
            cand_p = Path(candidate_image_path)
            if src_p.exists() and cand_p.exists():
                try:
                    with Image.open(src_p) as img_a, Image.open(cand_p) as img_b:
                        # Resize to standard size for SSIM computation
                        img_a_resized = img_a.convert("L").resize((256, 256))
                        img_b_resized = img_b.convert("L").resize((256, 256))
                        ssim_val = float(compute_ssim(img_a_resized, img_b_resized))
                except Exception:
                    ssim_val = 0.50

        # Identity drift derived from qa_report.identity_score (0-1)
        identity_score = qa_report.identity_score if hasattr(qa_report, "identity_score") else 0.80
        identity_drift = float(max(0.0, 1.0 - identity_score))

        # Flag over-edited if SSIM is too low AND identity drift exceeds threshold
        over_edited = bool(ssim_val < self.min_ssim and identity_drift > self.max_identity_drift)

        return EditMagnitudeScore(
            structural_similarity=ssim_val,
            identity_drift=identity_drift,
            over_edited=over_edited,
        )

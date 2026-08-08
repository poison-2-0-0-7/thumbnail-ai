"""
failure_analyzer.py
===================

FailureAnalyzer Implementation for Phase 6.1 Benchmark Framework.
Categorizes benchmark evaluation failures across 7 standardized categories.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from thumbnail_intelligence.benchmarks.models import FailureCategory
from thumbnail_intelligence.evaluation.models import EvaluationResult

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """Categorizes and analyzes benchmark execution failures."""

    @staticmethod
    def categorize_failure(
        error_message: Optional[str] = None,
        eval_result: Optional[EvaluationResult] = None,
    ) -> Tuple[FailureCategory, str]:
        """Categorize failure given error message and/or EvaluationResult."""

        if error_message:
            err_lower = error_message.lower()
            if "out of memory" in err_lower or "cuda" in err_lower or "vram" in err_lower or "oom" in err_lower:
                return FailureCategory.OOM_FAILURES, f"GPU memory allocation failed: {error_message}"
            if "face" in err_lower or "subject" in err_lower or "matter" in err_lower:
                return FailureCategory.POOR_FACE_EXTRACTION, f"Face extraction / subject segmentation failed: {error_message}"
            if "typography" in err_lower or "font" in err_lower or "text" in err_lower:
                return FailureCategory.TYPOGRAPHY_FAILURES, f"Typography rendering failed: {error_message}"
            if "background" in err_lower or "inpaint" in err_lower:
                return FailureCategory.BACKGROUND_FAILURES, f"Background synthesis failed: {error_message}"

            return FailureCategory.PIPELINE_FAILURES, f"Pipeline execution error: {error_message}"

        if eval_result:
            m = eval_result.metrics
            if m.get("text_readability") and m["text_readability"].score < 50.0:
                return FailureCategory.TYPOGRAPHY_FAILURES, "Text readability score (<50.0) below acceptable threshold."
            if m.get("font_contrast") and m["font_contrast"].score < 50.0:
                return FailureCategory.TYPOGRAPHY_FAILURES, "Font contrast ratio (<50.0) below WCAG requirements."
            if m.get("face_visibility") and m["face_visibility"].score < 50.0:
                return FailureCategory.POOR_FACE_EXTRACTION, "Face visibility score (<50.0) indicates obscured or missing hero face."
            if m.get("color_contrast") and m["color_contrast"].score < 50.0:
                return FailureCategory.LOW_CONTRAST, "Global color dynamic range (<50.0) is too low."
            if m.get("subject_saliency") and m["subject_saliency"].score < 50.0:
                return FailureCategory.LOW_CONTRAST, "Subject saliency signal-to-noise ratio (<50.0) is insufficient."
            if m.get("visual_hierarchy") and m["visual_hierarchy"].score < 50.0:
                return FailureCategory.WEAK_COMPOSITION, "Visual hierarchy layering (<50.0) is broken."
            if m.get("background_clutter") and m["background_clutter"].score < 50.0:
                return FailureCategory.BACKGROUND_FAILURES, "Background clutter density (<50.0) is excessively noisy."

            if eval_result.overall_score < 60.0:
                return FailureCategory.WEAK_COMPOSITION, f"Overall quality score ({eval_result.overall_score:.1f}) below minimum threshold (60.0)."

        return FailureCategory.NONE, "No failure detected."

"""
acceptance_gate.py
===================

Final acceptance gate evaluating thumbnail candidate readiness before shipment.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict
from loguru import logger

from modules.models import QualityAssuranceReport
from optimization.config import OPTIMIZATION_ACCEPTANCE_REPORT_ONLY
from optimization.comparative.beats_original_scorer import BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScore


class AcceptanceResult(BaseModel):
    """Result of acceptance gate evaluation."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    candidate_index: int
    accepted: bool
    reasons_rejected: list[str] = []


class AcceptanceGate:
    """Gates candidate thumbnail shipment based on quality and comparative criteria."""

    def __init__(self, report_only: bool = OPTIMIZATION_ACCEPTANCE_REPORT_ONLY) -> None:
        self.report_only = report_only

    def evaluate(
        self,
        video_id: str,
        candidate_index: int,
        qa_report: Optional[QualityAssuranceReport] = None,
        verdict: Optional[BeatsOriginalVerdict] = None,
        edit_score: Optional[EditMagnitudeScore] = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """
        Evaluate candidate against acceptance criteria.
        """
        reasons: list[str] = []

        if qa_report:
            if not qa_report.hard_gate_passed:
                reasons.append("hard_gate_failed")
            if qa_report.identity_score < 0.40 and qa_report.identity_score > 0.0:
                reasons.append("identity_drift_exceeded")
            if qa_report.text_safe_zone_score < 0.40 and qa_report.text_safe_zone_score > 0.0:
                reasons.append("text_obstruction_detected")
            if qa_report.composition_score < 0.40 and qa_report.composition_score > 0.0:
                reasons.append("incorrect_composition")

        if edit_score and edit_score.over_edited:
            reasons.append("over_edited")

        if verdict and not verdict.beats_original:
            reasons.append("did_not_beat_original")

        # Module 10 Creator Style Learning validation
        if kwargs.get("profile_established", False) and kwargs.get("style_similarity_score") is not None:
            sim_score = kwargs["style_similarity_score"]
            drift = kwargs.get("drift_detected", False)
            threshold = kwargs.get("similarity_threshold", 0.75)
            if sim_score < threshold and not drift:
                reasons.append("style_identity_lost")

        is_accepted = len(reasons) == 0


        if self.report_only and not is_accepted:
            logger.warning(
                "AcceptanceGate failed for {vid} cand {idx} (reasons: {reasons}) but running in report-only mode",
                vid=video_id,
                idx=candidate_index,
                reasons=reasons,
            )
            is_accepted = True

        return AcceptanceResult(
            video_id=video_id,
            candidate_index=candidate_index,
            accepted=is_accepted,
            reasons_rejected=reasons,
        )

"""
tests/test_optimization_acceptance_gate.py
============================================

Unit tests for AcceptanceGate decision logic.
"""

from modules.models import QualityAssuranceReport
from optimization.comparative.beats_original_scorer import BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScore
from optimization.validation.acceptance_gate import AcceptanceGate


def test_acceptance_gate_accepted():
    gate = AcceptanceGate(report_only=False)
    qa = QualityAssuranceReport(
        resolution_passed=True,
        file_integrity_passed=True,
        safety_passed=True,
        identity_score=0.8,
        composition_score=0.8,
        text_safe_zone_score=0.8,
        overall_score=0.8,
        hard_gate_passed=True,
    )
    verdict = BeatsOriginalVerdict(video_id="v1", candidate_index=0, baseline_overall_score=0.6, candidate_overall_score=0.8, delta=0.2, beats_original=True)
    edit = EditMagnitudeScore(structural_similarity=0.7, identity_drift=0.2, over_edited=False)

    res = gate.evaluate("v1", 0, qa, verdict, edit)
    assert res.accepted is True
    assert len(res.reasons_rejected) == 0


def test_acceptance_gate_rejected():
    gate = AcceptanceGate(report_only=False)
    qa = QualityAssuranceReport(
        resolution_passed=True,
        file_integrity_passed=True,
        safety_passed=True,
        identity_score=0.2,
        hard_gate_passed=True,
    )
    verdict = BeatsOriginalVerdict(video_id="v1", candidate_index=0, baseline_overall_score=0.8, candidate_overall_score=0.6, delta=-0.2, beats_original=False)
    edit = EditMagnitudeScore(structural_similarity=0.2, identity_drift=0.8, over_edited=True)

    res = gate.evaluate("v1", 0, qa, verdict, edit)
    assert res.accepted is False
    assert "identity_drift_exceeded" in res.reasons_rejected
    assert "over_edited" in res.reasons_rejected
    assert "did_not_beat_original" in res.reasons_rejected

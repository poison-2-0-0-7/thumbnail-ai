"""
tests/test_optimization_trace_porce.py
=======================================

Unit tests for trace extensions and PORCE diagnostic rules.
"""

from observability.models import GenerationTraceRecord
from observability.facts.models import TraceFacts
from observability.diagnostics.rules.optimization_diagnostic_rules import (
    GeneratedThumbnailDidNotBeatOriginalRule,
    OptimizationSelectionDisagreementRule,
    OverEditedAcceptedRule,
)
from optimization.comparative.baseline_scorer import BaselineScore
from optimization.comparative.beats_original_scorer import BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScore
from optimization.orchestration.optimization_loop import OptimizationLoopResult
from optimization.orchestration.winner_selector import OptimizedSelection
from optimization.validation.acceptance_gate import AcceptanceResult
from optimization.trace.trace_extension import attach_optimization_to_trace


def test_attach_optimization_to_trace():
    base_record = GenerationTraceRecord(video_id="v1")
    loop_result = OptimizationLoopResult(
        video_id="v1",
        baseline_score=BaselineScore(video_id="v1", overall_score=0.5, dimension_scores={}, source_path="s.png"),
        generation_result=None,
        selection=OptimizedSelection(video_id="v1", module7_selected_index=0, optimization_selected_index=0, selection_agrees=True, reason="ok"),
        acceptance=AcceptanceResult(video_id="v1", candidate_index=0, accepted=True),
        verdicts=[BeatsOriginalVerdict(video_id="v1", candidate_index=0, baseline_overall_score=0.5, candidate_overall_score=0.7, delta=0.2, beats_original=True)],
        edit_scores=[EditMagnitudeScore(structural_similarity=0.8, identity_drift=0.1, over_edited=False)],
        total_attempts=1,
    )

    extended = attach_optimization_to_trace(base_record, loop_result)

    assert extended.baseline_score == 0.5
    assert extended.beats_original is True
    assert extended.winning_candidate_index == 0
    assert extended.selection_agreed is True
    assert extended.edit_magnitude == 0.8
    assert extended.over_edited is False


def test_porce_optimization_rules():
    r1 = GeneratedThumbnailDidNotBeatOriginalRule()
    r2 = OverEditedAcceptedRule()
    r3 = OptimizationSelectionDisagreementRule()

    facts = TraceFacts(
        video_id="v1",
        extracted_at="now",
        beats_original=False,
        over_edited=True,
        selection_agreed=False,
        module7_selected_index=0,
        winning_candidate_index=1,
    )

    f1 = r1.check(facts)
    assert f1 is not None
    assert f1.finding_id == "RULE-OPT-01"
    assert f1.severity == "FAIL"

    f2 = r2.check(facts)
    assert f2 is not None
    assert f2.finding_id == "RULE-OPT-02"
    assert f2.severity == "FAIL"

    f3 = r3.check(facts)
    assert f3 is not None
    assert f3.finding_id == "RULE-OPT-03"
    assert f3.severity == "INFO"

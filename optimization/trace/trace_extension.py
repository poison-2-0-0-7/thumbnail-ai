"""
trace_extension.py
==================

Extends GenerationTraceRecord with Optimization Layer execution data.
"""

from __future__ import annotations

from typing import Optional
from observability.models import GenerationTraceRecord
from optimization.orchestration.optimization_loop import OptimizationLoopResult


def attach_optimization_to_trace(
    record: GenerationTraceRecord,
    loop_result: OptimizationLoopResult,
) -> GenerationTraceRecord:
    """
    Produce a new GenerationTraceRecord with populated optimization fields.
    """
    winning_idx = loop_result.selection.optimization_selected_index
    
    # Extract candidate scores list
    cand_scores = [v.candidate_overall_score for v in loop_result.verdicts]
    
    winning_verdict = next((v for v in loop_result.verdicts if v.candidate_index == winning_idx), None)
    winning_edit = loop_result.edit_scores[winning_idx] if winning_idx is not None and winning_idx < len(loop_result.edit_scores) else None

    beats_orig = winning_verdict.beats_original if winning_verdict else False
    edit_mag = winning_edit.structural_similarity if winning_edit else None
    over_ed = winning_edit.over_edited if winning_edit else None

    # Model copy with updated fields preserving immutability
    return record.model_copy(
        update={
            "baseline_score": loop_result.baseline_score.overall_score,
            "candidate_scores": cand_scores,
            "beats_original": beats_orig,
            "winning_candidate_index": winning_idx,
            "module7_selected_index": loop_result.selection.module7_selected_index,
            "selection_agreed": loop_result.selection.selection_agrees,
            "edit_magnitude": edit_mag,
            "over_edited": over_ed,
            "optimization_strategy_used": "optimization_loop_v1",
            "retry_attempt_count": max(0, loop_result.total_attempts - 1),
        }
    )

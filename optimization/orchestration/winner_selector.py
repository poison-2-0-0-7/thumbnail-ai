"""
winner_selector.py
==================

Selects winning thumbnail candidate by unifying Module 7 QA reports, head-to-head
BeatsOriginal verdicts, and EditMagnitude scores.
"""

from __future__ import annotations

from typing import Optional, Sequence
from pydantic import BaseModel, ConfigDict
from loguru import logger

from modules.models import CandidateScore, QualityAssuranceReport
from optimization.comparative.beats_original_scorer import BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScore
from optimization.orchestration.interfaces import IWinnerSelector


class OptimizedSelection(BaseModel):
    """Selection verdict produced by the Optimization Layer."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    module7_selected_index: Optional[int]
    optimization_selected_index: Optional[int]
    selection_agrees: bool
    reason: str


class WinnerSelector(IWinnerSelector):
    """Selects winner candidate using comparative quality analysis."""

    def select(
        self,
        video_id: str,
        candidate_scores: Sequence[CandidateScore],
        qa_reports: Sequence[QualityAssuranceReport],
        verdicts: Sequence[BeatsOriginalVerdict],
        edit_scores: Sequence[EditMagnitudeScore],
        budget_exhausted: bool = False,
    ) -> OptimizedSelection:
        # Determine Module 7's own selected index
        module7_selected: Optional[int] = None
        for cs in candidate_scores:
            if cs.selected:
                module7_selected = cs.candidate_index
                break
        if module7_selected is None and candidate_scores:
            # Fallback to rank 1 candidate
            rank1 = next((cs for cs in candidate_scores if cs.rank == 1), candidate_scores[0])
            module7_selected = rank1.candidate_index

        # Filter candidates that pass hard gates and beat original
        winning_candidates: list[tuple[int, float, float]] = []  # (index, overall_score, delta)

        for i, verdict in enumerate(verdicts):
            cand_idx = verdict.candidate_index
            qa = qa_reports[i] if i < len(qa_reports) else None
            edit = edit_scores[i] if i < len(edit_scores) else None

            # Hard gate pass check
            gate_passed = qa.hard_gate_passed if qa else True
            not_over_edited = not edit.over_edited if edit else True

            if gate_passed and not_over_edited and verdict.beats_original:
                winning_candidates.append((cand_idx, verdict.candidate_overall_score, verdict.delta))

        opt_selected: Optional[int] = None
        reason: str = ""

        if winning_candidates:
            # Pick candidate with highest delta
            winning_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
            opt_selected = winning_candidates[0][0]
            reason = f"Candidate {opt_selected} selected: passed gates and beat original by delta={winning_candidates[0][2]:.4f}"
        elif budget_exhausted:
            # Fallback when retry budget exhausted: pick best-scoring candidate among those passing hard gate
            eligible: list[tuple[int, float]] = []
            for i, verdict in enumerate(verdicts):
                cand_idx = verdict.candidate_index
                qa = qa_reports[i] if i < len(qa_reports) else None
                if qa and qa.hard_gate_passed:
                    eligible.append((cand_idx, verdict.candidate_overall_score))

            if eligible:
                eligible.sort(key=lambda x: x[1], reverse=True)
                opt_selected = eligible[0][0]
                reason = f"Budget exhausted: candidate {opt_selected} selected as best available passing hard gate (did not beat original)"
            else:
                opt_selected = module7_selected
                reason = f"Budget exhausted and no candidate passed hard gate: falling back to Module 7 selected candidate {opt_selected}"
        else:
            opt_selected = None
            reason = "No candidate passed hard gate and beat original baseline margin"

        selection_agrees = (module7_selected == opt_selected) if (opt_selected is not None) else False

        logger.info(
            "Winner selection for {vid}: M7={m7}, Opt={opt}, Agrees={agrees}, Reason={reason}",
            vid=video_id,
            m7=module7_selected,
            opt=opt_selected,
            agrees=selection_agrees,
            reason=reason,
        )

        return OptimizedSelection(
            video_id=video_id,
            module7_selected_index=module7_selected,
            optimization_selected_index=opt_selected,
            selection_agrees=selection_agrees,
            reason=reason,
        )

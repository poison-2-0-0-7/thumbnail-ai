"""
optimization/orchestration/interfaces.py
=========================================

Interfaces for optimization orchestration and winner selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from pydantic import BaseModel

from modules.models import CandidateScore, QualityAssuranceReport
from optimization.comparative.beats_original_scorer import BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScore


class IWinnerSelector(ABC):
    """Interface for winner candidate selection logic."""

    @abstractmethod
    def select(
        self,
        video_id: str,
        candidate_scores: Sequence[CandidateScore],
        qa_reports: Sequence[QualityAssuranceReport],
        verdicts: Sequence[BeatsOriginalVerdict],
        edit_scores: Sequence[EditMagnitudeScore],
        budget_exhausted: bool = False,
    ) -> BaseModel:
        """Select the winning candidate index."""
        pass

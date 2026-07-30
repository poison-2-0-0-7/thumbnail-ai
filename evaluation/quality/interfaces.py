"""
evaluation/quality/interfaces.py
=================================

Interfaces for PVQEF quality evaluation scorers.
"""

from abc import ABC, abstractmethod

from modules.models import DimensionScore
from .scoring_context import QualityScoringContext


class IQualityScorer(ABC):
    """Interface for quality evaluation dimension scorers."""

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Name of the quality dimension, e.g. 'face_preservation'."""

    @abstractmethod
    def score(self, context: QualityScoringContext) -> DimensionScore:
        """Compute score for one generated thumbnail in context."""

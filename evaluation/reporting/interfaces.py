"""
evaluation/reporting/interfaces.py
===================================

Interfaces for PVQEF report rendering.
"""

from abc import ABC, abstractmethod

from modules.models import PipelineRunReport


class IReportRenderer(ABC):
    """Interface for rendering a PipelineRunReport into a target string format."""

    @abstractmethod
    def render(self, report: PipelineRunReport) -> str:
        """Render report into formatted string representation."""

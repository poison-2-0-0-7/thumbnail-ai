"""
evaluation/batch/interfaces.py
===============================

Interface for PVQEF batch execution.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from modules.models import PipelineRunReport


class IBatchExecutor(ABC):
    """Interface for batch evaluation orchestrators."""

    @abstractmethod
    def run_batch(self, csv_path: Path, max_concurrency: int = 1) -> PipelineRunReport:
        """Execute concurrency-bounded batch evaluation across multiple creators."""

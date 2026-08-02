"""
observability/reporting/interfaces.py
======================================

Abstract Base Classes (ABCs) for Root Cause Report components in PORCE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from observability.diagnostics.models import FindingCollection
from observability.facts.models import FactCollection
from observability.models import PipelineTrace
from observability.reporting.models import RootCauseReport


class IRootCauseAssembler(ABC):
    """Interface for assembling a canonical RootCauseReport."""

    @abstractmethod
    def assemble(
        self,
        video_id: str,
        pipeline_trace: PipelineTrace,
        finding_collection: FindingCollection,
        fact_collection: Optional[FactCollection] = None,
    ) -> RootCauseReport:
        """Assemble a RootCauseReport from trace, findings, and facts."""
        pass


class IRootCausePersistence(ABC):
    """Interface for persisting and loading RootCauseReport objects."""

    @abstractmethod
    def save(self, report: RootCauseReport) -> Path:
        """Atomically persist RootCauseReport to disk."""
        pass

    @abstractmethod
    def load(self, video_id: str) -> Optional[RootCauseReport]:
        """Load RootCauseReport for video_id from disk."""
        pass

"""
observability/interfaces.py
===========================

Abstract Base Classes (ABCs) for PORCE components.
"""

from abc import ABC, abstractmethod

from observability.models import ArtifactIndex, LogLineRef, PipelineTrace


class IArtifactCollector(ABC):
    """Interface for discovering and indexing artifacts for a video_id."""

    @abstractmethod
    def collect(self, video_id: str) -> ArtifactIndex:
        """Discover and index all artifacts for the given video_id."""
        pass


class ILogCorrelator(ABC):
    """Interface for scanning and correlating log entries by video_id."""

    @abstractmethod
    def correlate(self, video_id: str) -> list[LogLineRef]:
        """Grep module logs for video_id and return chronologically ordered log lines."""
        pass


class ITraceAssembler(ABC):
    """Interface for assembling a complete PipelineTrace."""

    @abstractmethod
    def assemble(self, video_id: str) -> PipelineTrace:
        """Assemble the complete pipeline trace for the given video_id."""
        pass

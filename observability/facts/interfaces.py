"""
observability/facts/interfaces.py
==================================

Abstract Base Classes (ABCs) for Facts Extraction Layer components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from observability.facts.models import FactCollection, TraceFacts
from observability.models import PipelineTrace


class IFactExtractor(ABC):
    """Interface for extracting deterministic facts from a PipelineTrace."""

    @abstractmethod
    def extract(self, trace: PipelineTrace) -> FactCollection:
        """Extract a FactCollection from the given PipelineTrace."""
        pass


class IFactSerializer(ABC):
    """Interface for serializing and deserializing FactCollection objects."""

    @abstractmethod
    def serialize(self, collection: FactCollection) -> str:
        """Serialize FactCollection to JSON string."""
        pass

    @abstractmethod
    def deserialize(self, json_str: str) -> FactCollection:
        """Deserialize JSON string to FactCollection."""
        pass


class IFactPersistence(ABC):
    """Interface for persisting and loading FactCollection objects."""

    @abstractmethod
    def save(self, collection: FactCollection) -> Path:
        """Atomically persist FactCollection to disk."""
        pass

    @abstractmethod
    def load(self, video_id: str) -> Optional[FactCollection]:
        """Load FactCollection for video_id from disk."""
        pass

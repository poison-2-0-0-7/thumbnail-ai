"""
optimization/comparative/interfaces.py
=======================================

Interfaces and protocols for comparative scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class IComparativeScorer(ABC):
    """Abstract interface for comparative quality scoring."""

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Name of the comparative scoring dimension."""
        pass

    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Execute comparative scoring and return a Pydantic result model."""
        pass

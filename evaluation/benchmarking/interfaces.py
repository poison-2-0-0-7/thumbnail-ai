"""
evaluation/benchmarking/interfaces.py
======================================

Interfaces for historical benchmark store and regression rules.
"""

from abc import ABC, abstractmethod

from modules.models import BenchmarkRecord, RegressionFinding


class IHistoricalStore(ABC):
    """Interface for append-only historical benchmark storage."""

    @abstractmethod
    def append(self, record: BenchmarkRecord) -> None:
        """Append one BenchmarkRecord row to historical store."""

    @abstractmethod
    def load_recent(self, n: int = 5) -> list[BenchmarkRecord]:
        """Load up to N most recent BenchmarkRecord items."""


class IRegressionRule(ABC):
    """Interface for an independent statistical regression check."""

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Name of the regression rule, e.g. 'overall_score_drop'."""

    @abstractmethod
    def check(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> RegressionFinding | None:
        """Check for regression between current run and baseline."""

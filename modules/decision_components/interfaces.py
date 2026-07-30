"""
interfaces.py
=============

Abstract base classes defining component contracts for Module 9 (AI Decision Engine).
Follows the VRE component interface pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from modules.models import CandidateDecision, DecisionManifest, ResolvedDecision


class IRuleEngine(ABC):
    """Abstract contract for rule-based decision generation."""

    @abstractmethod
    def evaluate(self, bundle: Any) -> list[CandidateDecision]:
        """Evaluate rules against input bundle and emit candidate decisions."""


class IAmbiguityRouter(ABC):
    """Abstract contract for selecting candidates requiring LLM adjudication."""

    @abstractmethod
    def select(
        self, candidates: list[CandidateDecision]
    ) -> tuple[list[CandidateDecision], list[CandidateDecision]]:
        """Split candidates into (confident_candidates, needs_llm_review_candidates)."""


class ILLMReasoner(ABC):
    """Abstract contract for LLM adjudication of ambiguous candidate decisions."""

    @abstractmethod
    def adjudicate(
        self, candidates: list[CandidateDecision], bundle: Any
    ) -> list[CandidateDecision]:
        """Adjudicate low-confidence or conflicting candidates using LLM reasoning."""


class IConflictResolver(ABC):
    """Abstract contract for candidate decision conflict resolution and ranking."""

    @abstractmethod
    def resolve(self, candidates: list[CandidateDecision]) -> list[ResolvedDecision]:
        """Apply priority hierarchy and mutual exclusion to generate final resolved decisions."""


class IDecisionValidator(ABC):
    """Abstract contract for validation of resolved decisions."""

    @abstractmethod
    def validate(self, decisions: list[ResolvedDecision]) -> dict[str, Any]:
        """Validate structural and business rules of resolved decisions."""


class IManifestAssembler(ABC):
    """Abstract contract for assembling and partitioning decision manifests."""

    @abstractmethod
    def build(
        self,
        video_id: str,
        source_image_path: str,
        source_image_hash: str,
        decisions: list[ResolvedDecision],
        validation_report: dict[str, Any],
        duration_seconds: float,
    ) -> DecisionManifest:
        """Assemble the complete DecisionManifest instance."""


class IDecisionCache(ABC):
    """Abstract contract for loading and persisting decision manifests."""

    @abstractmethod
    def load(self, video_id: str) -> Optional[DecisionManifest]:
        """Load cached manifest if present and valid."""

    @abstractmethod
    def save(self, manifest: DecisionManifest) -> None:
        """Persist manifest atomically to disk."""

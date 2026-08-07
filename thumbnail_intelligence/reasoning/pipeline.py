"""
pipeline.py
===========

High-level pipeline facade orchestrating the flow:
NormalizedEvidenceGraph -> ReasoningCoordinator -> Registered Reasoners -> Collected Outputs -> ReasoningContext
"""

from __future__ import annotations

from typing import Optional

from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.reasoning.config import ReasoningConfig
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry


class ReasoningPipeline:
    """
    High-level orchestration pipeline for the Strategic Reasoning subsystem.
    Encapsulates registry lifecycle, coordinator orchestration, and context compilation.
    """

    def __init__(
        self,
        coordinator: Optional[ReasoningCoordinator] = None,
        config: Optional[ReasoningConfig] = None,
    ) -> None:
        if coordinator is not None:
            self.coordinator: ReasoningCoordinator = coordinator
            if config is not None:
                self.coordinator.config = config
        else:
            cfg = config if config is not None else ReasoningConfig()
            self.coordinator = ReasoningCoordinator(config=cfg)

    @classmethod
    def create_default(cls, config: Optional[ReasoningConfig] = None) -> ReasoningPipeline:
        """Create a default ReasoningPipeline instance."""
        return cls(config=config)

    @classmethod
    def from_registry(
        cls,
        registry: ReasonerRegistry,
        config: Optional[ReasoningConfig] = None,
    ) -> ReasoningPipeline:
        """Create a ReasoningPipeline backed by a pre-configured ReasonerRegistry."""
        coord = ReasoningCoordinator(registry=registry, config=config)
        return cls(coordinator=coord, config=config)

    @property
    def registry(self) -> ReasonerRegistry:
        """Convenience property accessing the underlying ReasonerRegistry."""
        return self.coordinator.registry

    @property
    def config(self) -> ReasoningConfig:
        """Convenience property accessing the pipeline configuration."""
        return self.coordinator.config

    def run(self, graph: NormalizedEvidenceGraph) -> ReasoningContext:
        """
        Execute the end-to-end strategic reasoning pipeline on the input evidence graph.

        Args:
            graph: Validated, conflict-resolved NormalizedEvidenceGraph.

        Returns:
            Fully populated, validated ReasoningContext.
        """
        return self.coordinator.coordinate(graph)

    def run_with_context(
        self,
        graph: NormalizedEvidenceGraph,
        base_context: ReasoningContext,
    ) -> ReasoningContext:
        """
        Execute the strategic reasoning pipeline augmenting an existing base context.

        Args:
            graph: Validated, conflict-resolved NormalizedEvidenceGraph.
            base_context: Pre-populated base ReasoningContext.

        Returns:
            Augmented ReasoningContext.
        """
        return self.coordinator.coordinate(graph, initial_context=base_context)

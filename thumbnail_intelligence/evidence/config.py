"""
config.py
=========

Configuration specifications for the Evidence Normalization Engine.
Defines thresholds for confidence floors, graph node limits, edge decay factors,
clustering similarity, and conflict resolution precedence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass(frozen=True)
class EvidenceSourcePriorities:
    """
    Base priority weights for different evidence origins.
    Brand constraints and verified creator rules carry highest priority,
    followed by direct historical baseline metrics and established archetypes.
    """

    brand_rule: float = 1.00
    identity_constraint: float = 1.00
    creator_profile: float = 0.95
    historical_thumbnail: float = 0.90
    archetype_example: float = 0.85
    thumbnail_pattern: float = 0.80
    visual_pattern: float = 0.75
    competitor_thumbnail: float = 0.70
    design_pattern: float = 0.70
    general_knowledge: float = 0.50

    def to_dict(self) -> Dict[str, float]:
        return {
            "brand_rule": self.brand_rule,
            "identity_constraint": self.identity_constraint,
            "creator_profile": self.creator_profile,
            "historical_thumbnail": self.historical_thumbnail,
            "archetype_example": self.archetype_example,
            "thumbnail_pattern": self.thumbnail_pattern,
            "visual_pattern": self.visual_pattern,
            "competitor_thumbnail": self.competitor_thumbnail,
            "design_pattern": self.design_pattern,
            "general_knowledge": self.general_knowledge,
        }


@dataclass
class EvidenceNormalizationConfig:
    """
    Master configuration for the Evidence Normalization pipeline.
    """

    min_confidence_threshold: float = 0.20
    high_confidence_threshold: float = 0.80
    confidence_decay_factor: float = 0.90
    duplicate_similarity_threshold: float = 0.95
    cluster_similarity_threshold: float = 0.70
    max_graph_nodes: int = 200
    allow_unresolved_conflicts: bool = False
    conflict_resolution_strategy: Literal[
        "hybrid", "brand_dominance", "highest_confidence", "most_recent"
    ] = "hybrid"
    enable_clustering: bool = True
    enable_provenance_tracing: bool = True
    source_priorities: EvidenceSourcePriorities = field(default_factory=EvidenceSourcePriorities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "confidence_decay_factor": self.confidence_decay_factor,
            "duplicate_similarity_threshold": self.duplicate_similarity_threshold,
            "cluster_similarity_threshold": self.cluster_similarity_threshold,
            "max_graph_nodes": self.max_graph_nodes,
            "allow_unresolved_conflicts": self.allow_unresolved_conflicts,
            "conflict_resolution_strategy": self.conflict_resolution_strategy,
            "enable_clustering": self.enable_clustering,
            "enable_provenance_tracing": self.enable_provenance_tracing,
            "source_priorities": self.source_priorities.to_dict(),
        }

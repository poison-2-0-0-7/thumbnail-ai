"""
evidence
========

Evidence Normalization Engine for Thumbnail AI.
Transforms retrieved multi-domain evidence bundles into a grounded, validated,
conflict-resolved directed evidence graph (NormalizedEvidenceGraph).

Provides:
- Grounding and provenance tracking (no anonymous evidence)
- Deterministic conflict detection and resolution
- Multi-signal confidence propagation with edge decay
- Domain and semantic evidence clustering
- Directed graph representations with supporting and contradicting edges
"""

from __future__ import annotations

from thumbnail_intelligence.evidence.clustering import EvidenceClusterer
from thumbnail_intelligence.evidence.confidence import ConfidencePropagator
from thumbnail_intelligence.evidence.config import (
    EvidenceNormalizationConfig,
    EvidenceSourcePriorities,
)
from thumbnail_intelligence.evidence.conflict_resolution import (
    ConflictDetector,
    ConflictResolver,
)
from thumbnail_intelligence.evidence.exceptions import (
    ClusterError,
    ConflictError,
    CyclicDependencyError,
    EvidenceError,
    GraphError,
    GroundingValidationError,
    NodeNotFoundError,
    NormalizationError,
    ProvenanceError,
    UnresolvableConflictError,
)
from thumbnail_intelligence.evidence.graph import EvidenceGraph
from thumbnail_intelligence.evidence.merger import EvidenceMerger
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    ConflictResolution,
    EvidenceCluster,
    EvidenceConflict,
    EvidenceEdge,
    EvidenceNode,
    EvidenceStatistics,
    EvidenceSummary,
    EvidenceWeight,
    NormalizedEvidence,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.evidence.normalizer import EvidenceNormalizer
from thumbnail_intelligence.evidence.provenance import ProvenanceTracker
from thumbnail_intelligence.evidence.validator import EvidenceGraphValidator
from thumbnail_intelligence.evidence.weighting import EvidenceWeighter

__all__ = [
    # Master Normalizer
    "EvidenceNormalizer",
    # Graph Engine
    "EvidenceGraph",
    "EvidenceGraphValidator",
    "ProvenanceTracker",
    "ConfidencePropagator",
    "EvidenceWeighter",
    "EvidenceMerger",
    "ConflictDetector",
    "ConflictResolver",
    "EvidenceClusterer",
    # Core Models
    "EvidenceNode",
    "EvidenceEdge",
    "EvidenceCluster",
    "ConfidenceScore",
    "EvidenceWeight",
    "ProvenanceRecord",
    "EvidenceConflict",
    "ConflictResolution",
    "EvidenceStatistics",
    "EvidenceSummary",
    "NormalizedEvidence",
    "NormalizedEvidenceGraph",
    # Configuration
    "EvidenceNormalizationConfig",
    "EvidenceSourcePriorities",
    # Exceptions
    "EvidenceError",
    "NormalizationError",
    "GroundingValidationError",
    "GraphError",
    "CyclicDependencyError",
    "NodeNotFoundError",
    "ConflictError",
    "UnresolvableConflictError",
    "ProvenanceError",
    "ClusterError",
]

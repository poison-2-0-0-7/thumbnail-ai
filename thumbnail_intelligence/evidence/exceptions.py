"""
exceptions.py
=============

Structured exception hierarchy for the Evidence Normalization Engine.
Provides domain-specific exceptions for grounding validation, graph integrity,
conflict resolution, cycle detection, and provenance tracking.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from thumbnail_intelligence.knowledge_base.exceptions import KnowledgeBaseError


class EvidenceError(KnowledgeBaseError):
    """Base exception for all errors occurring within the Evidence Normalization Engine."""

    default_error_code: str = "EVIDENCE_ERROR"


class NormalizationError(EvidenceError):
    """Raised when the end-to-end evidence normalization pipeline fails."""

    default_error_code: str = "EVIDENCE_NORMALIZATION_ERROR"


class GroundingValidationError(EvidenceError):
    """Raised when an evidence node lacks valid empirical backing, provenance, or timestamps."""

    default_error_code: str = "EVIDENCE_GROUNDING_VALIDATION_ERROR"


class GraphError(EvidenceError):
    """Raised when evidence graph operations (node insertion, edge lookup, adjacency) fail."""

    default_error_code: str = "EVIDENCE_GRAPH_ERROR"


class CyclicDependencyError(GraphError):
    """Raised when a circular dependency is detected within dependency or derivation edges."""

    default_error_code: str = "EVIDENCE_CYCLIC_DEPENDENCY_ERROR"


class NodeNotFoundError(GraphError):
    """Raised when a referenced node ID is missing from the EvidenceGraph."""

    default_error_code: str = "EVIDENCE_NODE_NOT_FOUND"


class ConflictError(EvidenceError):
    """Raised when evidence conflict detection or resolution encounters an invalid state."""

    default_error_code: str = "EVIDENCE_CONFLICT_ERROR"


class UnresolvableConflictError(ConflictError):
    """Raised when an unresolvable contradiction occurs under strict conflict policies."""

    default_error_code: str = "EVIDENCE_UNRESOLVABLE_CONFLICT_ERROR"


class ProvenanceError(EvidenceError):
    """Raised when provenance verification fails or origin attribution is missing."""

    default_error_code: str = "EVIDENCE_PROVENANCE_ERROR"


class ClusterError(EvidenceError):
    """Raised when evidence clustering fails or generates invalid cluster partitions."""

    default_error_code: str = "EVIDENCE_CLUSTER_ERROR"

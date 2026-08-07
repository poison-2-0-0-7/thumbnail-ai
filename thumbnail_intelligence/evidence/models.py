"""
models.py
=========

Core data models for the Evidence Normalization Engine.
Defines nodes, edges, clusters, confidence scores, weights, provenance records,
conflict structures, and the final master NormalizedEvidenceGraph.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
    EvidenceSourceType,
    KnowledgeEntryType,
    _utc_now_iso,
)
from thumbnail_intelligence.retrieval.evidence_bundle import RetrievedEvidence


class ConfidenceScore(BaseKBModel):
    """
    Structured, propagated confidence score.
    Tracks raw retrieval confidence, source quality factors, and decay hops across edges.
    """

    raw_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Base confidence from retrieval/entry")
    propagated_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Effective confidence after edge decay")
    source_quality_factor: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality multiplier from origin type")
    retrieval_quality_factor: float = Field(default=1.0, ge=0.0, le=1.0, description="Retrieval similarity and ranking factor")
    metadata_quality_factor: float = Field(default=1.0, ge=0.0, le=1.0, description="Completeness of empirical facets")
    decay_hops: int = Field(default=0, ge=0, description="Number of dependency edge hops traversed")
    explanation: str = Field(default="", description="Explainable audit explanation of confidence calculation")


class EvidenceWeight(BaseKBModel):
    """
    Computed weight reflecting the empirical importance and dominance of an evidence node.
    """

    base_weight: float = Field(default=1.0, ge=0.0, description="Base initial weight from scoring")
    effective_weight: float = Field(default=1.0, ge=0.0, description="Final normalized effective weight")
    cluster_support_multiplier: float = Field(default=1.0, ge=0.0, description="Multiplier from supporting cluster size")
    source_priority_multiplier: float = Field(default=1.0, ge=0.0, description="Multiplier from source hierarchy")
    explanation: str = Field(default="", description="Explainable audit explanation of weight calculation")


class ProvenanceRecord(BaseKBModel):
    """
    Immutable audit provenance record tracing the exact retrieval pathway,
    timestamps, and origin attribution. No anonymous evidence.
    """

    origin: str = Field(description="Structured origin (e.g. 'historical_thumbnail:video_123')")
    source_id: str = Field(description="Unique source entity identifier")
    source_type: EvidenceSourceType = Field(description="Grounding classification")
    retrieval_query_id: str = Field(description="Origin retrieval query ID")
    retrieval_reason: str = Field(description="Explainable reason why this evidence was retrieved")
    retrieved_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of retrieval")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of entry creation")
    parent_origins: List[str] = Field(default_factory=list, description="Upstream origin IDs for derived evidence")
    trace_id: str = Field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:12]}", description="Unique trace identifier")


class EvidenceNode(BaseKBModel):
    """
    An individual grounded, validated node in the EvidenceGraph.
    Encapsulates raw retrieved evidence, confidence, weight, provenance, and active status.
    """

    node_id: str = Field(description="Unique deterministic or generated node identifier")
    node_type: KnowledgeEntryType = Field(description="Knowledge classification of this node")
    evidence_item: RetrievedEvidence = Field(description="Underlying retrieved evidence payload")
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore, description="Propagated confidence")
    weight: EvidenceWeight = Field(default_factory=EvidenceWeight, description="Calculated empirical weight")
    provenance: ProvenanceRecord = Field(description="Full provenance and origin tracking")
    cluster_id: Optional[str] = Field(default=None, description="Assigned cluster identifier if clustered")
    is_active: bool = Field(default=True, description="Whether node is active or suppressed by conflict resolution")
    suppression_reason: Optional[str] = Field(default=None, description="Diagnostic reason if node is suppressed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary validated node metadata")

    @field_validator("node_id")
    @classmethod
    def validate_node_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("node_id must not be empty")
        return v.strip()


class EvidenceEdge(BaseKBModel):
    """
    A directed relationship connecting two evidence nodes in the EvidenceGraph.
    """

    edge_id: str = Field(description="Unique deterministic edge identifier")
    source_node_id: str = Field(description="Origin node ID")
    target_node_id: str = Field(description="Destination node ID")
    relation_type: Literal[
        "SUPPORTS",
        "CONTRADICTS",
        "DEPENDS_ON",
        "DERIVED_FROM",
        "PART_OF_CLUSTER",
        "SUPERSEDES",
    ] = Field(description="Semantic nature of the directed edge")
    weight: float = Field(default=1.0, ge=0.0, description="Edge strength or relationship weight")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this relationship")
    explanation: str = Field(default="", description="Explainable description of the relationship")

    @classmethod
    def create(
        cls,
        source_id: str,
        target_id: str,
        relation_type: Literal[
            "SUPPORTS",
            "CONTRADICTS",
            "DEPENDS_ON",
            "DERIVED_FROM",
            "PART_OF_CLUSTER",
            "SUPERSEDES",
        ],
        weight: float = 1.0,
        confidence: float = 1.0,
        explanation: str = "",
    ) -> EvidenceEdge:
        hasher = hashlib.sha256()
        hasher.update(f"{source_id}:{relation_type}:{target_id}".encode("utf-8"))
        edge_id = f"edge_{hasher.hexdigest()[:12]}"
        return cls(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            explanation=explanation,
        )


class EvidenceCluster(BaseKBModel):
    """
    A cluster of semantically or structurally related evidence nodes.
    """

    cluster_id: str = Field(description="Unique cluster identifier")
    cluster_type: str = Field(description="Domain classification (e.g. 'archetype', 'historical', 'brand')")
    node_ids: List[str] = Field(default_factory=list, description="Member node IDs")
    central_node_id: Optional[str] = Field(default=None, description="Exemplar / centroid node ID")
    aggregate_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Weighted average cluster confidence")
    aggregate_weight: float = Field(default=1.0, ge=0.0, description="Summed cluster weight")
    cohesion_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Internal cluster similarity cohesion")
    summary: str = Field(default="", description="Natural language summary of cluster insights")


class EvidenceConflict(BaseKBModel):
    """
    A detected contradiction or incompatibility between two or more evidence nodes.
    """

    conflict_id: str = Field(description="Unique conflict identifier")
    conflict_type: Literal[
        "CONTRADICTORY_CLAIM",
        "BRAND_CONSTRAINT_VIOLATION",
        "MUTUALLY_EXCLUSIVE_ARCHETYPE",
        "OUTDATED_EVIDENCE",
        "CONFIDENCE_DISPARITY",
    ] = Field(description="Classification of detected conflict")
    node_ids: List[str] = Field(default_factory=list, description="IDs of conflicting nodes")
    description: str = Field(description="Detailed explanation of the contradiction")
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    detected_at: str = Field(default_factory=_utc_now_iso)


class ConflictResolution(BaseKBModel):
    """
    Explainable, deterministic resolution record for an EvidenceConflict.
    """

    conflict_id: str = Field(description="Target conflict ID")
    winning_node_id: Optional[str] = Field(default=None, description="ID of prevailing evidence node")
    suppressed_node_ids: List[str] = Field(default_factory=list, description="IDs of suppressed conflicting nodes")
    strategy_applied: str = Field(description="Resolution strategy used (e.g. 'brand_dominance', 'recency')")
    rationale: str = Field(description="Explainable audit rationale for the decision")
    resolved_at: str = Field(default_factory=_utc_now_iso)


class EvidenceStatistics(BaseKBModel):
    """
    Operational telemetry and quality metrics for the normalized evidence graph.
    """

    total_raw_evidence_count: int = 0
    valid_nodes_count: int = 0
    active_nodes_count: int = 0
    suppressed_nodes_count: int = 0
    edges_count: int = 0
    clusters_count: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    average_confidence: float = 0.0
    average_weight: float = 0.0
    processing_time_ms: float = 0.0


class EvidenceSummary(BaseKBModel):
    """
    High-level domain digest synthesized from the normalized graph.
    """

    graph_id: str = Field(description="Associated graph ID")
    primary_archetype: Optional[str] = None
    dominant_patterns: List[str] = Field(default_factory=list)
    active_brand_constraints: List[str] = Field(default_factory=list)
    key_historical_findings: List[str] = Field(default_factory=list)
    competitor_differentiators: List[str] = Field(default_factory=list)
    overall_evidence_health: float = Field(default=1.0, ge=0.0, le=1.0)


class NormalizedEvidence(BaseKBModel):
    """
    Consolidated individual normalized evidence item.
    """

    node_id: str
    node_type: KnowledgeEntryType
    origin: str
    confidence: float
    weight: float
    reason: str
    is_active: bool = True
    cluster_id: Optional[str] = None
    data_payload: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)


class NormalizedEvidenceGraph(BaseKBModel):
    """
    The master output artifact of the Evidence Normalization Engine.
    Represents the complete, grounded, validated, conflict-resolved evidence graph.
    This is the exclusive evidence input consumed by future reasoning engines.
    """

    graph_id: str = Field(description="Unique graph identifier")
    nodes: Dict[str, EvidenceNode] = Field(default_factory=dict, description="All validated evidence nodes by ID")
    edges: List[EvidenceEdge] = Field(default_factory=list, description="All directed relationships")
    clusters: List[EvidenceCluster] = Field(default_factory=list, description="Evidence clusters")
    conflicts: List[EvidenceConflict] = Field(default_factory=list, description="Detected conflicts")
    resolutions: List[ConflictResolution] = Field(default_factory=list, description="Explainable resolution records")
    summary: EvidenceSummary = Field(description="High-level domain summary")
    statistics: EvidenceStatistics = Field(default_factory=EvidenceStatistics, description="Telemetry metrics")
    created_at: str = Field(default_factory=_utc_now_iso)

    def get_active_nodes(self) -> List[EvidenceNode]:
        """Return all active, non-suppressed evidence nodes."""
        return [node for node in self.nodes.values() if node.is_active]

    def get_node(self, node_id: str) -> Optional[EvidenceNode]:
        """Retrieve node by ID."""
        return self.nodes.get(node_id)

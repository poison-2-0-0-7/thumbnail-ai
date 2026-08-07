"""
intelligence_kb
===============

Knowledge Base, Retrieval, and Evidence Normalization package for Thumbnail Intelligence Engine.
"""

from __future__ import annotations

from thumbnail_intelligence.knowledge_base import *  # noqa: F401, F403
import thumbnail_intelligence.knowledge_base as knowledge_base
import thumbnail_intelligence.retrieval as retrieval
import thumbnail_intelligence.evidence as evidence
import thumbnail_intelligence.reasoning as reasoning
from thumbnail_intelligence.retrieval import *  # noqa: F401, F403
from thumbnail_intelligence.evidence import *  # noqa: F401, F403
from thumbnail_intelligence.reasoning import *  # noqa: F401, F403

__all__ = [
    "knowledge_base",
    "retrieval",
    "evidence",
    "reasoning",
    # Config
    "KnowledgeBaseConfig",
    "StorageConfig",
    "VersioningConfig",
    "RetrievalConfig",
    "RankingWeights",
    "EvidenceNormalizationConfig",
    "EvidenceSourcePriorities",
    # Models
    "BaseKBModel",
    "KnowledgeEntry",
    "KnowledgeEntryType",
    "CreatorProfile",
    "ChannelProfile",
    "CompetitorProfile",
    "Archetype",
    "EvidenceReference",
    "EvidenceSourceType",
    "EvidenceGrade",
    "DesignReason",
    "BrandConstraint",
    "IdentityConstraint",
    "VisualPattern",
    "DesignPattern",
    "ThumbnailPattern",
    # Retrieval
    "RetrievalQuery",
    "QueryContext",
    "SearchFilters",
    "EvidenceBundle",
    "RetrievedEvidence",
    "RetrievalResult",
    "RetrievalScore",
    "SearchStatistics",
    "RankingMetadata",
    "KnowledgeRetriever",
    # Evidence Normalization & Graph
    "EvidenceNormalizer",
    "EvidenceGraph",
    "EvidenceGraphValidator",
    "ProvenanceTracker",
    "ConfidencePropagator",
    "EvidenceWeighter",
    "EvidenceMerger",
    "ConflictDetector",
    "ConflictResolver",
    "EvidenceClusterer",
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
]

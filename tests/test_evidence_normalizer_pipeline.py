"""
Unit tests for the end-to-end EvidenceNormalizer pipeline and repository integration.
Tests pipeline execution from RetrievalResult to NormalizedEvidenceGraph,
summary synthesis, active node filtering, and telemetry statistics.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.evidence.normalizer import EvidenceNormalizer
from thumbnail_intelligence.knowledge_base.config import KnowledgeBaseConfig
from thumbnail_intelligence.knowledge_base.models import (
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.knowledge_base.repository import KnowledgeBaseRepository
from thumbnail_intelligence.retrieval.config import RetrievalConfig
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
)
from thumbnail_intelligence.retrieval.query import QueryContext, RetrievalQuery
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.retriever import KnowledgeRetriever
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


def test_evidence_normalizer_pipeline_end_to_end(tmp_path: Path) -> None:
    kb_config = KnowledgeBaseConfig(base_dir=tmp_path)
    repo = KnowledgeBaseRepository(config=kb_config)

    # 1. Seed repository
    repo.seed_default_archetypes()
    repo.seed_default_patterns()

    # 2. Add creator history and competitor profile
    entry1 = KnowledgeEntry(
        entry_id="video_tech_01",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[0.1] * 512,
        niche="tech",
        source_channel_id="channel_tech_01",
        facets={"resolution": "4K", "ctr": 0.14},
    )
    repo.entries.register(entry1)

    # 3. Retrieve using KnowledgeRetriever
    retrieval_config = RetrievalConfig(embedding_dim=512)
    retriever = KnowledgeRetriever(repository=repo, config=retrieval_config)

    retrieval_result = retriever.retrieve(
        RetrievalQuery(
            query_id="query_norm_test",
            query_embedding=[0.1] * 512,
            context=QueryContext(niche="tech", channel_id="channel_tech_01"),
            top_k=8,
        )
    )
    assert retrieval_result.status == "success"
    assert len(retrieval_result.bundle.items) >= 2

    # 4. Normalize evidence into NormalizedEvidenceGraph
    normalizer = EvidenceNormalizer()
    graph = normalizer.normalize(retrieval_result)

    assert isinstance(graph, NormalizedEvidenceGraph)
    assert len(graph.nodes) >= 2
    assert len(graph.get_active_nodes()) >= 2
    assert graph.statistics.valid_nodes_count >= 2
    assert graph.statistics.average_confidence > 0.0
    assert graph.statistics.processing_time_ms > 0.0

    # Verify domain summary
    assert graph.summary.overall_evidence_health > 0.0
    assert graph.summary.graph_id == graph.graph_id

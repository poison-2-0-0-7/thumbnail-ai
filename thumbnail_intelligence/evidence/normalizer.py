"""
normalizer.py
=============

Master orchestrator for the Evidence Normalization Engine.
Transforms raw RetrievalResult and EvidenceBundle inputs into a validated,
grounded, conflict-resolved NormalizedEvidenceGraph consumed by future reasoning engines.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from thumbnail_intelligence.evidence.clustering import EvidenceClusterer
from thumbnail_intelligence.evidence.confidence import ConfidencePropagator
from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.conflict_resolution import (
    ConflictDetector,
    ConflictResolver,
)
from thumbnail_intelligence.evidence.graph import EvidenceGraph
from thumbnail_intelligence.evidence.merger import EvidenceMerger
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceCluster,
    EvidenceConflict,
    EvidenceEdge,
    EvidenceNode,
    EvidenceStatistics,
    EvidenceSummary,
    EvidenceWeight,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.evidence.provenance import ProvenanceTracker
from thumbnail_intelligence.evidence.validator import EvidenceGraphValidator
from thumbnail_intelligence.evidence.weighting import EvidenceWeighter
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceSourceType,
    KnowledgeEntryType,
    _utc_now_iso,
)
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
)


class EvidenceNormalizer:
    """
    Primary engine for normalizing retrieved multi-domain evidence into a clean,
    grounded, conflict-resolved directed evidence graph.
    """

    def __init__(self, config: Optional[EvidenceNormalizationConfig] = None) -> None:
        self.config = config or EvidenceNormalizationConfig()
        self.validator = EvidenceGraphValidator()
        self.provenance_tracker = ProvenanceTracker()
        self.confidence_propagator = ConfidencePropagator()
        self.weighter = EvidenceWeighter()
        self.merger = EvidenceMerger()
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
        self.clusterer = EvidenceClusterer()

    def normalize(
        self,
        retrieval_input: Union[RetrievalResult, EvidenceBundle],
        graph_id: Optional[str] = None,
    ) -> NormalizedEvidenceGraph:
        """
        Execute end-to-end evidence normalization pipeline.
        Returns immutable NormalizedEvidenceGraph.
        """
        start_time = time.perf_counter()

        # Step 1: Extract bundle and query metadata
        if isinstance(retrieval_input, RetrievalResult):
            bundle = retrieval_input.bundle
            query_id = retrieval_input.query.query_id
        else:
            bundle = retrieval_input
            query_id = bundle.query_id

        raw_items: List[RetrievedEvidence] = list(bundle.items)
        gid = graph_id or f"graph_{uuid.uuid4().hex[:10]}"

        # Step 2: Construct initial EvidenceNodes
        initial_nodes: Dict[str, EvidenceNode] = {}
        for item in raw_items:
            # Calibrate initial confidence and weight
            conf = self.confidence_propagator.compute_initial_confidence(item, self.config)
            prov = self.provenance_tracker.create_record(item, query_id=query_id)

            node_id = f"node_{item.entry_id}"
            node = EvidenceNode(
                node_id=node_id,
                node_type=item.entry_type,
                evidence_item=item,
                confidence=conf,
                weight=EvidenceWeight(),  # computed in next step
                provenance=prov,
                is_active=True,
                metadata=dict(item.data_payload),
            )
            initial_weight = self.weighter.compute_initial_weight(node, self.config)
            object.__setattr__(node, "weight", initial_weight)

            # Validate grounding of individual node
            self.validator.validate_node(node)
            initial_nodes[node_id] = node

        # Step 3: Duplicate Merging
        merged_nodes, id_map = self.merger.merge_duplicates(
            initial_nodes,
            threshold=self.config.duplicate_similarity_threshold,
        )

        # Step 4: Conflict Detection & Resolution
        conflicts = self.conflict_detector.detect_conflicts(merged_nodes)
        resolutions, conflict_edges = self.conflict_resolver.resolve_conflicts(
            conflicts=conflicts,
            nodes=merged_nodes,
            config=self.config,
        )

        # Step 5: Evidence Clustering
        clusters: List[EvidenceCluster] = []
        cluster_edges: List[EvidenceEdge] = []
        if self.config.enable_clustering:
            clusters, cluster_edges = self.clusterer.cluster_evidence(
                nodes=merged_nodes,
                threshold=self.config.cluster_similarity_threshold,
            )
            # Reweight nodes based on cluster membership and cohesion
            self.weighter.reweight_with_clusters(merged_nodes, clusters)

        # Step 6: Construct and Populate Directed Graph
        graph = EvidenceGraph(graph_id=gid)
        for node in merged_nodes.values():
            graph.add_node(node)

        # Combine all edges (conflict resolution edges + cluster edges)
        all_edges: List[EvidenceEdge] = []
        all_edges.extend(conflict_edges)
        all_edges.extend(cluster_edges)

        # Remap edges in case endpoints were merged
        remapped_edges = self.merger.remap_edges(all_edges, id_map)
        for edge in remapped_edges:
            graph.add_edge(edge)

        # Step 7: Confidence Propagation
        self.confidence_propagator.propagate_confidence(
            nodes=graph.nodes,
            edges=list(graph.edges.values()),
            decay_factor=self.config.confidence_decay_factor,
        )

        # Step 8: Final Graph Integrity and Grounding Validation
        self.validator.validate_graph(
            nodes=graph.nodes,
            edges=list(graph.edges.values()),
            max_nodes=self.config.max_graph_nodes,
        )

        # Step 9: Synthesize Domain Summary & Telemetry Statistics
        active_nodes = graph.get_active_nodes()
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        avg_conf = (
            sum(n.confidence.propagated_confidence for n in active_nodes) / len(active_nodes)
            if active_nodes
            else 0.0
        )
        avg_weight = (
            sum(n.weight.effective_weight for n in active_nodes) / len(active_nodes)
            if active_nodes
            else 0.0
        )

        stats = EvidenceStatistics(
            total_raw_evidence_count=len(raw_items),
            valid_nodes_count=len(graph.nodes),
            active_nodes_count=len(active_nodes),
            suppressed_nodes_count=len(graph.nodes) - len(active_nodes),
            edges_count=len(graph.edges),
            clusters_count=len(clusters),
            conflicts_detected=len(conflicts),
            conflicts_resolved=len(resolutions),
            average_confidence=round(avg_conf, 4),
            average_weight=round(avg_weight, 4),
            processing_time_ms=round(total_time_ms, 2),
        )

        # Extract primary archetype and key findings
        arch_nodes = graph.get_nodes_by_type(KnowledgeEntryType.ARCHETYPE_EXAMPLE)
        primary_arch = arch_nodes[0].node_id if arch_nodes else None

        pattern_nodes = graph.get_nodes_by_type(KnowledgeEntryType.VISUAL_PATTERN)
        pattern_names = [n.node_id for n in pattern_nodes[:5]]

        brand_nodes = [
            n for n in graph.nodes.values()
            if n.is_active and (n.node_type == KnowledgeEntryType.CREATOR_PROFILE_ENTRY or n.provenance.source_type == EvidenceSourceType.BRAND_RULE)
        ]
        brand_rules = [n.node_id for n in brand_nodes[:5]]

        summary = EvidenceSummary(
            graph_id=gid,
            primary_archetype=primary_arch,
            dominant_patterns=pattern_names,
            active_brand_constraints=brand_rules,
            key_historical_findings=[
                n.node_id for n in graph.get_nodes_by_type(KnowledgeEntryType.HISTORICAL_THUMBNAIL)[:5]
            ],
            competitor_differentiators=[
                n.node_id for n in graph.get_nodes_by_type(KnowledgeEntryType.COMPETITOR_THUMBNAIL)[:5]
            ],
            overall_evidence_health=round(avg_conf, 4),
        )

        # Step 10: Export Immutable NormalizedEvidenceGraph
        return graph.to_normalized_graph(
            clusters=clusters,
            conflicts=conflicts,
            resolutions=resolutions,
            summary=summary,
            statistics=stats,
        )

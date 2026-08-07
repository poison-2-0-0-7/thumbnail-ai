"""
clustering.py
=============

Evidence clustering and semantic partition generation for the Evidence Normalization Engine.
Groups related archetype templates, historical findings, competitor benchmarks,
and visual patterns into coherent clusters with central exemplar nodes.
"""

from __future__ import annotations

import collections
import hashlib
from typing import Dict, List, Optional, Set, Tuple

from thumbnail_intelligence.evidence.models import (
    EvidenceCluster,
    EvidenceEdge,
    EvidenceNode,
)
from thumbnail_intelligence.knowledge_base.models import KnowledgeEntryType
from thumbnail_intelligence.retrieval.embedding import VectorMath


class EvidenceClusterer:
    """
    Groups evidence nodes into domain and semantic clusters, calculating cohesion and summary metrics.
    """

    @classmethod
    def cluster_evidence(
        cls,
        nodes: Dict[str, EvidenceNode],
        threshold: float = 0.70,
    ) -> Tuple[List[EvidenceCluster], List[EvidenceEdge]]:
        """
        Group active evidence nodes into domain-specific semantic clusters.
        Returns list of EvidenceCluster instances and associated PART_OF_CLUSTER directed edges.
        """
        active_nodes = [n for n in nodes.values() if n.is_active]
        if not active_nodes:
            return [], []

        # 1. Group nodes by knowledge domain type
        domain_groups: Dict[str, List[EvidenceNode]] = collections.defaultdict(list)
        for node in active_nodes:
            t = node.node_type
            if t == KnowledgeEntryType.ARCHETYPE_EXAMPLE:
                domain_groups["archetype"].append(node)
            elif t == KnowledgeEntryType.HISTORICAL_THUMBNAIL:
                domain_groups["historical"].append(node)
            elif t == KnowledgeEntryType.COMPETITOR_THUMBNAIL:
                domain_groups["competitor"].append(node)
            elif t in (
                KnowledgeEntryType.VISUAL_PATTERN,
                KnowledgeEntryType.DESIGN_PATTERN,
                KnowledgeEntryType.THUMBNAIL_PATTERN,
            ):
                domain_groups["pattern"].append(node)
            elif t in (KnowledgeEntryType.CREATOR_PROFILE_ENTRY, KnowledgeEntryType.BRAND_RULE):
                domain_groups["brand_constraint"].append(node)
            else:
                domain_groups["general"].append(node)

        clusters: List[EvidenceCluster] = []
        cluster_edges: List[EvidenceEdge] = []

        for domain_name, group in domain_groups.items():
            if not group:
                continue

            # Identify central node (highest effective weight)
            sorted_nodes = sorted(group, key=lambda n: n.weight.effective_weight, reverse=True)
            central_node = sorted_nodes[0]
            node_ids = [n.node_id for n in group]

            # Calculate aggregate metrics
            avg_conf = sum(n.confidence.propagated_confidence for n in group) / len(group)
            total_weight = sum(n.weight.effective_weight for n in group)

            # Calculate internal cohesion
            cohesion = 1.0
            if len(group) > 1:
                pairwise_sims: List[float] = []
                for i in range(len(group)):
                    vec_i = getattr(group[i].evidence_item, "embedding", None)
                    for j in range(i + 1, len(group)):
                        vec_j = getattr(group[j].evidence_item, "embedding", None)
                        if vec_i and vec_j and len(vec_i) == len(vec_j):
                            try:
                                sim = VectorMath.cosine_similarity(vec_i, vec_j)
                                pairwise_sims.append(max(0.0, sim))
                            except Exception:
                                pass
                if pairwise_sims:
                    cohesion = sum(pairwise_sims) / len(pairwise_sims)

            hasher = hashlib.sha256()
            hasher.update(f"{domain_name}:{central_node.node_id}:{len(group)}".encode("utf-8"))
            cluster_id = f"cluster_{domain_name}_{hasher.hexdigest()[:8]}"

            # Assign cluster_id to member nodes
            for n in group:
                object.__setattr__(n, "cluster_id", cluster_id)

            summary = f"{domain_name.capitalize()} cluster with {len(group)} items anchored on '{central_node.node_id}'."

            cluster = EvidenceCluster(
                cluster_id=cluster_id,
                cluster_type=domain_name,
                node_ids=node_ids,
                central_node_id=central_node.node_id,
                aggregate_confidence=round(avg_conf, 4),
                aggregate_weight=round(total_weight, 4),
                cohesion_score=round(cohesion, 4),
                summary=summary,
            )
            clusters.append(cluster)

            # Generate PART_OF_CLUSTER edges to central node
            for n in group:
                if n.node_id != central_node.node_id:
                    edge = EvidenceEdge.create(
                        source_id=n.node_id,
                        target_id=central_node.node_id,
                        relation_type="PART_OF_CLUSTER",
                        weight=1.0,
                        confidence=avg_conf,
                        explanation=f"Node belongs to {domain_name} cluster anchored on {central_node.node_id}",
                    )
                    cluster_edges.append(edge)

        return clusters, cluster_edges

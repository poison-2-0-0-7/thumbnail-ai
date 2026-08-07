"""
weighting.py
============

Evidence weighting and influence calculation for the Evidence Normalization Engine.
Computes base empirical weights, cluster support reinforcement, and source hierarchy scaling.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.models import (
    EvidenceCluster,
    EvidenceNode,
    EvidenceWeight,
)


class EvidenceWeighter:
    """
    Evaluates and calculates the effective empirical weight for each evidence node.
    """

    @classmethod
    def compute_initial_weight(
        cls,
        node: EvidenceNode,
        config: EvidenceNormalizationConfig,
    ) -> EvidenceWeight:
        """
        Calculate initial EvidenceWeight from retrieval scoring and origin priority.
        """
        base_w = getattr(node.evidence_item.score, "overall_score", 1.0)
        prio_dict = config.source_priorities.to_dict()
        type_str = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
        source_prio = prio_dict.get(type_str.lower(), 0.80)

        effective = base_w * source_prio
        effective = max(0.05, min(2.0, effective))

        explanation = (
            f"Base: {base_w:.3f} * SourcePriority: {source_prio:.2f} -> Effective: {effective:.3f}"
        )

        return EvidenceWeight(
            base_weight=round(base_w, 4),
            effective_weight=round(effective, 4),
            cluster_support_multiplier=1.0,
            source_priority_multiplier=round(source_prio, 4),
            explanation=explanation,
        )

    @classmethod
    def reweight_with_clusters(
        cls,
        nodes: Dict[str, EvidenceNode],
        clusters: List[EvidenceCluster],
    ) -> None:
        """
        Scale effective node weight based on membership in high-cohesion evidence clusters.
        """
        for cluster in clusters:
            multiplier = 1.0 + min(0.5, len(cluster.node_ids) * 0.05) * cluster.cohesion_score
            for nid in cluster.node_ids:
                node = nodes.get(nid)
                if node and node.is_active:
                    new_effective = node.weight.effective_weight * multiplier
                    updated_weight = EvidenceWeight(
                        base_weight=node.weight.base_weight,
                        effective_weight=round(new_effective, 4),
                        cluster_support_multiplier=round(multiplier, 4),
                        source_priority_multiplier=node.weight.source_priority_multiplier,
                        explanation=f"{node.weight.explanation} | Cluster '{cluster.cluster_id}' boost x{multiplier:.2f}",
                    )
                    object.__setattr__(node, "weight", updated_weight)

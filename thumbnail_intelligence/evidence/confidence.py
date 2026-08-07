"""
confidence.py
=============

Confidence propagation and multi-signal calibration for the Evidence Normalization Engine.
Evaluates source quality multipliers, retrieval score factors, metadata completeness,
and edge-based confidence decay across supporting graph dependencies.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Set

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceEdge,
    EvidenceNode,
)
from thumbnail_intelligence.knowledge_base.models import EvidenceGrade, KnowledgeEntryType
from thumbnail_intelligence.retrieval.evidence_bundle import RetrievedEvidence


class ConfidencePropagator:
    """
    Computes calibrated confidence scores and propagates confidence across graph edges.
    """

    @classmethod
    def compute_initial_confidence(
        cls,
        evidence: RetrievedEvidence,
        config: EvidenceNormalizationConfig,
    ) -> ConfidenceScore:
        """
        Calculate initial ConfidenceScore from retrieval confidence, origin priority,
        and empirical metadata quality.
        """
        raw_conf = max(0.0, min(1.0, evidence.confidence))
        retrieval_factor = max(0.0, min(1.0, evidence.score.overall_score))

        # Origin priority factor
        prio_dict = config.source_priorities.to_dict()
        type_str = evidence.entry_type.value if hasattr(evidence.entry_type, "value") else str(evidence.entry_type)
        source_factor = prio_dict.get(type_str.lower(), 0.80)

        # Metadata quality factor
        meta_factor = max(0.0, min(1.0, evidence.score.metadata_quality_score))

        # Initial effective confidence
        effective = raw_conf * 0.40 + retrieval_factor * 0.30 + source_factor * 0.20 + meta_factor * 0.10
        effective = max(0.0, min(1.0, effective))

        explanation = (
            f"Raw: {raw_conf:.2f}, Retrieval: {retrieval_factor:.2f}, "
            f"Source: {source_factor:.2f}, Meta: {meta_factor:.2f} -> Initial: {effective:.3f}"
        )

        return ConfidenceScore(
            raw_confidence=round(raw_conf, 4),
            propagated_confidence=round(effective, 4),
            source_quality_factor=round(source_factor, 4),
            retrieval_quality_factor=round(retrieval_factor, 4),
            metadata_quality_factor=round(meta_factor, 4),
            decay_hops=0,
            explanation=explanation,
        )

    @classmethod
    def propagate_confidence(
        cls,
        nodes: Dict[str, EvidenceNode],
        edges: List[EvidenceEdge],
        decay_factor: float = 0.90,
    ) -> None:
        """
        Propagate confidence across SUPPORTS and DEPENDS_ON edges in topological or BFS order.
        Reinforces confidence when multiple nodes support an insight, and decays confidence
        over multi-hop dependency chains.
        """
        if not nodes or not edges:
            return

        # Build incoming support map: target_id -> [(source_id, edge)]
        incoming_support: Dict[str, List[EvidenceEdge]] = collections.defaultdict(list)
        for edge in edges:
            if edge.relation_type in ("SUPPORTS", "DEPENDS_ON", "DERIVED_FROM"):
                incoming_support[edge.target_node_id].append(edge)

        for node_id, node in nodes.items():
            if not node.is_active:
                continue

            incoming_edges = incoming_support.get(node_id, [])
            if not incoming_edges:
                continue

            current_conf = node.confidence.propagated_confidence
            supporting_confs: List[float] = []

            for edge in incoming_edges:
                source_node = nodes.get(edge.source_node_id)
                if source_node and source_node.is_active:
                    source_conf = source_node.confidence.propagated_confidence
                    decayed = source_conf * decay_factor * edge.weight
                    supporting_confs.append(decayed)

            if supporting_confs:
                # Bayesian / Noisy-OR style reinforcement combination
                # 1 - prod(1 - c_i)
                prob_false = 1.0 - current_conf
                for sc in supporting_confs:
                    prob_false *= (1.0 - (sc * 0.5))  # partial support multiplier

                combined = 1.0 - prob_false
                combined = max(current_conf, min(1.0, combined))

                updated_conf = ConfidenceScore(
                    raw_confidence=node.confidence.raw_confidence,
                    propagated_confidence=round(combined, 4),
                    source_quality_factor=node.confidence.source_quality_factor,
                    retrieval_quality_factor=node.confidence.retrieval_quality_factor,
                    metadata_quality_factor=node.confidence.metadata_quality_factor,
                    decay_hops=node.confidence.decay_hops + 1,
                    explanation=f"{node.confidence.explanation} | Boosted by {len(supporting_confs)} supporting edges to {combined:.3f}",
                )
                object.__setattr__(node, "confidence", updated_conf)

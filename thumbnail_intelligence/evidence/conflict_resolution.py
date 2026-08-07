"""
conflict_resolution.py
======================

Conflict detection and deterministic resolution engine for the Evidence Normalization Engine.
Identifies contradictory claims, brand constraint violations, and mutually exclusive archetypes,
resolving them with explainable audit trails and edge annotations.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.exceptions import UnresolvableConflictError
from thumbnail_intelligence.evidence.models import (
    ConflictResolution,
    EvidenceConflict,
    EvidenceEdge,
    EvidenceNode,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceSourceType,
    KnowledgeEntryType,
    _utc_now_iso,
)


class ConflictDetector:
    """
    Scans candidate evidence nodes to identify contradictory claims,
    brand constraint violations, and mutually exclusive archetypes.
    """

    @classmethod
    def detect_conflicts(
        cls,
        nodes: Dict[str, EvidenceNode],
    ) -> List[EvidenceConflict]:
        """
        Scan all active evidence nodes and return list of detected EvidenceConflict instances.
        """
        conflicts: List[EvidenceConflict] = []
        active_nodes = [n for n in nodes.values() if n.is_active]

        # 1. Brand Constraint vs General Pattern Violations
        brand_nodes = [
            n for n in active_nodes
            if n.node_type == KnowledgeEntryType.CREATOR_PROFILE_ENTRY or n.provenance.source_type == EvidenceSourceType.BRAND_RULE
        ]
        pattern_nodes = [
            n for n in active_nodes
            if n.node_type in (
                KnowledgeEntryType.VISUAL_PATTERN,
                KnowledgeEntryType.DESIGN_PATTERN,
                KnowledgeEntryType.COMPETITOR_THUMBNAIL,
            )
        ]

        for b_node in brand_nodes:
            b_payload = b_node.evidence_item.data_payload
            prohibited = b_payload.get("prohibited_elements", []) + b_payload.get("prohibited_tropes", [])
            prohibited_set = {str(p).lower() for p in prohibited}

            for p_node in pattern_nodes:
                p_payload = p_node.evidence_item.data_payload
                p_name = str(p_payload.get("name", "")).lower()
                p_desc = str(p_payload.get("description", "")).lower()

                # Check if pattern promotes a prohibited element
                for prob in prohibited_set:
                    if prob and (prob in p_name or prob in p_desc):
                        cid = f"conf_brand_{b_node.node_id}_{p_node.node_id}"
                        conflicts.append(
                            EvidenceConflict(
                                conflict_id=cid,
                                conflict_type="BRAND_CONSTRAINT_VIOLATION",
                                node_ids=[b_node.node_id, p_node.node_id],
                                description=f"Pattern node '{p_node.node_id}' promotes '{prob}', which is prohibited by Brand Constraint '{b_node.node_id}'.",
                                severity="critical",
                            )
                        )

        # 2. Mutually Exclusive Archetypes
        archetype_nodes = [
            n for n in active_nodes if n.node_type == KnowledgeEntryType.ARCHETYPE_EXAMPLE
        ]
        if len(archetype_nodes) > 1:
            # If multiple distinct archetypes have high confidence (> 0.85), flag archetype collision
            top_archetypes = [
                a for a in archetype_nodes if a.confidence.propagated_confidence >= 0.85
            ]
            if len(top_archetypes) >= 2:
                cid = f"conf_arch_{top_archetypes[0].node_id}_{top_archetypes[1].node_id}"
                conflicts.append(
                    EvidenceConflict(
                        conflict_id=cid,
                        conflict_type="MUTUALLY_EXCLUSIVE_ARCHETYPE",
                        node_ids=[a.node_id for a in top_archetypes],
                        description=f"Multiple high-confidence archetypes detected: {[a.node_id for a in top_archetypes]}.",
                        severity="medium",
                    )
                )

        return conflicts


class ConflictResolver:
    """
    Deterministically resolves detected evidence conflicts based on brand dominance,
    source hierarchy, recency, and calibrated confidence.
    """

    @classmethod
    def resolve_conflicts(
        cls,
        conflicts: List[EvidenceConflict],
        nodes: Dict[str, EvidenceNode],
        config: EvidenceNormalizationConfig,
    ) -> Tuple[List[ConflictResolution], List[EvidenceEdge]]:
        """
        Execute deterministic conflict resolution.
        Modifies node active states in-place, returns explainable resolutions and graph edges.
        """
        resolutions: List[ConflictResolution] = []
        resolution_edges: List[EvidenceEdge] = []

        for conflict in conflicts:
            conflicting_nodes = [nodes[nid] for nid in conflict.node_ids if nid in nodes]
            if len(conflicting_nodes) < 2:
                continue

            winning_node: Optional[EvidenceNode] = None
            suppressed_nodes: List[EvidenceNode] = []
            strategy = config.conflict_resolution_strategy
            rationale = ""

            # Strategy 1: Brand Dominance (Brand rules and identity constraints always win)
            if conflict.conflict_type == "BRAND_CONSTRAINT_VIOLATION" or strategy == "brand_dominance":
                brand_candidates = [
                    n for n in conflicting_nodes
                    if n.node_type == KnowledgeEntryType.CREATOR_PROFILE_ENTRY or n.provenance.source_type == EvidenceSourceType.BRAND_RULE
                ]

                if brand_candidates:
                    winning_node = brand_candidates[0]
                    suppressed_nodes = [n for n in conflicting_nodes if n.node_id != winning_node.node_id]
                    rationale = f"Brand dominance: '{winning_node.node_id}' takes absolute precedence over empirical patterns."
                else:
                    strategy = "highest_confidence"

            # Strategy 2: Highest Confidence
            if not winning_node and strategy in ("highest_confidence", "hybrid"):
                sorted_by_conf = sorted(
                    conflicting_nodes,
                    key=lambda n: (n.confidence.propagated_confidence, n.weight.effective_weight),
                    reverse=True,
                )
                winning_node = sorted_by_conf[0]
                suppressed_nodes = sorted_by_conf[1:]
                rationale = f"Confidence dominance: '{winning_node.node_id}' (conf={winning_node.confidence.propagated_confidence:.3f}) selected over suppressed nodes."

            # Strategy 3: Most Recent
            if not winning_node and strategy == "most_recent":
                sorted_by_recency = sorted(
                    conflicting_nodes,
                    key=lambda n: n.provenance.created_at,
                    reverse=True,
                )
                winning_node = sorted_by_recency[0]
                suppressed_nodes = sorted_by_recency[1:]
                rationale = f"Recency dominance: '{winning_node.node_id}' is the freshest evidence."

            # Apply suppression
            if winning_node:
                for supp in suppressed_nodes:
                    object.__setattr__(supp, "is_active", False)
                    object.__setattr__(
                        supp,
                        "suppression_reason",
                        f"Suppressed by conflict resolution '{conflict.conflict_id}' in favor of '{winning_node.node_id}': {rationale}",
                    )

                    # Add SUPERSEDES directed edge: winner -> suppressed
                    edge_sup = EvidenceEdge.create(
                        source_id=winning_node.node_id,
                        target_id=supp.node_id,
                        relation_type="SUPERSEDES",
                        weight=1.0,
                        confidence=winning_node.confidence.propagated_confidence,
                        explanation=f"Winner {winning_node.node_id} supersedes {supp.node_id}",
                    )
                    resolution_edges.append(edge_sup)

                    # Add CONTRADICTS directed edge: suppressed -> winner
                    edge_con = EvidenceEdge.create(
                        source_id=supp.node_id,
                        target_id=winning_node.node_id,
                        relation_type="CONTRADICTS",
                        weight=0.5,
                        confidence=supp.confidence.propagated_confidence,
                        explanation=f"Contradiction identified in {conflict.conflict_id}",
                    )
                    resolution_edges.append(edge_con)

                res = ConflictResolution(
                    conflict_id=conflict.conflict_id,
                    winning_node_id=winning_node.node_id,
                    suppressed_node_ids=[s.node_id for s in suppressed_nodes],
                    strategy_applied=strategy,
                    rationale=rationale,
                )
                resolutions.append(res)
            elif not config.allow_unresolved_conflicts:
                raise UnresolvableConflictError(
                    message=f"Conflict '{conflict.conflict_id}' could not be resolved deterministically",
                    context={"conflict": conflict.to_dict()},
                )

        return resolutions, resolution_edges

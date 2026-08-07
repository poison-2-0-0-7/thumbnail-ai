"""
validator.py
============

Strict grounding and graph integrity validation for the Evidence Normalization Engine.
Guarantees:
- No anonymous or fabricated evidence
- Full provenance and timestamp validity
- Graph integrity: valid edge endpoints and cycle detection on dependency DAGs
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Set

from thumbnail_intelligence.evidence.exceptions import (
    CyclicDependencyError,
    GraphError,
    GroundingValidationError,
    NodeNotFoundError,
)
from thumbnail_intelligence.evidence.models import EvidenceEdge, EvidenceNode


class EvidenceGraphValidator:
    """
    Validates evidence nodes and graph relationships against strict grounding invariants.
    """

    @classmethod
    def validate_node(cls, node: EvidenceNode) -> None:
        """
        Validate an individual EvidenceNode for grounding, origin, and non-negativity.
        Raises GroundingValidationError if requirements are violated.
        """
        if not node.node_id or not node.node_id.strip():
            raise GroundingValidationError(
                message="EvidenceNode must have a non-empty node_id",
                context={"node": str(node)},
            )

        prov = node.provenance
        if not prov.origin or not prov.origin.strip():
            raise GroundingValidationError(
                message=f"Node '{node.node_id}' lacks origin attribution (no anonymous evidence allowed)",
                context={"node_id": node.node_id},
            )

        if not prov.source_id or not prov.source_id.strip():
            raise GroundingValidationError(
                message=f"Node '{node.node_id}' lacks source_id attribution",
                context={"node_id": node.node_id},
            )

        if not prov.retrieval_reason or not prov.retrieval_reason.strip():
            raise GroundingValidationError(
                message=f"Node '{node.node_id}' lacks explainable retrieval_reason",
                context={"node_id": node.node_id},
            )

        # Validate timestamps
        for ts_name, ts_val in [("retrieved_at", prov.retrieved_at), ("created_at", prov.created_at)]:
            if not ts_val:
                raise GroundingValidationError(
                    message=f"Node '{node.node_id}' has empty {ts_name}",
                    context={"node_id": node.node_id, "timestamp_field": ts_name},
                )
            try:
                datetime.fromisoformat(ts_val)
            except Exception as e:
                raise GroundingValidationError(
                    message=f"Node '{node.node_id}' has invalid ISO timestamp '{ts_val}' in {ts_name}: {e}",
                    context={"node_id": node.node_id, "value": ts_val},
                )

        # Validate confidence bounds
        c = node.confidence.propagated_confidence
        if not (0.0 <= c <= 1.0) or c != c:
            raise GroundingValidationError(
                message=f"Node '{node.node_id}' confidence {c} is out of bounds [0.0, 1.0]",
                context={"node_id": node.node_id, "confidence": c},
            )

    @classmethod
    def validate_edge(cls, edge: EvidenceEdge, valid_node_ids: Set[str]) -> None:
        """
        Validate edge connectivity and endpoint existence.
        Raises NodeNotFoundError if an endpoint is missing.
        """
        if edge.source_node_id not in valid_node_ids:
            raise NodeNotFoundError(
                message=f"Edge '{edge.edge_id}' source node '{edge.source_node_id}' does not exist in graph",
                context={"edge_id": edge.edge_id, "source_id": edge.source_node_id},
            )

        if edge.target_node_id not in valid_node_ids:
            raise NodeNotFoundError(
                message=f"Edge '{edge.edge_id}' target node '{edge.target_node_id}' does not exist in graph",
                context={"edge_id": edge.edge_id, "target_id": edge.target_node_id},
            )

    @classmethod
    def check_for_cycles(cls, nodes: Dict[str, EvidenceNode], edges: List[EvidenceEdge]) -> None:
        """
        Detect circular dependencies among directed DEPENDS_ON and DERIVED_FROM edges.
        Raises CyclicDependencyError if a directed cycle exists.
        """
        adj: Dict[str, List[str]] = {nid: [] for nid in nodes.keys()}
        for edge in edges:
            if edge.relation_type in ("DEPENDS_ON", "DERIVED_FROM"):
                adj[edge.source_node_id].append(edge.target_node_id)

        # Standard 3-color DFS cycle detection (0=white, 1=gray, 2=black)
        color: Dict[str, int] = {nid: 0 for nid in nodes.keys()}

        def dfs(u: str, path: List[str]) -> None:
            color[u] = 1
            path.append(u)
            for v in adj.get(u, []):
                if color[v] == 1:
                    cycle_path = " -> ".join(path + [v])
                    raise CyclicDependencyError(
                        message=f"Cyclic dependency detected in evidence graph: {cycle_path}",
                        context={"cycle_path": cycle_path, "nodes": path + [v]},
                    )
                if color[v] == 0:
                    dfs(v, path)
            path.pop()
            color[u] = 2

        for nid in nodes.keys():
            if color[nid] == 0:
                dfs(nid, [])

    @classmethod
    def validate_graph(
        cls,
        nodes: Dict[str, EvidenceNode],
        edges: List[EvidenceEdge],
        max_nodes: int = 200,
    ) -> None:
        """
        Validate complete graph integrity, node capacity limits, endpoint connectivity, and acyclicity.
        """
        if len(nodes) > max_nodes:
            raise GraphError(
                message=f"Evidence graph node count {len(nodes)} exceeds maximum allowed {max_nodes}",
                context={"count": len(nodes), "max_nodes": max_nodes},
            )

        node_ids = set(nodes.keys())
        for node in nodes.values():
            cls.validate_node(node)

        for edge in edges:
            cls.validate_edge(edge, node_ids)

        cls.check_for_cycles(nodes, edges)

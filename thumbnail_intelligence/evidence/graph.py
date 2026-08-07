"""
graph.py
========

Directed evidence graph implementation for the Evidence Normalization Engine.
Maintains in-memory adjacency index, dependency DAG traversal, graph queries,
and output export into NormalizedEvidenceGraph.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Set

from thumbnail_intelligence.evidence.exceptions import NodeNotFoundError
from thumbnail_intelligence.evidence.models import (
    ConflictResolution,
    EvidenceCluster,
    EvidenceConflict,
    EvidenceEdge,
    EvidenceNode,
    EvidenceStatistics,
    EvidenceSummary,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.knowledge_base.models import KnowledgeEntryType, _utc_now_iso


class EvidenceGraph:
    """
    Directed Graph container storing normalized evidence nodes and multi-typed relationship edges.
    """

    def __init__(self, graph_id: Optional[str] = None) -> None:
        self.graph_id = graph_id or f"graph_{_utc_now_iso()[:10]}"
        self.nodes: Dict[str, EvidenceNode] = {}
        self.edges: Dict[str, EvidenceEdge] = {}
        self._outgoing: Dict[str, List[EvidenceEdge]] = collections.defaultdict(list)
        self._incoming: Dict[str, List[EvidenceEdge]] = collections.defaultdict(list)

    def add_node(self, node: EvidenceNode) -> None:
        """Add or replace an EvidenceNode in the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        """
        Add a directed edge connecting source and target nodes.
        Updates internal adjacency indexes.
        """
        self.edges[edge.edge_id] = edge
        self._outgoing[edge.source_node_id].append(edge)
        self._incoming[edge.target_node_id].append(edge)

    def get_node(self, node_id: str) -> Optional[EvidenceNode]:
        """Retrieve an EvidenceNode by ID."""
        return self.nodes.get(node_id)

    def get_active_nodes(self) -> List[EvidenceNode]:
        """Return all active, un-suppressed evidence nodes."""
        return [n for n in self.nodes.values() if n.is_active]

    def get_nodes_by_type(self, node_type: KnowledgeEntryType) -> List[EvidenceNode]:
        """Filter active nodes by knowledge entry classification."""
        return [n for n in self.nodes.values() if n.is_active and n.node_type == node_type]

    def get_outgoing(self, node_id: str) -> List[EvidenceEdge]:
        """Return all directed edges originating from node_id."""
        return self._outgoing.get(node_id, [])

    def get_incoming(self, node_id: str) -> List[EvidenceEdge]:
        """Return all directed edges terminating at node_id."""
        return self._incoming.get(node_id, [])

    def get_supporting_evidence(self, node_id: str) -> List[EvidenceNode]:
        """Return all active nodes connected via SUPPORTS edges."""
        supporting_nodes: List[EvidenceNode] = []
        for edge in self.get_incoming(node_id):
            if edge.relation_type == "SUPPORTS":
                src_node = self.get_node(edge.source_node_id)
                if src_node and src_node.is_active:
                    supporting_nodes.append(src_node)
        return supporting_nodes

    def get_contradictions(self, node_id: str) -> List[EvidenceNode]:
        """Return all nodes connected via CONTRADICTS edges."""
        contradictions: List[EvidenceNode] = []
        for edge in self.get_outgoing(node_id):
            if edge.relation_type == "CONTRADICTS":
                tgt_node = self.get_node(edge.target_node_id)
                if tgt_node:
                    contradictions.append(tgt_node)
        return contradictions

    def get_dependencies(self, node_id: str) -> List[EvidenceNode]:
        """Return all upstream nodes required via DEPENDS_ON or DERIVED_FROM edges."""
        deps: List[EvidenceNode] = []
        for edge in self.get_outgoing(node_id):
            if edge.relation_type in ("DEPENDS_ON", "DERIVED_FROM"):
                tgt_node = self.get_node(edge.target_node_id)
                if tgt_node:
                    deps.append(tgt_node)
        return deps

    def topological_sort(self) -> List[str]:
        """
        Return topologically ordered node IDs for dependency/derivation DAGs.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes.keys()}
        dep_adj: Dict[str, List[str]] = collections.defaultdict(list)

        for edge in self.edges.values():
            if edge.relation_type in ("DEPENDS_ON", "DERIVED_FROM"):
                dep_adj[edge.source_node_id].append(edge.target_node_id)
                in_degree[edge.target_node_id] = in_degree.get(edge.target_node_id, 0) + 1

        queue = collections.deque([nid for nid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in dep_adj.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return order

    def to_normalized_graph(
        self,
        clusters: List[EvidenceCluster],
        conflicts: List[EvidenceConflict],
        resolutions: List[ConflictResolution],
        summary: EvidenceSummary,
        statistics: EvidenceStatistics,
    ) -> NormalizedEvidenceGraph:
        """
        Export internal graph into the immutable NormalizedEvidenceGraph output model.
        """
        return NormalizedEvidenceGraph(
            graph_id=self.graph_id,
            nodes=dict(self.nodes),
            edges=list(self.edges.values()),
            clusters=clusters,
            conflicts=conflicts,
            resolutions=resolutions,
            summary=summary,
            statistics=statistics,
        )

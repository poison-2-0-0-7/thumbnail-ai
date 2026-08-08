"""
graph.py
========

Execution Graph and Scheduler for Phase 4.1 Execution Engine.
Constructs execution DAGs from RenderExecutionPackage operations,
validates dependencies, detects cycles (Kahn's / Tarjan's algorithms),
computes topological ordering, and schedules operations for execution.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    RenderExecutionPackage,
    RenderOperation,
    RenderOperationType,
)
from renderer_v2.execution.exceptions import GraphValidationError

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Execution status for individual DAG nodes."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ExecutionNode:
    """DAG node wrapping a single RenderOperation."""

    def __init__(self, operation: RenderOperation) -> None:
        self.operation = operation
        self.op_id: str = operation.op_id
        self.op_type: RenderOperationType = operation.op_type
        self.prerequisites: Set[str] = set()
        self.dependents: Set[str] = set()
        self.status: NodeStatus = NodeStatus.PENDING
        self.is_gpu_bound: bool = self._check_gpu_bound(operation.op_type)

    @staticmethod
    def _check_gpu_bound(op_type: RenderOperationType) -> bool:
        """Identify if an operation primitive is GPU-bound per architecture spec."""
        return op_type in {
            RenderOperationType.EXTRACT_SUBJECT,
            RenderOperationType.GENERATE_BACKGROUND,
            RenderOperationType.ENHANCE_SUBJECT,
        }


class ExecutionGraph:
    """
    Execution DAG constructed from a RenderExecutionPackage.
    Performs dependency validation, cycle detection, and topological sorting.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, ExecutionNode] = {}
        self._topological_order: List[str] = []

    @classmethod
    def build_from_package(cls, package: RenderExecutionPackage) -> ExecutionGraph:
        """
        Build ExecutionGraph from RenderExecutionPackage render_operations.
        Infers dependencies from explicit ordering + input_key / output_key dataflow.
        """
        graph = cls()

        # Step 1: Add all operation nodes
        for op in package.render_operations:
            graph.nodes[op.op_id] = ExecutionNode(op)

        # Step 2: Build dependencies based on output_key -> input_key producers
        producer_map: Dict[str, str] = {}
        for op in package.render_operations:
            for out_key in op.output_keys:
                producer_map[out_key] = op.op_id

        # Dataflow edges
        for op in package.render_operations:
            node = graph.nodes[op.op_id]
            for in_key in op.input_keys:
                if in_key in producer_map and producer_map[in_key] != op.op_id:
                    producer_id = producer_map[in_key]
                    node.prerequisites.add(producer_id)
                    graph.nodes[producer_id].dependents.add(op.op_id)

        # Sequential sequence fallback edges: if no explicit dataflow link exists,
        # maintain sequential order from package.render_operations
        ops = package.render_operations
        for i in range(len(ops) - 1):
            curr_id = ops[i].op_id
            next_id = ops[i + 1].op_id
            # Add edge if next_id does not already depend on curr_id transitively
            if next_id not in graph.nodes[curr_id].dependents:
                graph.nodes[next_id].prerequisites.add(curr_id)
                graph.nodes[curr_id].dependents.add(next_id)

        # Validate graph & compute topological ordering
        graph.validate_and_sort()
        return graph

    def validate_and_sort(self) -> List[str]:
        """
        Validate graph for cycles and compute topological ordering using Kahn's algorithm.
        Raises GraphValidationError if cycles are detected.
        """
        in_degree: Dict[str, int] = {op_id: len(node.prerequisites) for op_id, node in self.nodes.items()}
        queue: deque[str] = deque([op_id for op_id, deg in in_degree.items() if deg == 0])

        topo_order: List[str] = []
        while queue:
            curr_id = queue.popleft()
            topo_order.append(curr_id)

            for dep_id in self.nodes[curr_id].dependents:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(topo_order) != len(self.nodes):
            # Cycle detected — find cycle nodes using DFS
            cycles = self._detect_cycles()
            raise GraphValidationError(
                message=f"ExecutionGraph contains cyclic dependencies ({len(self.nodes) - len(topo_order)} unvisited nodes).",
                cycles=cycles,
            )

        self._topological_order = topo_order
        return self._topological_order

    def _detect_cycles(self) -> List[List[str]]:
        """Perform Tarjan's / DFS cycle detection to extract cycle paths."""
        visited: Dict[str, int] = {node_id: 0 for node_id in self.nodes}  # 0: unvisited, 1: visiting, 2: visited
        cycles: List[List[str]] = []
        path: List[str] = []

        def dfs(node_id: str):
            visited[node_id] = 1
            path.append(node_id)

            for dep_id in self.nodes[node_id].dependents:
                if visited[dep_id] == 1:
                    cycle_start = path.index(dep_id)
                    cycles.append(path[cycle_start:] + [dep_id])
                elif visited[dep_id] == 0:
                    dfs(dep_id)

            path.pop()
            visited[node_id] = 2

        for node_id in self.nodes:
            if visited[node_id] == 0:
                dfs(node_id)

        return cycles

    def get_topological_order(self) -> List[str]:
        """Return computed topological order of operation IDs."""
        if not self._topological_order:
            self.validate_and_sort()
        return list(self._topological_order)

    def validate_dependencies(self) -> List[str]:
        """Validate that node prerequisite relationships are consistent."""
        errors: List[str] = []
        for op_id, node in self.nodes.items():
            for prereq in node.prerequisites:
                if prereq not in self.nodes:
                    errors.append(f"Operation '{op_id}' references unknown prerequisite '{prereq}'.")
            for dep in node.dependents:
                if dep not in self.nodes:
                    errors.append(f"Operation '{op_id}' references unknown dependent '{dep}'.")
        return errors


class ExecutionScheduler:
    """
    Scheduler orchestrating operation execution traversal over an ExecutionGraph.
    Tracks node completion status, yields ready operations, and enforces
    single-GPU-slot residency constraints.
    """

    def __init__(self, graph: ExecutionGraph) -> None:
        self.graph = graph
        self.completed_nodes: Set[str] = set()
        self.failed_nodes: Set[str] = set()
        self.running_gpu_node: Optional[str] = None

        # Initialize node statuses
        for op_id, node in self.graph.nodes.items():
            if len(node.prerequisites) == 0:
                node.status = NodeStatus.READY
            else:
                node.status = NodeStatus.PENDING

    def get_next_runnable_operations(self, max_concurrent: int = 1) -> List[ExecutionNode]:
        """
        Return next set of ready operations eligible for execution.
        Enforces single-GPU residency rule: at most one GPU node active at a time.
        """
        runnable: List[ExecutionNode] = []
        topo_order = self.graph.get_topological_order()

        for op_id in topo_order:
            node = self.graph.nodes[op_id]
            if node.status != NodeStatus.READY:
                continue

            # GPU residency constraint check
            if node.is_gpu_bound and self.running_gpu_node is not None:
                continue  # Hold GPU-bound node until active GPU node releases slot

            runnable.append(node)
            if len(runnable) >= max_concurrent:
                break

        return runnable

    def mark_started(self, op_id: str) -> None:
        """Mark node execution started."""
        if op_id in self.graph.nodes:
            node = self.graph.nodes[op_id]
            node.status = NodeStatus.RUNNING
            if node.is_gpu_bound:
                self.running_gpu_node = op_id

    def mark_completed(self, op_id: str) -> None:
        """Mark node execution complete and update dependent readiness."""
        if op_id not in self.graph.nodes:
            return

        node = self.graph.nodes[op_id]
        node.status = NodeStatus.COMPLETED
        self.completed_nodes.add(op_id)

        if self.running_gpu_node == op_id:
            self.running_gpu_node = None

        # Update readiness of dependents
        for dep_id in node.dependents:
            dep_node = self.graph.nodes[dep_id]
            if dep_node.status == NodeStatus.PENDING:
                if all(self.graph.nodes[p].status == NodeStatus.COMPLETED for p in dep_node.prerequisites):
                    dep_node.status = NodeStatus.READY

    def mark_failed(self, op_id: str) -> None:
        """Mark node execution failed."""
        if op_id in self.graph.nodes:
            node = self.graph.nodes[op_id]
            node.status = NodeStatus.FAILED
            self.failed_nodes.add(op_id)
            if self.running_gpu_node == op_id:
                self.running_gpu_node = None

    def is_complete(self) -> bool:
        """Check if all nodes in graph have finished (COMPLETED, FAILED, or SKIPPED)."""
        return len(self.completed_nodes) + len(self.failed_nodes) == len(self.graph.nodes)

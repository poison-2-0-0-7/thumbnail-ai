"""
execution_plan_models.py
========================

Strongly typed Pydantic data contracts for the Execution Planner (Phase 3.6).
Defines the deterministic, renderer-agnostic ExecutionPlan, ExecutionGraph,
ExecutionStep, and ResourceEstimates models.

The ExecutionPlan translates a DesignBrief into an executable Directed Acyclic Graph (DAG)
specifying WHAT operations happen, WHEN they happen, IN WHAT ORDER, and under WHAT resource budgets.

Contains ZERO renderer-specific parameters (no ComfyUI nodes, no Stable Diffusion prompts,
no SDXL models, no SAM/YOLO references).
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import yaml
from pydantic import Field, field_validator

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.reasoning.exceptions import CircularDependencyError


class ExecutionStepType(str, Enum):
    """Extensible taxonomy of renderer-agnostic execution operations."""

    LOAD_ASSETS = "load_assets"
    PREPARE_CANVAS = "prepare_canvas"
    BACKGROUND_PLANNING = "background_planning"
    BACKGROUND_GENERATION = "background_generation"
    SUBJECT_EXTRACTION = "subject_extraction"
    SUBJECT_ENHANCEMENT = "subject_enhancement"
    LIGHTING = "lighting"
    SHADOW = "shadow"
    COMPOSITION = "composition"
    OBJECT_PLACEMENT = "object_placement"
    TYPOGRAPHY_PLANNING = "typography_planning"
    TYPOGRAPHY_PLACEMENT = "typography_placement"
    COLOR_HARMONIZATION = "color_harmonization"
    CONTRAST_ADJUSTMENT = "contrast_adjustment"
    VALIDATION = "validation"
    FINAL_COMPOSITE = "final_composite"
    CLEANUP = "cleanup"
    CUSTOM = "custom"


class RetryPolicy(BaseKBModel):
    """Fault tolerance and retry policy configuration per step."""

    max_retries: int = Field(default=3, ge=0, description="Maximum automated retry attempts")
    backoff_factor: float = Field(default=1.5, ge=1.0, description="Exponential backoff multiplier")
    retry_on_vram_oom: bool = Field(default=True, description="Automatically retry with VRAM offload on OOM")
    fallback_action: Optional[str] = Field(default=None, description="Optional fallback operation or bypass path")


class ResourceEstimates(BaseKBModel):
    """Estimated computational resource budgets per step and aggregated graph."""

    vram_bytes: int = Field(default=0, ge=0, description="Estimated VRAM footprint in bytes")
    vram_mb: float = Field(default=0.0, ge=0.0, description="Estimated VRAM footprint in Megabytes")
    cpu_usage_pct: float = Field(default=10.0, ge=0.0, le=100.0, description="Estimated CPU usage percentage")
    temp_storage_mb: float = Field(default=0.0, ge=0.0, description="Temporary disk storage in MB")
    estimated_runtime_ms: float = Field(default=100.0, ge=0.0, description="Estimated execution latency in ms")
    model_loading_overhead_ms: float = Field(default=0.0, ge=0.0, description="Cold model loading overhead in ms")
    estimated_cost: float = Field(default=0.0, ge=0.0, description="Normalized computational cost units")


class ExecutionStep(BaseKBModel):
    """Individual node in the ExecutionGraph declaring operation, inputs, outputs, and dependencies."""

    step_id: str = Field(description="Unique deterministic step identifier (e.g. step_01_load_assets)")
    step_type: ExecutionStepType = Field(description="Classified operation taxonomy type")
    description: str = Field(default="", description="Human-readable operational summary")
    inputs: List[str] = Field(default_factory=list, description="Required input asset keys or state dependencies")
    outputs: List[str] = Field(default_factory=list, description="Generated artifact keys or state outputs")
    dependencies: List[str] = Field(default_factory=list, description="List of step_ids that MUST precede this step")
    resources: ResourceEstimates = Field(default_factory=ResourceEstimates, description="Resource estimates")
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, description="Fault-tolerance policy")
    sourced_from_brief_field: str = Field(
        default="",
        description="Audit traceability key tying step to DesignBrief goal (e.g. composition.visual_hierarchy)",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Renderer-agnostic operational parameters")
    is_optional: bool = Field(default=False, description="Flag indicating if step can be bypassed on failure")
    execution_stage: int = Field(default=0, ge=0, description="Scheduled parallel execution stage index")


class ExecutionMetadata(BaseKBModel):
    """Metadata tracking plan identity, brief linkage, timestamps, and stage counts."""

    plan_id: str = Field(
        default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}",
        description="Unique execution plan ID",
    )
    brief_ref: str = Field(description="Reference to source DesignBrief brief_id")
    schema_version: str = Field(default="1.0.0", description="Plan schema semver version")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC creation timestamp")
    planner_id: str = Field(default="execution_planner_v1", description="Identifier of emitting planner")
    total_steps: int = Field(default=0, ge=0, description="Total steps in execution graph")
    total_stages: int = Field(default=0, ge=0, description="Total parallel stages in execution graph")


class ExecutionGraph(BaseKBModel):
    """
    Directed Acyclic Graph (DAG) of execution steps.
    Provides topological ordering, cycle detection, parallel stage scheduling, and resource calculation.
    """

    graph_id: str = Field(
        default_factory=lambda: f"graph_{uuid.uuid4().hex[:8]}",
        description="Unique execution graph ID",
    )
    steps: Dict[str, ExecutionStep] = Field(default_factory=dict, description="Mapping of step_id to ExecutionStep")
    topological_order: List[str] = Field(default_factory=list, description="Valid topological sequence of step_ids")
    parallel_stages: List[List[str]] = Field(default_factory=list, description="Ordered stages of parallel steps")
    total_vram_peak_mb: float = Field(default=0.0, ge=0.0, description="Estimated peak VRAM footprint across stages")
    total_estimated_runtime_ms: float = Field(default=0.0, ge=0.0, description="Critical path estimated runtime")
    total_estimated_cost: float = Field(default=0.0, ge=0.0, description="Summed total computational cost units")

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect circular dependencies in the execution graph using DFS.
        Returns a list of cycle paths if any cycles exist.
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_id: str):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            step = self.steps.get(node_id)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        dfs(dep)
                    elif dep in rec_stack:
                        cycle_start = path.index(dep)
                        cycles.append(path[cycle_start:] + [dep])

            path.pop()
            rec_stack.remove(node_id)

        for s_id in self.steps:
            if s_id not in visited:
                dfs(s_id)

        return cycles

    def compute_topological_sort(self) -> List[str]:
        """
        Compute valid topological ordering of step_ids using Kahn's Algorithm.
        Raises CircularDependencyError if a cycle is detected.
        """
        in_degree: Dict[str, int] = {s_id: 0 for s_id in self.steps}
        adj_list: Dict[str, List[str]] = defaultdict(list)

        for s_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep in self.steps:
                    adj_list[dep].append(s_id)
                    in_degree[s_id] += 1

        queue: deque[str] = deque([s_id for s_id, deg in in_degree.items() if deg == 0])
        topo: List[str] = []

        while queue:
            node = queue.popleft()
            topo.append(node)
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo) != len(self.steps):
            cycle_paths = self.detect_cycles()
            flattened = cycle_paths[0] if cycle_paths else list(set(self.steps.keys()) - set(topo))
            raise CircularDependencyError(cycle_path=flattened)

        return topo

    def compute_parallel_stages(self) -> List[List[str]]:
        """
        Group independent steps into parallel execution stages based on dependency depth.
        """
        if not self.steps:
            return []

        topo = self.compute_topological_sort()
        stage_map: Dict[str, int] = {}

        for s_id in topo:
            step = self.steps[s_id]
            if not step.dependencies:
                stage_map[s_id] = 0
            else:
                max_dep_stage = max(stage_map.get(dep, 0) for dep in step.dependencies if dep in stage_map)
                stage_map[s_id] = max_dep_stage + 1

        max_stage = max(stage_map.values(), default=0)
        stages: List[List[str]] = [[] for _ in range(max_stage + 1)]

        for s_id, stg in stage_map.items():
            stages[stg].append(s_id)

        return stages

    def validate_graph(self) -> List[str]:
        """
        Validate graph integrity:
        - Check missing dependencies
        - Detect circular dependencies
        - Check missing required inputs
        - Detect duplicate steps
        """
        errors: List[str] = []
        step_ids = set(self.steps.keys())

        # Check missing dependencies
        for s_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step '{s_id}' declares missing dependency '{dep}'.")

        # Check cycles
        cycles = self.detect_cycles()
        if cycles:
            for c in cycles:
                errors.append(f"Circular dependency detected: {' -> '.join(c)}")

        # Check unproduced inputs (excluding base assets)
        all_outputs: Set[str] = set()
        for step in self.steps.values():
            all_outputs.update(step.outputs)

        for s_id, step in self.steps.items():
            for inp in step.inputs:
                if (
                    not inp.startswith("asset:")
                    and not inp.startswith("brief:")
                    and not inp.startswith("canvas:")
                    and inp not in all_outputs
                ):
                    errors.append(f"Step '{s_id}' requires input '{inp}' which is not produced by any preceding step.")

        return errors


class ExecutionPlan(BaseKBModel):
    """
    Master ExecutionPlan contract.
    Contains metadata, ExecutionGraph, quality targets, and execution constraints.
    """

    metadata: ExecutionMetadata = Field(description="ExecutionPlan metadata and provenance")
    graph: ExecutionGraph = Field(description="ExecutionGraph DAG containing steps and schedules")
    quality_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Copied quality score thresholds from DesignBrief",
    )
    execution_constraints: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Copied must-preserve and forbidden transformation rules",
    )

    # ---------------------------------------------------------------------------
    # Serialization Methods
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExecutionPlan to python dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionPlan:
        """Construct ExecutionPlan from python dictionary."""
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize ExecutionPlan to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ExecutionPlan:
        """Deserialize ExecutionPlan from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """Serialize ExecutionPlan to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ExecutionPlan:
        """Deserialize ExecutionPlan from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def validate_plan(self) -> List[str]:
        """Validate ExecutionPlan metadata and underlying ExecutionGraph integrity."""
        errors: List[str] = []
        if not self.metadata.plan_id:
            errors.append("Metadata plan_id must be non-empty.")
        if not self.metadata.brief_ref:
            errors.append("Metadata brief_ref must specify source brief_id.")

        graph_errors = self.graph.validate_graph()
        errors.extend(graph_errors)
        return errors

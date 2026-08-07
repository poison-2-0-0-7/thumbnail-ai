"""
test_execution_planner.py
==========================

Comprehensive unit test suite for ExecutionPlanner and ExecutionPlan data models (Phase 3.6).
Tests cover:
- Deterministic ExecutionPlan DAG generation from DesignBrief
- Topological ordering and dependency verification
- Cycle detection (CircularDependencyError) using Kahn's algorithm and DFS
- Resource estimation (VRAM peak, runtime, CPU, storage, cost)
- Execution graph validation (missing dependencies, unproduced inputs, circular paths)
- Multi-format serialization (JSON, YAML, Dict)
- BaseReasoner contract & interface integration
- Strict renderer independence invariant (zero ComfyUI, SD, SDXL, SAM, YOLO tokens)
"""

import pytest
from typing import Dict, Any

from thumbnail_intelligence.evidence.models import EvidenceSummary, NormalizedEvidenceGraph
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.design_brief_generator import DesignBriefGenerator
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.exceptions import CircularDependencyError, ReasonerValidationError
from thumbnail_intelligence.reasoning.execution_plan_models import (
    ExecutionGraph,
    ExecutionMetadata,
    ExecutionPlan,
    ExecutionStep,
    ExecutionStepType,
    ResourceEstimates,
    RetryPolicy,
)
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.validator import StrategicReasoningValidator


@pytest.fixture
def sample_brief() -> DesignBrief:
    """Construct a complete, valid DesignBrief for testing."""
    return DesignBrief()


class TestExecutionPlanner:

    def test_execution_plan_generation_from_brief(self, sample_brief: DesignBrief):
        """Test generation of strongly typed ExecutionPlan DAG from DesignBrief."""
        planner = ExecutionPlanner()
        plan = planner.plan(sample_brief)

        assert isinstance(plan, ExecutionPlan)
        assert plan.metadata.plan_id.startswith("plan_")
        assert plan.metadata.brief_ref == sample_brief.metadata.brief_id
        assert plan.metadata.schema_version == "1.0.0"
        assert plan.metadata.total_steps == 17
        assert plan.metadata.total_stages > 0

        # Verify step taxonomy coverage
        steps = plan.graph.steps
        assert "step_01_load_assets" in steps
        assert "step_04_background_generation" in steps
        assert "step_07_lighting" in steps
        assert "step_12_typography_placement" in steps
        assert "step_16_final_composite" in steps
        assert "step_17_cleanup" in steps

        # Verify step attribute completeness
        for s_id, step in steps.items():
            assert isinstance(step.step_type, ExecutionStepType)
            assert step.description != ""
            assert step.resources.estimated_runtime_ms > 0
            assert step.retry_policy.max_retries >= 1
            assert step.sourced_from_brief_field != ""

    def test_topological_sort_and_parallel_stages(self, sample_brief: DesignBrief):
        """Test topological ordering and parallel execution stage grouping."""
        planner = ExecutionPlanner()
        plan = planner.plan(sample_brief)
        graph = plan.graph

        topo = graph.topological_order
        assert len(topo) == len(graph.steps)

        # Assert topological dependency invariant: every step appears after all its dependencies
        executed_nodes = set()
        for node_id in topo:
            step = graph.steps[node_id]
            for dep in step.dependencies:
                assert dep in executed_nodes, f"Step '{node_id}' executed before dependency '{dep}'!"
            executed_nodes.add(node_id)

        # Verify parallel stages
        stages = graph.parallel_stages
        assert len(stages) > 0
        flattened_stages = [n for stg in stages for n in stg]
        assert len(flattened_stages) == len(graph.steps)

    def test_cycle_detection(self, sample_brief: DesignBrief):
        """Test circular dependency detection and CircularDependencyError raising."""
        planner = ExecutionPlanner()
        plan = planner.plan(sample_brief)
        graph = plan.graph

        # Introduce artificial circular dependency: step_04 -> step_07 -> step_04
        corrupt_steps = dict(graph.steps)
        step04 = corrupt_steps["step_04_background_generation"].model_copy(
            update={"dependencies": ["step_07_lighting"]}
        )
        corrupt_steps["step_04_background_generation"] = step04

        cyclic_graph = graph.model_copy(update={"steps": corrupt_steps})

        # Cycle detection
        cycles = cyclic_graph.detect_cycles()
        assert len(cycles) > 0
        assert any("step_04_background_generation" in c for c in cycles)

        # Topological sort should raise CircularDependencyError
        with pytest.raises(CircularDependencyError):
            cyclic_graph.compute_topological_sort()

    def test_graph_validation_rules(self, sample_brief: DesignBrief):
        """Test graph.validate_graph() for missing dependencies and unproduced inputs."""
        planner = ExecutionPlanner()
        plan = planner.plan(sample_brief)
        graph = plan.graph

        # Clean graph should have 0 validation errors
        errors = graph.validate_graph()
        assert len(errors) == 0

        # Inject missing dependency and unproduced input
        corrupt_steps = dict(graph.steps)
        bad_step = corrupt_steps["step_01_load_assets"].model_copy(
            update={"dependencies": ["non_existent_step"], "inputs": ["unproduced_artifact_key"]}
        )
        corrupt_steps["step_01_load_assets"] = bad_step
        corrupt_graph = graph.model_copy(update={"steps": corrupt_steps})

        corrupt_errors = corrupt_graph.validate_graph()
        assert len(corrupt_errors) >= 2
        assert any("non_existent_step" in err for err in corrupt_errors)
        assert any("unproduced_artifact_key" in err for err in corrupt_errors)

    def test_resource_estimation(self, sample_brief: DesignBrief):
        """Test peak VRAM, runtime, and cost aggregation."""
        planner = ExecutionPlanner()
        plan = planner.plan(sample_brief)
        graph = plan.graph

        assert graph.total_vram_peak_mb > 0.0
        assert graph.total_estimated_runtime_ms > 0.0
        assert graph.total_estimated_cost > 0.0

        # Assert peak VRAM is at least equal to maximum single step VRAM
        max_single_vram = max(s.resources.vram_mb for s in graph.steps.values())
        assert graph.total_vram_peak_mb >= max_single_vram

    def test_multi_format_serialization(self, sample_brief: DesignBrief):
        """Test JSON, YAML, and Dictionary serialization round-trip fidelity."""
        planner = ExecutionPlanner()
        original_plan = planner.plan(sample_brief)

        # Dict
        plan_dict = original_plan.to_dict()
        assert isinstance(plan_dict, dict)
        dict_restored = ExecutionPlan.from_dict(plan_dict)
        assert dict_restored.metadata.plan_id == original_plan.metadata.plan_id

        # JSON
        plan_json = original_plan.to_json()
        assert isinstance(plan_json, str)
        json_restored = ExecutionPlan.from_json(plan_json)
        assert len(json_restored.graph.steps) == len(original_plan.graph.steps)

        # YAML
        plan_yaml = original_plan.to_yaml()
        assert isinstance(plan_yaml, str)
        yaml_restored = ExecutionPlan.from_yaml(plan_yaml)
        assert yaml_restored.metadata.brief_ref == original_plan.metadata.brief_ref

    def test_reasoner_contract_and_interface(self, sample_brief: DesignBrief):
        """Test BaseReasoner contract integration and reason() method."""
        planner = ExecutionPlanner()

        assert planner.name == "execution_planner"
        assert planner.contract.reasoner_type.value == "execution_planner"
        assert "design_brief_generator" in planner.dependencies

        # Reason via BaseReasoner interface
        ctx = ReasoningContext(graph_id="graph_test_ep", design_brief=sample_brief)
        graph_dummy = NormalizedEvidenceGraph(
            graph_id="graph_test_ep",
            summary=EvidenceSummary(graph_id="graph_test_ep"),
        )
        plan = planner.reason(graph=graph_dummy, context=ctx)

        assert isinstance(plan, ExecutionPlan)
        assert planner.validate_output(plan) is True
        assert planner.validate_output(None) is False

    def test_strict_renderer_independence_invariant(self, sample_brief: DesignBrief):
        """
        Critical Invariant Test: Verify that ExecutionPlan contains ZERO renderer-specific tokens
        (no ComfyUI, Stable Diffusion, SDXL, SAM, GroundingDINO, YOLO, or BrushNet references).
        """
        planner = ExecutionPlanner()
        plan = planner.plan(sample_brief)
        json_dump = plan.to_json().lower()

        forbidden_tokens = [
            "comfyui",
            "stable_diffusion",
            "sdxl",
            "sam",
            "groundingdino",
            "yolo",
            "brushnet",
            "inpainting_mask",
            "lora_weight",
            "controlnet_model",
            "cfg_scale",
            "sampler_name",
        ]

        for token in forbidden_tokens:
            assert token not in json_dump, f"Forbidden renderer token '{token}' found in ExecutionPlan!"

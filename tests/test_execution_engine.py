"""
test_execution_engine.py
=========================

Comprehensive test suite for Phase 4.1 Execution Engine Foundation.
Tests cover:
- Invariant enforcement: ExecutionEngine consumes ONLY RenderExecutionPackage
- Package validation & error handling
- RenderJobContext creation, timing, metadata, and thread-safe cancellation
- RenderWorkspace lifecycle (initialize, layer/mask/instance storage, checkpoints, restore, cleanup)
- ExecutionFSM state transitions, legal transition rules, and invalid transition errors
- ExecutionGraph creation, dependency validation, Kahn's topological sorting, and cycle detection
- ExecutionScheduler operation readiness and single-GPU-slot residency constraints
- ExecutionDispatcher routing to placeholder stages for all 14 RenderOperationType primitives
- Execution reports (StageExecutionReport, RenderJobReport) assembly and status accuracy
- End-to-end execution flow via ExecutionEngine.execute()
- Edge cases and error handling (missing assets, malformed packages, job cancellation)
"""

import time
import pytest
from typing import Dict, Any

from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    PixelBoundingBox,
    RenderAssetReference,
    RenderBackgroundInstruction,
    RenderExecutionPackage,
    RenderLayerEntry,
    RenderLightingInstruction,
    RenderMaskInstruction,
    RenderOperation,
    RenderOperationType,
    RenderPackageMetadata,
    RenderPlacementCoordinate,
    RenderSceneGraph,
    RenderTypographyInstruction,
)
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner
from renderer_v2.execution.context import RenderJobContext
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.engine import ExecutionEngine
from renderer_v2.execution.exceptions import (
    GraphValidationError,
    InvalidStateTransitionError,
    JobCancellationError,
    PackageValidationError,
)
from renderer_v2.execution.fsm import ExecutionFSM, ExecutionState
from renderer_v2.execution.graph import ExecutionGraph, ExecutionScheduler, NodeStatus
from renderer_v2.execution.models import LayerBuffer, SceneInstance
from renderer_v2.execution.reports import (
    RenderJobReport,
    RenderJobStatus,
    StageExecutionReport,
    StageStatus,
)
from renderer_v2.execution.stages import BaseExecutionStage
from renderer_v2.execution.validation import GraphValidator, PackageValidator
from renderer_v2.execution.workspace import RenderWorkspace, WorkspaceState


@pytest.fixture
def sample_package() -> RenderExecutionPackage:
    """Construct a complete valid RenderExecutionPackage fixture for testing."""
    brief = DesignBrief()
    plan = ExecutionPlanner().plan(brief)
    comp = SpatialCompositionPlanner().plan(plan, brief)
    adapter = RendererV2Adapter()
    return adapter.translate(comp, plan)


class TestExecutionEngineFoundation:

    # ---------------------------------------------------------------------------
    # Invariant Enforcement
    # ---------------------------------------------------------------------------

    def test_engine_consumes_only_render_execution_package(self, sample_package: RenderExecutionPackage):
        """Verify ExecutionEngine rejects any input that is not a RenderExecutionPackage."""
        engine = ExecutionEngine()

        # Valid package runs
        report = engine.execute(sample_package)
        assert isinstance(report, RenderJobReport)

        # Invalid upstream inputs raise TypeError
        brief = DesignBrief()
        with pytest.raises(TypeError, match="consumes ONLY RenderExecutionPackage"):
            engine.execute(brief)  # type: ignore

        plan = ExecutionPlanner().plan(brief)
        with pytest.raises(TypeError, match="consumes ONLY RenderExecutionPackage"):
            engine.execute(plan)  # type: ignore

    # ---------------------------------------------------------------------------
    # RenderJobContext & Cancellation
    # ---------------------------------------------------------------------------

    def test_job_context_lifecycle_and_cancellation(self, sample_package: RenderExecutionPackage):
        """Test RenderJobContext metadata, timing, attempt tracking, and cancellation."""
        ctx = RenderJobContext(
            package=sample_package,
            job_id="job_test_123",
            correlation_id="corr_test_123",
            attempt=1,
            execution_metadata={"env": "test"},
        )

        assert ctx.job_id == "job_test_123"
        assert ctx.correlation_id == "corr_test_123"
        assert ctx.attempt == 1
        assert ctx.get_meta("env") == "test"
        assert not ctx.is_cancelled()

        # Test timing
        time.sleep(0.01)
        elapsed = ctx.elapsed_time
        assert elapsed > 0.0

        # Test next attempt context
        next_ctx = ctx.create_next_attempt_context()
        assert next_ctx.attempt == 2
        assert next_ctx.job_id == ctx.job_id

        # Test cancellation
        ctx.cancel("Test cancellation trigger")
        assert ctx.is_cancelled()
        with pytest.raises(JobCancellationError, match="Test cancellation trigger"):
            ctx.check_cancellation()

    # ---------------------------------------------------------------------------
    # RenderWorkspace Lifecycle & Checkpoints
    # ---------------------------------------------------------------------------

    def test_workspace_lifecycle_and_checkpoints(self, sample_package: RenderExecutionPackage):
        """Test workspace storage, initialization, layer/mask operations, checkpoints, and restore."""
        ws = RenderWorkspace(job_id="job_ws_123", canvas_width_px=1280, canvas_height_px=720)
        assert ws.state == WorkspaceState.CREATED

        ws.initialize(sample_package)
        assert ws.state == WorkspaceState.INITIALIZED
        assert len(ws.layers) > 0

        # Add custom layer
        ws.add_layer("custom_layer", LayerBuffer(layer_id="custom_layer", layer_name="Test Layer"))
        assert ws.has_layer("custom_layer")
        assert ws.get_layer("custom_layer") is not None

        # Add mask & instance
        ws.add_mask("mask_1", {"mask": "data"})
        assert ws.get_mask("mask_1") == {"mask": "data"}

        inst = SceneInstance(instance_id="inst_1", class_label="person")
        ws.add_scene_instance("inst_1", inst)
        assert ws.get_scene_instance("inst_1") == inst

        # Set depth map
        ws.set_depth_map({"depth": "map_data"})
        assert ws.depth_map == {"depth": "map_data"}

        # Checkpoint workspace
        snapshot = ws.checkpoint("checkpoint_stage_1")
        assert snapshot.checkpoint_name == "checkpoint_stage_1"
        assert "checkpoint_stage_1" in ws.checkpoints
        assert "custom_layer" in snapshot.layer_ids

        # Modify workspace after checkpoint
        ws.add_layer("layer_after_cp", LayerBuffer(layer_id="layer_after_cp"))
        assert ws.has_layer("layer_after_cp")

        # Restore checkpoint
        restored = ws.restore_checkpoint("checkpoint_stage_1")
        assert restored is True
        assert not ws.has_layer("layer_after_cp")
        assert ws.has_layer("custom_layer")

        # Cleanup workspace
        ws.cleanup()
        assert ws.state == WorkspaceState.CLEANED
        assert len(ws.layers) == 0

    # ---------------------------------------------------------------------------
    # Execution FSM State Machine
    # ---------------------------------------------------------------------------

    def test_execution_fsm_transitions(self):
        """Test ExecutionFSM state transition rules and invalid transition handling."""
        fsm = ExecutionFSM()
        assert fsm.current_state == ExecutionState.UNINITIALIZED

        # Legal sequence
        fsm.transition_to(ExecutionState.INITIALIZING)
        fsm.transition_to(ExecutionState.VALIDATING)
        fsm.transition_to(ExecutionState.SCHEDULING)
        fsm.transition_to(ExecutionState.DISPATCHING)
        fsm.transition_to(ExecutionState.RUNNING_STAGE)
        fsm.transition_to(ExecutionState.VALIDATING_STAGE)
        fsm.transition_to(ExecutionState.CHECKPOINTING)
        fsm.transition_to(ExecutionState.COMPLETED)

        assert fsm.is_terminal is True
        assert len(fsm.event_history) == 8

        # Test illegal transition from terminal state
        with pytest.raises(InvalidStateTransitionError):
            fsm.transition_to(ExecutionState.RUNNING_STAGE)

    # ---------------------------------------------------------------------------
    # Execution Graph, Topological Ordering, Cycle Detection
    # ---------------------------------------------------------------------------

    def test_execution_graph_and_cycle_detection(self, sample_package: RenderExecutionPackage):
        """Test building ExecutionGraph, topological sorting, and cycle detection."""
        graph = ExecutionGraph.build_from_package(sample_package)
        assert len(graph.nodes) == len(sample_package.render_operations)

        topo = graph.get_topological_order()
        assert len(topo) == len(sample_package.render_operations)

        # Verify topological ordering invariant: prerequisites appear before dependents
        node_indices = {op_id: idx for idx, op_id in enumerate(topo)}
        for op_id, node in graph.nodes.items():
            for prereq in node.prerequisites:
                assert node_indices[prereq] < node_indices[op_id]

        # Test Cycle Detection
        cyclic_pkg = sample_package.model_copy(deep=True)
        # Introduce explicit cycle between op 0 and op 1
        op0 = cyclic_pkg.render_operations[0]
        op1 = cyclic_pkg.render_operations[1]
        op0.input_keys.append("cycle_key_b")
        op0.output_keys.append("cycle_key_a")
        op1.input_keys.append("cycle_key_a")
        op1.output_keys.append("cycle_key_b")

        with pytest.raises(GraphValidationError, match="contains cyclic dependencies"):
            ExecutionGraph.build_from_package(cyclic_pkg)

    # ---------------------------------------------------------------------------
    # Execution Scheduler & GPU Residency Rules
    # ---------------------------------------------------------------------------

    def test_execution_scheduler_and_gpu_residency(self, sample_package: RenderExecutionPackage):
        """Test ExecutionScheduler node readiness and single-GPU-slot residency constraint."""
        graph = ExecutionGraph.build_from_package(sample_package)
        scheduler = ExecutionScheduler(graph)

        runnable = scheduler.get_next_runnable_operations(max_concurrent=10)
        assert len(runnable) > 0

        # Simulate GPU node execution
        gpu_nodes = [n for n in graph.nodes.values() if n.is_gpu_bound]
        if gpu_nodes:
            gpu_node = gpu_nodes[0]
            scheduler.mark_started(gpu_node.op_id)
            assert scheduler.running_gpu_node == gpu_node.op_id

            # Verify no second GPU node can run concurrently
            next_runnable = scheduler.get_next_runnable_operations(max_concurrent=10)
            for n in next_runnable:
                assert not n.is_gpu_bound

            scheduler.mark_completed(gpu_node.op_id)
            assert scheduler.running_gpu_node is None

    # ---------------------------------------------------------------------------
    # Execution Dispatcher & Placeholder Stages
    # ---------------------------------------------------------------------------

    def test_execution_dispatcher_covers_all_14_primitives(self, sample_package: RenderExecutionPackage):
        """Verify ExecutionDispatcher maps and dispatches all 14 RenderOperationType primitives."""
        dispatcher = ExecutionDispatcher()
        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id)
        ws.initialize(sample_package)

        for op_type in RenderOperationType:
            op = RenderOperation(
                op_id=f"op_{op_type.value}",
                op_type=op_type,
                target_layer_id="test_layer",
            )
            report = dispatcher.dispatch(op, ctx, ws)
            assert isinstance(report, StageExecutionReport)
            assert report.status in {StageStatus.SUCCESS, StageStatus.SUCCESS_WITH_DEGRADATION}
            assert report.op_id == op.op_id

    # ---------------------------------------------------------------------------
    # End-to-End ExecutionEngine Flow & Reports
    # ---------------------------------------------------------------------------

    def test_execution_engine_end_to_end(self, sample_package: RenderExecutionPackage):
        """Test complete end-to-end execution flow returning RenderJobReport."""
        engine = ExecutionEngine()
        report = engine.execute(sample_package, context_overrides={"output_path": "output/test_thumb.jpg"})

        assert isinstance(report, RenderJobReport)
        assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
        assert report.total_latency_s > 0.0
        assert len(report.stage_reports) >= len(sample_package.render_operations)
        assert report.output_image_path == "output/test_thumb.jpg"
        assert report.validation_summary.get("valid") is True

    # ---------------------------------------------------------------------------
    # Package Validation Failure
    # ---------------------------------------------------------------------------

    def test_invalid_package_rejection(self):
        """Test that malformed packages fail validation cleanly before job startup."""
        invalid_pkg = RenderExecutionPackage(
            metadata=RenderPackageMetadata(
                package_id="",  # Empty package_id fails validation
                comp_ref="",
                plan_ref="",
                brief_ref="",
            ),
            scene_graph=RenderSceneGraph(canvas_width_px=1280, canvas_height_px=720),
            render_operations=[],  # Zero operations fails validation
        )

        engine = ExecutionEngine()
        report = engine.execute(invalid_pkg)
        assert report.status == RenderJobStatus.FAILED_FATAL
        assert len(report.errors) > 0

    # ---------------------------------------------------------------------------
    # Cancellation Support
    # ---------------------------------------------------------------------------

    def test_job_cancellation_during_execution(self, sample_package: RenderExecutionPackage):
        """Test early cancellation during dispatch loop returns CANCELLED report."""
        class CancellingStage(BaseExecutionStage):
            @property
            def stage_name(self) -> str:
                return "CancellingStage"

            def execute(self, op: RenderOperation, ctx: RenderJobContext, ws: RenderWorkspace) -> StageExecutionReport:
                ctx.cancel("Cancelled mid-stage")
                return StageExecutionReport(stage=self.stage_name, op_id=op.op_id, status=StageStatus.SUCCESS)

            def validate(self, op: RenderOperation, ws: RenderWorkspace) -> list[str]:
                return []

            def cleanup(self, ws: RenderWorkspace) -> None:
                pass

        dispatcher = ExecutionDispatcher()
        dispatcher.map_operation_type(RenderOperationType.LOAD_ASSET, CancellingStage())

        engine = ExecutionEngine(dispatcher=dispatcher)
        report = engine.execute(sample_package)

        assert report.status == RenderJobStatus.CANCELLED
        assert "Cancelled mid-stage" in report.errors[0]

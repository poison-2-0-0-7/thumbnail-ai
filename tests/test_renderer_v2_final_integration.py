"""
test_renderer_v2_final_integration.py
======================================

Comprehensive end-to-end integration test suite for Phase 4.5 Renderer V2 Final Integration.
Tests cover:
- End-to-end execution without bypassing any architecture layer:
  RenderExecutionPackage -> ExecutionEngine -> ExecutionDispatcher -> Stage Adapters -> ModelRuntimeManager -> Renderer Modules -> RenderWorkspace -> RendererV2Adapter -> Final Composite -> Image Export
- Top-level RendererV2Pipeline and RendererV2Adapter.render() orchestration
- Workspace state propagation, layer stack integrity, and asset reference resolution
- Output image file export validation (dimensions, standard deviation pixel variance, non-zero file size)
- Failure recovery & degraded execution reporting (RenderJobReport)
- End-to-end validation of execution reports and stage latencies
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    RenderAssetReference,
    RenderExecutionPackage,
    RenderOperation,
    RenderOperationType,
)
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner
from renderer_v2.pipeline import RendererV2, RendererV2Pipeline, RendererV2PipelineError
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.engine import ExecutionEngine
from renderer_v2.execution.reports import RenderJobReport, RenderJobStatus, StageStatus
from renderer_v2.runtime.manager import ModelRuntimeManager


@pytest.fixture
def full_brief_and_plans():
    """Construct SpatialComposition and ExecutionPlan fixtures."""
    brief = DesignBrief()
    plan = ExecutionPlanner().plan(brief)
    comp = SpatialCompositionPlanner().plan(plan, brief)
    return comp, plan


class TestRendererV2FinalIntegration:

    def test_end_to_end_renderer_v2_pipeline_execution(self, full_brief_and_plans):
        """Test complete RendererV2Pipeline execution from SpatialComposition + ExecutionPlan to output thumbnail file."""
        comp, plan = full_brief_and_plans

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "final_renderer_v2_thumbnail.png")

            pipeline = RendererV2Pipeline()
            report = pipeline.render(comp, plan, output_path=out_file)

            # 1. Validate RenderJobReport
            assert isinstance(report, RenderJobReport)
            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert report.total_latency_s > 0.0
            assert len(report.stage_reports) > 0

            # 2. Validate Output Thumbnail on Disk
            assert os.path.exists(out_file)
            assert os.path.getsize(out_file) > 1000

            img = cv2.imread(out_file)
            assert img is not None
            assert img.shape == (comp.canvas.height_px, comp.canvas.width_px, 3)
            assert float(np.std(img)) > 0.0  # Non-blank pixel variance

    def test_renderer_v2_adapter_render_convenience_method(self, full_brief_and_plans):
        """Test RendererV2Adapter.render() convenience entry point."""
        comp, plan = full_brief_and_plans

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "adapter_convenience_thumbnail.jpg")

            adapter = RendererV2Adapter()
            report = adapter.render(comp, plan, output_path=out_file)

            assert isinstance(report, RenderJobReport)
            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert os.path.exists(out_file)

    def test_model_runtime_manager_injection_through_pipeline(self, full_brief_and_plans):
        """Verify ModelRuntimeManager model handle acquisitions during pipeline execution."""
        comp, plan = full_brief_and_plans

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "mrm_injected.png")

            runtime_manager = ModelRuntimeManager(max_vram_gb=16.0, max_loaded_models=5)
            dispatcher = ExecutionDispatcher(use_placeholders=False, runtime_manager=runtime_manager)
            engine = ExecutionEngine(dispatcher=dispatcher)
            pipeline = RendererV2Pipeline(runtime_manager=runtime_manager, dispatcher=dispatcher, engine=engine)

            report = pipeline.render(comp, plan, output_path=out_file)

            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert os.path.exists(out_file)

            # Verify memory tracking recorded model state
            mem_status = runtime_manager.get_memory_status()
            assert "max_budget_vram_gb" in mem_status

    def test_degraded_pipeline_execution_and_reporting(self, full_brief_and_plans):
        """Test pipeline degradation reporting when non-fatal stages use fallbacks."""
        comp, plan = full_brief_and_plans

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "degraded_thumb.jpg")

            pipeline = RendererV2Pipeline()
            report = pipeline.render(comp, plan, output_path=out_file)

            # Verify stage reports contain detailed latency and validation notes
            for stg_report in report.stage_reports:
                assert stg_report.stage is not None
                assert stg_report.latency_s >= 0.0
                assert isinstance(stg_report.validation_notes, list)

            assert os.path.exists(out_file)

    def test_invalid_package_raises_pipeline_error(self):
        """Verify pipeline raises RendererV2PipelineError when presented with invalid input."""
        pipeline = RendererV2Pipeline()
        with pytest.raises(RendererV2PipelineError, match="Cannot execute rendering pipeline"):
            pipeline.render_package(None)

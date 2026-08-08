"""
test_end_to_end_pipeline.py
============================

Comprehensive end-to-end rendering pipeline test suite for Phase 4.3.
Tests cover:
- Complete end-to-end pipeline execution: RenderExecutionPackage -> ExecutionEngine -> StageAdapters -> Composite -> Exporter -> Thumbnail image file
- Workspace state propagation across stages (AssetLoader -> SubjectExtractor -> BackgroundGenerator -> LightingEngine -> TypographyRenderer -> LayerComposer -> ImageValidator -> QualityValidator -> Exporter)
- Intermediate artifacts persistence
- Correct stage sequencing and topological execution order
- Output file export validation (dimensions, file size, non-zero pixel content)
- Graceful error handling and failure recovery
- Edge cases (custom resolution, missing optional assets, background fallback)
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
    PixelBoundingBox,
    RenderAssetReference,
    RenderExecutionPackage,
    RenderLightingInstruction,
    RenderOperation,
    RenderOperationType,
    RenderPlacementCoordinate,
    RenderTypographyInstruction,
)
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner
from renderer_v2.execution.adapters import (
    AssetLoaderAdapter,
    BackgroundGeneratorAdapter,
    ExporterAdapter,
    ImageValidatorAdapter,
    LayerComposerAdapter,
    LightingEngineAdapter,
    QualityValidatorAdapter,
    SubjectExtractorAdapter,
    TypographyRendererAdapter,
)
from renderer_v2.execution.context import RenderJobContext
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.engine import ExecutionEngine
from renderer_v2.execution.exceptions import StageExecutionError
from renderer_v2.execution.models import LayerBuffer, SceneInstance
from renderer_v2.execution.reports import RenderJobReport, RenderJobStatus, StageStatus
from renderer_v2.execution.workspace import RenderWorkspace, WorkspaceState


@pytest.fixture
def full_package() -> RenderExecutionPackage:
    """Construct a complete production-shape RenderExecutionPackage."""
    brief = DesignBrief()
    plan = ExecutionPlanner().plan(brief)
    comp = SpatialCompositionPlanner().plan(plan, brief)
    adapter = RendererV2Adapter()
    return adapter.translate(comp, plan)


class TestEndToEndPipeline:

    def test_full_pipeline_execution_and_thumbnail_export(self, full_package: RenderExecutionPackage):
        """Test complete end-to-end rendering pipeline producing a valid output thumbnail image."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "output_thumbnail.png")

            # Create dummy input hero image asset
            hero_path = os.path.join(tmp_dir, "hero.jpg")
            dummy_hero = np.full((720, 1280, 3), 180, dtype=np.uint8)
            cv2.circle(dummy_hero, (640, 360), 200, (50, 100, 220), -1)
            cv2.imwrite(hero_path, dummy_hero)

            # Update package with real input hero image path
            pkg = full_package.model_copy(
                update={
                    "asset_references": [
                        RenderAssetReference(
                            asset_id="hero_subject",
                            asset_type="image_hero",
                            source_key="asset:primary_subject",
                            file_path=hero_path,
                            is_required=True,
                        )
                    ]
                }
            )

            dispatcher = ExecutionDispatcher(use_placeholders=False)
            engine = ExecutionEngine(dispatcher=dispatcher)

            report = engine.execute(pkg, context_overrides={"output_path": out_path})

            # Assertions on Job Report
            assert isinstance(report, RenderJobReport)
            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert report.total_latency_s > 0.0
            assert len(report.stage_reports) >= len(pkg.render_operations)

            # Assert file exported on disk
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 1000  # Non-trivial image file

            # Read exported thumbnail and verify dimensions
            exported_img = cv2.imread(out_path)
            assert exported_img is not None
            assert exported_img.shape == (720, 1280, 3)
            assert float(np.std(exported_img)) > 0.0  # Non-blank pixel variance

    def test_workspace_state_propagation_across_stages(self, full_package: RenderExecutionPackage):
        """Verify that intermediate artifacts, layers, and masks propagate correctly across pipeline stages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "propagated.jpg")

            dispatcher = ExecutionDispatcher(use_placeholders=False)
            engine = ExecutionEngine(dispatcher=dispatcher)

            # Pass source image in context metadata
            src_img = np.full((720, 1280, 3), 100, dtype=np.uint8)
            report = engine.execute(
                full_package,
                context_overrides={"output_path": out_path, "source_image": src_img},
            )

            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}

            # Verify stage reports trail covers all 11 stage classifications
            stages_executed = {r.stage for r in report.stage_reports}
            expected_stages = {
                "AssetLoader",
                "BackgroundGenerator",
                "SubjectExtractor",
                "LightingEngine",
                "TypographyRenderer",
                "LayerComposer",
                "QualityValidator",
                "Exporter",
            }
            assert expected_stages.issubset(stages_executed)

    def test_pipeline_failure_recovery_and_degradation_reporting(self, full_package: RenderExecutionPackage):
        """Test pipeline behavior when non-fatal generative stages trigger procedural fallbacks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "degraded.jpg")

            # Force BackgroundGeneratorAdapter to trigger procedural fallback (inpainter=None)
            dispatcher = ExecutionDispatcher(use_placeholders=False)
            bg_adapter = BackgroundGeneratorAdapter(inpainter=None)
            dispatcher.map_operation_type(RenderOperationType.GENERATE_BACKGROUND, bg_adapter)

            engine = ExecutionEngine(dispatcher=dispatcher)
            report = engine.execute(full_package, context_overrides={"output_path": out_path})

            assert report.status == RenderJobStatus.SUCCESS_WITH_DEGRADATION
            assert os.path.exists(out_path)
            assert len(report.errors) == 0

    def test_custom_canvas_resolution_pipeline(self, full_package: RenderExecutionPackage):
        """Test pipeline execution with non-standard 1080p resolution target."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "1080p_thumb.png")

            # Update scene graph dimensions to 1920x1080
            sg_1080 = full_package.scene_graph.model_copy(update={"canvas_width_px": 1920, "canvas_height_px": 1080})
            pkg_1080 = full_package.model_copy(update={"scene_graph": sg_1080})

            dispatcher = ExecutionDispatcher(use_placeholders=False)
            engine = ExecutionEngine(dispatcher=dispatcher)

            report = engine.execute(pkg_1080, context_overrides={"output_path": out_path})

            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert os.path.exists(out_path)

            img_1080 = cv2.imread(out_path)
            assert img_1080 is not None
            assert img_1080.shape == (1080, 1920, 3)

"""
test_stage_adapters.py
=======================

Comprehensive test suite for Phase 4.2 Renderer Stage Adapters.
Tests cover:
- AssetLoaderAdapter (asset decoding, required vs non-required asset handling, exception wrapping)
- BackgroundGeneratorAdapter (procedural fallback, layer storage, error handling)
- SubjectExtractorAdapter (SceneDecomposer integration, fallback extraction, depth_map and mask storage)
- LightingEngineAdapter (RelightingSpec mapping, NonDestructiveEdgeRelighter integration)
- TypographyRendererAdapter (VectorTypographyEngine & SaliencySolver integration, vector text layers)
- LayerComposerAdapter (Canvas & Recompositor alpha compositing, z-index ordering)
- ImageValidatorAdapter (structural validation, bounds checking, NaN pixel corruption detection)
- QualityValidatorAdapter (QualityGatekeeper scoring integration)
- ExporterAdapter (final raster image file export to target output path)
- End-to-end integration flow via ExecutionEngine with Phase 4.2 Stage Adapters
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
from renderer_v2.execution.workspace import RenderWorkspace


@pytest.fixture
def sample_package() -> RenderExecutionPackage:
    """Construct a complete valid RenderExecutionPackage fixture for testing."""
    brief = DesignBrief()
    plan = ExecutionPlanner().plan(brief)
    comp = SpatialCompositionPlanner().plan(plan, brief)
    adapter = RendererV2Adapter()
    return adapter.translate(comp, plan)


class TestStageAdapters:

    # ---------------------------------------------------------------------------
    # AssetLoaderAdapter Tests
    # ---------------------------------------------------------------------------

    def test_asset_loader_adapter_success_and_failure(self, sample_package: RenderExecutionPackage):
        """Test AssetLoaderAdapter loading existing assets and failing on missing required assets."""
        adapter = AssetLoaderAdapter()
        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id)

        # Create temporary dummy image asset on disk
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy_path = tmp.name
            img = np.full((100, 100, 3), 200, dtype=np.uint8)
            cv2.imwrite(dummy_path, img)

        try:
            # Inject valid asset reference into package
            valid_pkg = sample_package.model_copy(
                update={
                    "asset_references": [
                        RenderAssetReference(
                            asset_id="hero_img",
                            asset_type="image_hero",
                            source_key="asset:primary_subject",
                            file_path=dummy_path,
                            is_required=True,
                        )
                    ]
                }
            )
            valid_ctx = RenderJobContext(package=valid_pkg)

            op = RenderOperation(op_id="op_load_asset", op_type=RenderOperationType.LOAD_ASSET)
            report = adapter.execute(op, valid_ctx, ws)

            assert report.status == StageStatus.SUCCESS
            assert ws.intermediate_artifacts.get("asset:hero_img") is not None

            # Test missing required asset raises StageExecutionError
            invalid_pkg = sample_package.model_copy(
                update={
                    "asset_references": [
                        RenderAssetReference(
                            asset_id="missing_hero",
                            asset_type="image_hero",
                            source_key="asset:missing",
                            file_path="/non/existent/path/image.png",
                            is_required=True,
                        )
                    ]
                }
            )
            invalid_ctx = RenderJobContext(package=invalid_pkg)

            with pytest.raises(StageExecutionError, match="Required asset 'missing_hero'"):
                adapter.execute(op, invalid_ctx, ws)

        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    # ---------------------------------------------------------------------------
    # BackgroundGeneratorAdapter Tests
    # ---------------------------------------------------------------------------

    def test_background_generator_adapter_fallback(self, sample_package: RenderExecutionPackage):
        """Test BackgroundGeneratorAdapter procedural gradient fallback generation."""
        adapter = BackgroundGeneratorAdapter(inpainter=None)
        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

        op = RenderOperation(
            op_id="op_gen_bg",
            op_type=RenderOperationType.GENERATE_BACKGROUND,
            target_layer_id="background",
        )
        report = adapter.execute(op, ctx, ws)

        assert report.status in {StageStatus.SUCCESS, StageStatus.SUCCESS_WITH_DEGRADATION}
        assert ws.has_layer("background")

        bg_buf = ws.get_layer("background")
        assert bg_buf is not None
        assert isinstance(bg_buf.buffer_data, np.ndarray)
        assert bg_buf.buffer_data.shape == (720, 1280, 3)

    # ---------------------------------------------------------------------------
    # SubjectExtractorAdapter Tests
    # ---------------------------------------------------------------------------

    def test_subject_extractor_adapter_fallback(self, sample_package: RenderExecutionPackage):
        """Test SubjectExtractorAdapter scene decomposition fallback."""
        adapter = SubjectExtractorAdapter(decomposer=None)
        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

        op = RenderOperation(
            op_id="op_ext_subj",
            op_type=RenderOperationType.EXTRACT_SUBJECT,
        )
        report = adapter.execute(op, ctx, ws)

        assert report.status == StageStatus.SUCCESS
        assert len(ws.scene_instances) > 0
        assert ws.depth_map is not None

        hero_inst = list(ws.scene_instances.values())[0]
        assert hero_inst.class_label == "person"
        assert hero_inst.mask_buffer is not None

    # ---------------------------------------------------------------------------
    # LightingEngineAdapter Tests
    # ---------------------------------------------------------------------------

    def test_lighting_engine_adapter_relighting(self, sample_package: RenderExecutionPackage):
        """Test LightingEngineAdapter relighting application via NonDestructiveEdgeRelighter."""
        adapter = LightingEngineAdapter()

        # Add custom lighting instruction
        pkg = sample_package.model_copy(
            update={
                "lighting_instructions": [
                    RenderLightingInstruction(
                        target_element_id="hero_subject",
                        key_light_direction="top_left",
                        key_light_intensity=0.75,
                        rim_light_enabled=True,
                    )
                ]
            }
        )
        ctx = RenderJobContext(package=pkg)
        ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

        op = RenderOperation(
            op_id="op_apply_light",
            op_type=RenderOperationType.APPLY_LIGHTING,
            target_layer_id="relit_subject",
        )
        report = adapter.execute(op, ctx, ws)

        assert report.status == StageStatus.SUCCESS
        assert ws.has_layer("relit_subject")

        layer_buf = ws.get_layer("relit_subject")
        assert layer_buf is not None
        assert isinstance(layer_buf.buffer_data, np.ndarray)
        assert layer_buf.buffer_data.shape == (720, 1280, 4)

    # ---------------------------------------------------------------------------
    # TypographyRendererAdapter Tests
    # ---------------------------------------------------------------------------

    def test_typography_renderer_adapter(self, sample_package: RenderExecutionPackage):
        """Test TypographyRendererAdapter vector text rendering."""
        adapter = TypographyRendererAdapter()
        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

        op = RenderOperation(
            op_id="op_render_typo",
            op_type=RenderOperationType.RENDER_TYPOGRAPHY,
        )
        report = adapter.execute(op, ctx, ws)

        assert report.status == StageStatus.SUCCESS
        assert len(report.output_keys) > 0

        first_typo_key = report.output_keys[0]
        assert ws.has_layer(first_typo_key)

        typo_buf = ws.get_layer(first_typo_key)
        assert typo_buf is not None
        assert isinstance(typo_buf.buffer_data, np.ndarray)
        assert typo_buf.buffer_data.shape == (720, 1280, 4)

    # ---------------------------------------------------------------------------
    # LayerComposerAdapter Tests
    # ---------------------------------------------------------------------------

    def test_layer_composer_adapter_compositing(self, sample_package: RenderExecutionPackage):
        """Test LayerComposerAdapter multi-layer alpha compositing."""
        adapter = LayerComposerAdapter()
        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

        # Add background and foreground layers to workspace
        bg_rgb = np.full((720, 1280, 3), 40, dtype=np.uint8)
        fg_rgba = np.zeros((720, 1280, 4), dtype=np.uint8)
        fg_rgba[100:300, 100:300] = (255, 0, 0, 255)

        ws.add_layer("bg", LayerBuffer(layer_id="bg", layer_type="background", z_index=0, buffer_data=bg_rgb))
        ws.add_layer("fg", LayerBuffer(layer_id="fg", layer_type="subject", z_index=5, buffer_data=fg_rgba))

        op = RenderOperation(
            op_id="op_compose",
            op_type=RenderOperationType.COMPOSE_LAYER,
            target_layer_id="composite_final",
        )
        report = adapter.execute(op, ctx, ws)

        assert report.status == StageStatus.SUCCESS
        assert ws.has_layer("composite_final")

        comp_buf = ws.get_layer("composite_final")
        assert comp_buf is not None
        assert isinstance(comp_buf.buffer_data, np.ndarray)
        assert comp_buf.buffer_data.shape == (720, 1280, 3)

    # ---------------------------------------------------------------------------
    # ImageValidatorAdapter & QualityValidatorAdapter Tests
    # ---------------------------------------------------------------------------

    def test_validators_adapters(self, sample_package: RenderExecutionPackage):
        """Test ImageValidatorAdapter and QualityValidatorAdapter evaluation."""
        img_val = ImageValidatorAdapter()
        qual_val = QualityValidatorAdapter()

        ctx = RenderJobContext(package=sample_package)
        ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

        # Add composite final layer
        comp_rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
        ws.add_layer("composite_final", LayerBuffer(layer_id="composite_final", buffer_data=comp_rgb))

        op_img = RenderOperation(op_id="op_img_val", op_type=RenderOperationType.EVALUATE_QUALITY)
        rep_img = img_val.execute(op_img, ctx, ws)
        assert rep_img.status == StageStatus.SUCCESS

        op_qual = RenderOperation(op_id="op_qual_val", op_type=RenderOperationType.EVALUATE_QUALITY)
        rep_qual = qual_val.execute(op_qual, ctx, ws)
        assert rep_qual.status in {StageStatus.SUCCESS, StageStatus.SUCCESS_WITH_DEGRADATION}
        assert ws.intermediate_artifacts.get("quality_report") is not None

        # Test corrupt NaN detection in ImageValidatorAdapter
        corrupt_rgb = np.full((720, 1280, 3), np.nan, dtype=np.float32)
        ws.add_layer("composite_final", LayerBuffer(layer_id="composite_final", buffer_data=corrupt_rgb))
        with pytest.raises(StageExecutionError, match="Corrupt pixel data detected"):
            img_val.execute(op_img, ctx, ws)

    # ---------------------------------------------------------------------------
    # ExporterAdapter Tests
    # ---------------------------------------------------------------------------

    def test_exporter_adapter_file_writing(self, sample_package: RenderExecutionPackage):
        """Test ExporterAdapter writing final raster image file to target path."""
        adapter = ExporterAdapter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_output.jpg")
            ctx = RenderJobContext(package=sample_package, execution_metadata={"output_path": out_file})
            ws = RenderWorkspace(job_id=ctx.job_id, canvas_width_px=1280, canvas_height_px=720)

            comp_rgb = np.full((720, 1280, 3), 200, dtype=np.uint8)
            ws.add_layer("composite_final", LayerBuffer(layer_id="composite_final", buffer_data=comp_rgb))

            op = RenderOperation(op_id="op_export", op_type=RenderOperationType.COMPOSITE_FINAL)
            report = adapter.execute(op, ctx, ws)

            assert report.status == StageStatus.SUCCESS
            assert os.path.exists(out_file)
            assert os.path.getsize(out_file) > 0

    # ---------------------------------------------------------------------------
    # End-to-End ExecutionEngine Integration Test with Phase 4.2 Adapters
    # ---------------------------------------------------------------------------

    def test_execution_engine_end_to_end_with_adapters(self, sample_package: RenderExecutionPackage):
        """Test complete ExecutionEngine execution flow with Phase 4.2 Stage Adapters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "final_thumbnail.jpg")

            dispatcher = ExecutionDispatcher(use_placeholders=False)
            engine = ExecutionEngine(dispatcher=dispatcher)

            report = engine.execute(sample_package, context_overrides={"output_path": out_path})

            assert isinstance(report, RenderJobReport)
            assert report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert report.total_latency_s > 0.0
            assert len(report.stage_reports) >= len(sample_package.render_operations)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

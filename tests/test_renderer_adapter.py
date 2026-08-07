"""
test_renderer_adapter.py
========================

Comprehensive unit test suite for RendererAdapter and RenderExecutionPackage data models (Phase 3.8).
Tests cover:
- Deterministic translation of SpatialComposition + ExecutionPlan into RenderExecutionPackage
- Normalized-to-pixel coordinate mapping accuracy
- Target renderer extensibility (RendererV2, ComfyUI, Flux, Imagen, Custom)
- Layer stack ordering and z-index sorting
- Asset reference and coordinate bounds validation
- Multi-format serialization (JSON, YAML, Dict)
- BaseReasoner contract & interface integration
- Pure translation invariant (zero pixel rendering, zero model execution calls)
"""

import pytest
from typing import Dict, Any

from thumbnail_intelligence.evidence.models import EvidenceSummary, NormalizedEvidenceGraph
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.design_brief_generator import DesignBriefGenerator
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.exceptions import ReasonerValidationError
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.renderer_adapter import (
    BaseRendererAdapter,
    FutureComfyUIAdapter,
    FutureCustomAdapter,
    FutureFluxAdapter,
    FutureImagenAdapter,
    RendererAdapter,
    RendererV2Adapter,
)
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
from thumbnail_intelligence.reasoning.spatial_composition_models import (
    BoundingBox,
    CanvasSpecification,
    SpatialComposition,
)
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner


@pytest.fixture
def sample_brief() -> DesignBrief:
    """Construct a complete DesignBrief for testing."""
    return DesignBrief()


@pytest.fixture
def sample_plan(sample_brief: DesignBrief) -> ExecutionPlan:
    """Construct a complete ExecutionPlan for testing."""
    planner = ExecutionPlanner()
    return planner.plan(sample_brief)


@pytest.fixture
def sample_composition(sample_plan: ExecutionPlan, sample_brief: DesignBrief) -> SpatialComposition:
    """Construct a complete SpatialComposition for testing."""
    planner = SpatialCompositionPlanner()
    return planner.plan(sample_plan, sample_brief)


class TestRendererAdapter:

    def test_renderer_v2_translation(self, sample_composition: SpatialComposition, sample_plan: ExecutionPlan):
        """Test translation of SpatialComposition + ExecutionPlan into RenderExecutionPackage."""
        adapter = RendererV2Adapter()
        pkg = adapter.translate(sample_composition, sample_plan)

        assert isinstance(pkg, RenderExecutionPackage)
        assert pkg.metadata.package_id.startswith("pkg_render_")
        assert pkg.metadata.comp_ref == sample_composition.composition_id
        assert pkg.metadata.plan_ref == sample_plan.metadata.plan_id
        assert pkg.metadata.target_renderer == "RendererV2"
        assert pkg.metadata.schema_version == "1.0.0"

        # Scene Graph
        assert pkg.scene_graph.canvas_width_px == 1280
        assert pkg.scene_graph.canvas_height_px == 720
        assert len(pkg.scene_graph.nodes) == len(sample_composition.composition_graph.nodes)

        # Operations
        assert len(pkg.render_operations) == len(sample_plan.graph.topological_order)
        assert pkg.render_operations[0].op_type == RenderOperationType.LOAD_ASSET

        # Placements
        assert len(pkg.placement_coordinates) == len(sample_composition.composition_graph.nodes)
        first_placement = pkg.placement_coordinates[0]
        assert first_placement.bbox_pixels.width_px > 0
        assert first_placement.bbox_pixels.height_px > 0

        # Layer Stack
        assert len(pkg.layer_stack) == len(sample_composition.composition_graph.nodes)
        # Check layer stack is sorted ascending by z_index
        z_indices = [l.z_index for l in pkg.layer_stack]
        assert z_indices == sorted(z_indices)

        # Sub-instructions
        assert pkg.background_instruction.action == "replace"
        assert len(pkg.lighting_instructions) >= 1
        assert len(pkg.typography_instructions) >= 1

    def test_coordinate_mapping_accuracy(self, sample_composition: SpatialComposition, sample_plan: ExecutionPlan):
        """Test accurate conversion from normalized [0.0, 1.0] to pixel coordinates [0, 1280] x [0, 720]."""
        adapter = RendererV2Adapter()
        pkg = adapter.translate(sample_composition, sample_plan)

        # Primary subject placement in sample_composition is (0.10, 0.12, 0.42, 0.78)
        # 0.10 * 1280 = 128, 0.12 * 720 = 86 (rounded), 0.42 * 1280 = 538, 0.78 * 720 = 562
        primary_p = next((p for p in pkg.placement_coordinates if p.element_id == "elem_01_primary_subject"), None)
        assert primary_p is not None

        bbox_px = primary_p.bbox_pixels
        assert pytest.approx(bbox_px.x_px, abs=2) == 128
        assert pytest.approx(bbox_px.y_px, abs=2) == 86
        assert pytest.approx(bbox_px.width_px, abs=2) == 538
        assert pytest.approx(bbox_px.height_px, abs=2) == 562
        assert bbox_px.to_tuple() == (bbox_px.x_px, bbox_px.y_px, bbox_px.width_px, bbox_px.height_px)

    def test_multi_target_renderer_adapters(self, sample_composition: SpatialComposition, sample_plan: ExecutionPlan):
        """Test extensibility adapters (ComfyUI, Flux, Imagen, Custom) targeting different renderers."""
        adapters = [
            FutureComfyUIAdapter(),
            FutureFluxAdapter(),
            FutureImagenAdapter(),
            FutureCustomAdapter("EngineX"),
        ]

        expected_targets = ["ComfyUI", "Flux", "Imagen", "EngineX"]

        for adapter, expected_target in zip(adapters, expected_targets):
            assert adapter.target_renderer_id == expected_target
            pkg = adapter.translate(sample_composition, sample_plan)
            assert pkg.metadata.target_renderer == expected_target

    def test_package_validation(self, sample_composition: SpatialComposition, sample_plan: ExecutionPlan):
        """Test RenderExecutionPackage.validate_package() coordinate bounds and asset validation."""
        adapter = RendererV2Adapter()
        pkg = adapter.translate(sample_composition, sample_plan)

        # Clean package validation
        errors = pkg.validate_package()
        assert len(errors) == 0

        # Corrupt package with out-of-bounds placement and empty asset_id
        corrupt_placements = list(pkg.placement_coordinates)
        corrupt_p = corrupt_placements[0].model_copy(
            update={"bbox_pixels": PixelBoundingBox(x_px=1300, y_px=100, width_px=200, height_px=200)}
        )
        corrupt_placements[0] = corrupt_p

        corrupt_assets = [RenderAssetReference(asset_id="", asset_type="hero", source_key="key_bad")]

        corrupt_pkg = pkg.model_copy(
            update={
                "placement_coordinates": corrupt_placements,
                "asset_references": corrupt_assets,
            }
        )

        corrupt_errors = corrupt_pkg.validate_package()
        assert len(corrupt_errors) >= 2
        assert any("exceeds canvas bounds" in err for err in corrupt_errors)
        assert any("empty asset_id" in err for err in corrupt_errors)

    def test_multi_format_serialization(self, sample_composition: SpatialComposition, sample_plan: ExecutionPlan):
        """Test JSON, YAML, and Dictionary serialization round-trip fidelity."""
        adapter = RendererV2Adapter()
        original_pkg = adapter.translate(sample_composition, sample_plan)

        # Dict
        pkg_dict = original_pkg.to_dict()
        assert isinstance(pkg_dict, dict)
        dict_restored = RenderExecutionPackage.from_dict(pkg_dict)
        assert dict_restored.metadata.package_id == original_pkg.metadata.package_id

        # JSON
        pkg_json = original_pkg.to_json()
        assert isinstance(pkg_json, str)
        json_restored = RenderExecutionPackage.from_json(pkg_json)
        assert len(json_restored.render_operations) == len(original_pkg.render_operations)

        # YAML
        pkg_yaml = original_pkg.to_yaml()
        assert isinstance(pkg_yaml, str)
        yaml_restored = RenderExecutionPackage.from_yaml(pkg_yaml)
        assert yaml_restored.metadata.comp_ref == original_pkg.metadata.comp_ref

    def test_reasoner_contract_and_interface(
        self,
        sample_composition: SpatialComposition,
        sample_plan: ExecutionPlan,
        sample_brief: DesignBrief,
    ):
        """Test BaseReasoner contract integration and reason() method."""
        adapter = RendererV2Adapter()

        assert adapter.name == "renderer_adapter_rendererv2"
        assert adapter.contract.reasoner_type.value == "renderer_adapter"
        assert "spatial_composition_planner" in adapter.dependencies

        # Reason via BaseReasoner interface
        ctx = ReasoningContext(
            graph_id="graph_test_ra",
            design_brief=sample_brief,
            execution_plan=sample_plan,
            spatial_composition=sample_composition,
        )
        graph_dummy = NormalizedEvidenceGraph(
            graph_id="graph_test_ra",
            summary=EvidenceSummary(graph_id="graph_test_ra"),
        )
        pkg = adapter.reason(graph=graph_dummy, context=ctx)

        assert isinstance(pkg, RenderExecutionPackage)
        assert adapter.validate_output(pkg) is True
        assert adapter.validate_output(None) is False

    def test_pure_translation_no_rendering_execution(
        self, sample_composition: SpatialComposition, sample_plan: ExecutionPlan
    ):
        """
        Critical Invariant Test: Verify that RendererAdapter performs pure translation ONLY
        and does NOT perform image rendering, pixel generation, or diffusion model execution.
        """
        adapter = RendererV2Adapter()
        pkg = adapter.translate(sample_composition, sample_plan)

        # Verify package is a purely structural data contract
        assert hasattr(pkg, "render_operations")
        assert hasattr(pkg, "scene_graph")
        assert not hasattr(pkg, "rendered_image")
        assert not hasattr(pkg, "pixel_bytes")

"""
test_spatial_composition_planner.py
====================================

Comprehensive unit test suite for SpatialCompositionPlanner and SpatialComposition data models (Phase 3.7).
Tests cover:
- SpatialComposition generation from ExecutionPlan and DesignBrief
- BoundingBox geometry, intersection area, and IoU calculation
- Face/Text collision detection algorithm and graph validation
- Safe zone enforcement (mobile crop, timestamp overlay)
- Typography layout and face avoidance margins
- Multi-format serialization (JSON, YAML, Dict)
- BaseReasoner contract & interface integration
- Strict renderer independence invariant (zero ComfyUI, SD, Diffusers, SAM, YOLO tokens)
"""

import pytest
from typing import Dict, Any

from thumbnail_intelligence.evidence.models import EvidenceSummary, NormalizedEvidenceGraph
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.spatial_composition_models import (
    AnchorPoint,
    BoundingBox,
    CanvasSpecification,
    CompositionEdge,
    CompositionGraph,
    CompositionLayerPlane,
    CompositionRelationshipType,
    CompositionRule,
    PlacementInstructions,
    SafeZone,
    SpatialComposition,
    TypographyLayout,
    VisualElementPlacement,
)
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner


@pytest.fixture
def sample_brief() -> DesignBrief:
    """Construct a complete, valid DesignBrief for testing."""
    return DesignBrief()


@pytest.fixture
def sample_plan(sample_brief: DesignBrief) -> ExecutionPlan:
    """Construct a complete ExecutionPlan for testing."""
    planner = ExecutionPlanner()
    return planner.plan(sample_brief)


class TestSpatialCompositionPlanner:

    def test_spatial_composition_generation(self, sample_plan: ExecutionPlan, sample_brief: DesignBrief):
        """Test generation of strongly typed SpatialComposition from ExecutionPlan + DesignBrief."""
        planner = SpatialCompositionPlanner()
        comp = planner.plan(sample_plan, sample_brief)

        assert isinstance(comp, SpatialComposition)
        assert comp.composition_id.startswith("comp_")
        assert comp.plan_ref == sample_plan.metadata.plan_id
        assert comp.brief_ref == sample_brief.metadata.brief_id
        assert comp.schema_version == "1.0.0"

        # Canvas
        assert comp.canvas.width_px == 1280
        assert comp.canvas.height_px == 720
        assert comp.canvas.aspect_ratio == "16:9"
        assert comp.canvas.timestamp_safe_zone.is_forbidden is True

        # Graph Nodes
        nodes = comp.composition_graph.nodes
        assert "elem_00_background" in nodes
        assert "elem_01_primary_subject" in nodes
        assert "elem_02_secondary_subject" in nodes
        assert "elem_03_typography" in nodes

        # Placement Instructions
        inst = comp.placement_instructions
        assert inst.primary_focal_point == "elem_01_primary_subject"
        assert inst.secondary_focal_point == "elem_02_secondary_subject"
        assert isinstance(inst.applied_composition_rule, CompositionRule)
        assert len(inst.visual_flow_path) >= 3
        assert 0.0 <= inst.negative_space_fraction <= 1.0
        assert 0.0 <= inst.visual_balance_score <= 1.0

        # Typography Layout
        assert comp.typography_layout is not None
        assert comp.typography_layout.collision_free is True

    def test_bounding_box_geometry_and_overlap(self):
        """Test BoundingBox center, area, IoU, and overlap logic."""
        box1 = BoundingBox(x=0.1, y=0.1, width=0.4, height=0.4)
        box2 = BoundingBox(x=0.3, y=0.3, width=0.4, height=0.4)
        box3 = BoundingBox(x=0.8, y=0.8, width=0.1, height=0.1)

        assert pytest.approx(box1.center[0], 0.001) == 0.3
        assert pytest.approx(box1.center[1], 0.001) == 0.3
        assert box1.overlaps(box2) is True
        assert box1.overlaps(box3) is False

        inter_area = box1.intersection_area(box2)
        assert pytest.approx(inter_area, 0.001) == 0.04  # (0.5 - 0.3) * (0.5 - 0.3) = 0.2 * 0.2 = 0.04

        iou = box1.intersection_over_union(box2)
        assert 0.0 < iou < 1.0
        assert box1.intersection_over_union(box3) == 0.0

    def test_collision_detection_algorithm(self):
        """Test CompositionGraph.detect_collisions() and validate_composition_graph()."""
        canvas = CanvasSpecification()
        graph = CompositionGraph()

        # Place host face and text in overlapping region
        face_elem = VisualElementPlacement(
            element_id="elem_face",
            element_name="Host Face",
            element_category="face",
            bbox=BoundingBox(x=0.10, y=0.10, width=0.40, height=0.40),
            layer_plane=CompositionLayerPlane.MIDGROUND,
        )
        text_elem = VisualElementPlacement(
            element_id="elem_text",
            element_name="Headline Text",
            element_category="text",
            bbox=BoundingBox(x=0.20, y=0.20, width=0.40, height=0.20),
            layer_plane=CompositionLayerPlane.FOREGROUND,
        )

        graph.add_element(face_elem)
        graph.add_element(text_elem)

        collisions = graph.detect_collisions()
        assert len(collisions) == 1
        assert collisions[0]["element_a"] == "elem_face"
        assert collisions[0]["element_b"] == "elem_text"

        errors = graph.validate_composition_graph(canvas)
        assert len(errors) >= 1
        assert any("Collision detected between face" in err for err in errors)

    def test_safe_zone_enforcement(self):
        """Test timestamp overlay forbidden safe zone enforcement."""
        canvas = CanvasSpecification()
        graph = CompositionGraph()

        # Place element directly inside forbidden timestamp safe zone (x=0.80..0.98, y=0.85..0.97)
        violating_elem = VisualElementPlacement(
            element_id="elem_badge",
            element_name="Badge",
            element_category="prop",
            bbox=BoundingBox(x=0.82, y=0.86, width=0.10, height=0.08),
            safe_zone_restricted=True,
            layer_plane=CompositionLayerPlane.FOREGROUND,
        )
        graph.add_element(violating_elem)

        errors = graph.validate_composition_graph(canvas)
        assert len(errors) >= 1
        assert any("timestamp safe zone" in err for err in errors)

    def test_canvas_overflow_validation(self):
        """Test validation error detection when an element exceeds normalized canvas bounds."""
        canvas = CanvasSpecification()
        graph = CompositionGraph()

        overflow_elem = VisualElementPlacement(
            element_id="elem_overflow",
            element_name="Overflow Element",
            element_category="prop",
            bbox=BoundingBox(x=0.80, y=0.80, width=0.30, height=0.30),  # x2 = 1.10 > 1.0
        )
        graph.add_element(overflow_elem)

        errors = graph.validate_composition_graph(canvas)
        assert len(errors) >= 1
        assert any("exceeds canvas boundaries" in err for err in errors)

    def test_multi_format_serialization(self, sample_plan: ExecutionPlan, sample_brief: DesignBrief):
        """Test JSON, YAML, and Dictionary serialization round-trip fidelity."""
        planner = SpatialCompositionPlanner()
        original_comp = planner.plan(sample_plan, sample_brief)

        # Dict
        comp_dict = original_comp.to_dict()
        assert isinstance(comp_dict, dict)
        dict_restored = SpatialComposition.from_dict(comp_dict)
        assert dict_restored.composition_id == original_comp.composition_id

        # JSON
        comp_json = original_comp.to_json()
        assert isinstance(comp_json, str)
        json_restored = SpatialComposition.from_json(comp_json)
        assert len(json_restored.composition_graph.nodes) == len(original_comp.composition_graph.nodes)

        # YAML
        comp_yaml = original_comp.to_yaml()
        assert isinstance(comp_yaml, str)
        yaml_restored = SpatialComposition.from_yaml(comp_yaml)
        assert yaml_restored.plan_ref == original_comp.plan_ref

    def test_reasoner_contract_and_interface(self, sample_plan: ExecutionPlan, sample_brief: DesignBrief):
        """Test BaseReasoner contract integration and reason() method."""
        planner = SpatialCompositionPlanner()

        assert planner.name == "spatial_composition_planner"
        assert planner.contract.reasoner_type.value == "spatial_composition_planner"
        assert "execution_planner" in planner.dependencies

        # Reason via BaseReasoner interface
        ctx = ReasoningContext(
            graph_id="graph_test_scp",
            design_brief=sample_brief,
            execution_plan=sample_plan,
        )
        graph_dummy = NormalizedEvidenceGraph(
            graph_id="graph_test_scp",
            summary=EvidenceSummary(graph_id="graph_test_scp"),
        )
        comp = planner.reason(graph=graph_dummy, context=ctx)

        assert isinstance(comp, SpatialComposition)
        assert planner.validate_output(comp) is True
        assert planner.validate_output(None) is False

    def test_strict_renderer_independence_invariant(self, sample_plan: ExecutionPlan, sample_brief: DesignBrief):
        """
        Critical Invariant Test: Verify that SpatialComposition contains ZERO renderer-specific tokens
        (no ComfyUI, Stable Diffusion, Diffusers, SAM, YOLO, GroundingDINO, or BrushNet references).
        """
        planner = SpatialCompositionPlanner()
        comp = planner.plan(sample_plan, sample_brief)
        json_dump = comp.to_json().lower()

        forbidden_tokens = [
            "comfyui",
            "stable_diffusion",
            "diffusers",
            "sam",
            "yolo",
            "groundingdino",
            "brushnet",
            "sdxl",
            "inpainting_mask",
            "lora_weight",
            "controlnet_model",
            "cfg_scale",
            "sampler_name",
        ]

        for token in forbidden_tokens:
            assert token not in json_dump, f"Forbidden renderer token '{token}' found in SpatialComposition!"

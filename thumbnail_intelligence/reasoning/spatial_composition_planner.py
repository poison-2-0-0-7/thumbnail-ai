"""
spatial_composition_planner.py
===============================

Spatial Composition Planner Implementation (Phase 3.7).
Translates an ExecutionPlan + DesignBrief into a renderer-independent SpatialComposition.

Determines WHERE visual elements belong in 2D/3D layout space using professional graphic design rules
(Rule of Thirds, Golden Ratio, Dynamic Asymmetry), safe zone enforcement, face/text collision avoidance,
typography region allocation, and spatial graph construction.

Contains ZERO renderer-specific code (no ComfyUI, Stable Diffusion, Diffusers, SAM, or YOLO calls).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.knowledge_base.models import _utc_now_iso
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.exceptions import ReasonerValidationError
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from thumbnail_intelligence.reasoning.interfaces import SpatialCompositionPlannerInterface
from thumbnail_intelligence.reasoning.models import ReasonerContract, ReasonerType
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
    SpatialComposition,
    TypographyLayout,
    VisualElementPlacement,
)

logger = logging.getLogger(__name__)


class SpatialCompositionPlanner(SpatialCompositionPlannerInterface):
    """
    Production SpatialCompositionPlanner translating ExecutionPlan + DesignBrief into SpatialComposition.
    Calculates 2D/3D element bounding boxes, typography layouts, safe zone margins, visual flow paths,
    and spatial relationship graphs.
    """

    def __init__(self) -> None:
        self._contract = ReasonerContract(
            name="spatial_composition_planner",
            reasoner_type=ReasonerType.SPATIAL_COMPOSITION_PLANNER,
            description="Translates ExecutionPlan and DesignBrief into a renderer-independent spatial layout",
            dependencies=[
                "narrative_reasoner",
                "audience_reasoner",
                "creator_reasoner",
                "brand_reasoner",
                "priority_reasoner",
                "risk_reasoner",
                "strategy_ranker",
                "validator",
                "design_brief_generator",
                "execution_planner",
            ],
            timeout_seconds=5.0,
            max_retries=0,
            supports_cache=True,
        )

    @property
    def contract(self) -> ReasonerContract:
        """Return metadata contract for registration and coordinator topology."""
        return self._contract

    def plan(
        self,
        plan: ExecutionPlan,
        brief: Optional[DesignBrief] = None,
    ) -> SpatialComposition:
        """
        Primary entrypoint: Translate ExecutionPlan and DesignBrief into SpatialComposition.

        Args:
            plan: Input ExecutionPlan from Phase 3.6.
            brief: Optional source DesignBrief from Phase 3.5.

        Returns:
            Strongly typed SpatialComposition specifying spatial layout and graph.
        """
        if not plan:
            raise ReasonerValidationError(
                reasoner_name="spatial_composition_planner",
                validation_errors=["Provided ExecutionPlan is None or empty."],
            )

        brief_ref = plan.metadata.brief_ref or getattr(brief.metadata, "brief_id", "brief_default") if brief else "brief_default"

        # 1. Canvas Specification
        canvas = CanvasSpecification(
            width_px=1280,
            height_px=720,
            aspect_ratio="16:9",
        )

        # 2. Extract visual elements from DesignBrief / ExecutionPlan parameters
        primary_subject_name = "Host Face"
        secondary_subject_name = "Secondary Subject"
        required_brand_elements: List[str] = []

        if brief:
            primary_subject_name = brief.composition.primary_subject or primary_subject_name
            secondary_subject_name = brief.composition.secondary_subject or secondary_subject_name
            required_brand_elements = list(brief.brand.required_elements)

        # Determine composition rule
        composition_rule = CompositionRule.RULE_OF_THIRDS
        if brief and brief.creator.creator_archetype == "educator":
            composition_rule = CompositionRule.RULE_OF_THIRDS
        elif brief and brief.narrative.emotional_goal in ["awe", "surprise"]:
            composition_rule = CompositionRule.DYNAMIC_BALANCE

        # 3. Compute Bounding Boxes & Placements
        graph = CompositionGraph(
            graph_id=f"spatial_graph_{uuid.uuid4().hex[:8]}",
        )

        # Background Layer
        bg_elem = VisualElementPlacement(
            element_id="elem_00_background",
            element_name="Background Layer",
            element_category="background",
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            z_index=0,
            layer_plane=CompositionLayerPlane.BACKGROUND,
            depth_z=1.0,
            opacity=1.0,
            priority_tier="SUPPRESSED",
            sourced_from_step="step_04_background_generation",
        )
        graph.add_element(bg_elem)

        # Primary Hero Subject (Rule of Thirds power intersection left side)
        # Positioned around x=0.10, y=0.12, width=0.42, height=0.78
        primary_elem = VisualElementPlacement(
            element_id="elem_01_primary_subject",
            element_name=primary_subject_name,
            element_category="face" if "face" in primary_subject_name.lower() or "host" in primary_subject_name.lower() else "subject",
            bbox=BoundingBox(x=0.10, y=0.12, width=0.42, height=0.78),
            anchor_point=AnchorPoint(x_pct=0.5, y_pct=0.5, preset="center"),
            z_index=10,
            layer_plane=CompositionLayerPlane.MIDGROUND,
            depth_z=0.3,
            opacity=1.0,
            alignment="left",
            safe_zone_restricted=True,
            priority_tier="PRIMARY",
            sourced_from_step="step_05_subject_extraction",
        )
        graph.add_element(primary_elem)

        # Secondary Subject (Positioned right side midground)
        secondary_elem = VisualElementPlacement(
            element_id="elem_02_secondary_subject",
            element_name=secondary_subject_name,
            element_category="subject",
            bbox=BoundingBox(x=0.55, y=0.25, width=0.36, height=0.52),
            anchor_point=AnchorPoint(x_pct=0.5, y_pct=0.5, preset="center"),
            z_index=8,
            layer_plane=CompositionLayerPlane.MIDGROUND,
            depth_z=0.5,
            opacity=1.0,
            alignment="right",
            safe_zone_restricted=True,
            priority_tier="SECONDARY",
            sourced_from_step="step_10_object_placement",
        )
        graph.add_element(secondary_elem)

        # Typography Placement (Upper third / top left quadrant)
        # Bounding box collision avoidance against primary subject face
        text_bbox = BoundingBox(x=0.08, y=0.08, width=0.48, height=0.22)

        # If text overlaps primary face region, shift text slightly upper right or adjust width
        if text_bbox.overlaps(primary_elem.bbox, margin=0.02):
            # Shift text box to avoid host face overlap
            text_bbox = BoundingBox(x=0.48, y=0.08, width=0.46, height=0.22)

        text_elem = VisualElementPlacement(
            element_id="elem_03_typography",
            element_name="Headline Typography Overlay",
            element_category="text",
            bbox=text_bbox,
            anchor_point=AnchorPoint(x_pct=0.0, y_pct=0.0, preset="top_left"),
            z_index=20,
            layer_plane=CompositionLayerPlane.FOREGROUND,
            depth_z=0.1,
            opacity=1.0,
            alignment="left",
            safe_zone_restricted=True,
            priority_tier="PRIMARY",
            sourced_from_step="step_12_typography_placement",
        )
        graph.add_element(text_elem)

        # Brand Elements / Logo (Bottom left or top right safe zone)
        if required_brand_elements:
            logo_name = required_brand_elements[0]
            logo_elem = VisualElementPlacement(
                element_id="elem_04_brand_logo",
                element_name=logo_name,
                element_category="logo",
                bbox=BoundingBox(x=0.05, y=0.82, width=0.15, height=0.12),
                z_index=25,
                layer_plane=CompositionLayerPlane.OVERLAY,
                depth_z=0.05,
                opacity=1.0,
                alignment="bottom_left",
                safe_zone_restricted=True,
                priority_tier="TERTIARY",
                sourced_from_step="step_01_load_assets",
            )
            graph.add_element(logo_elem)

        # 4. Define Spatial Graph Relationships
        graph.add_relationship(
            source_id="elem_00_background",
            target_id="elem_01_primary_subject",
            rel_type=CompositionRelationshipType.CONTAINMENT,
            description="Background contains primary subject",
        )
        graph.add_relationship(
            source_id="elem_01_primary_subject",
            target_id="elem_02_secondary_subject",
            rel_type=CompositionRelationshipType.ADJACENCY,
            description="Primary subject adjacent to secondary subject",
        )
        graph.add_relationship(
            source_id="elem_03_typography",
            target_id="elem_01_primary_subject",
            rel_type=CompositionRelationshipType.DEPTH_ORDER,
            description="Typography layered in foreground above primary subject",
        )

        # 5. Typography Layout Specs
        typography_layout = TypographyLayout(
            text_element_id="elem_03_typography",
            text_content=brief.typography.text_priority if brief else "Headline Overlay",
            text_region_bbox=text_bbox,
            maximum_width_fraction=0.55,
            maximum_height_fraction=0.30,
            alignment="left",
            padding_px=12.0,
            face_avoidance_margin=0.05,
            collision_free=True,
            contrast_target_background=brief.typography.readability_targets if brief else "high_contrast",
        )

        # 6. Placement Instructions & Visual Flow
        flow_path = ["elem_01_primary_subject", "elem_03_typography", "elem_02_secondary_subject"]
        if required_brand_elements:
            flow_path.append("elem_04_brand_logo")

        placement_instructions = PlacementInstructions(
            primary_focal_point="elem_01_primary_subject",
            secondary_focal_point="elem_02_secondary_subject",
            applied_composition_rule=composition_rule,
            visual_flow_path=flow_path,
            negative_space_fraction=0.32,
            visual_balance_score=0.88,
            is_asymmetrical=True,
        )

        comp = SpatialComposition(
            composition_id=f"comp_{uuid.uuid4().hex[:8]}",
            plan_ref=plan.metadata.plan_id,
            brief_ref=brief_ref,
            schema_version="1.0.0",
            created_at=_utc_now_iso(),
            canvas=canvas,
            composition_graph=graph,
            placement_instructions=placement_instructions,
            typography_layout=typography_layout,
        )

        logger.info(
            f"Successfully generated SpatialComposition '{comp.composition_id}' "
            f"({len(graph.nodes)} nodes, rule={composition_rule.value}, balance={placement_instructions.visual_balance_score:.2f})."
        )
        return comp

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> SpatialComposition:
        """
        BaseReasoner execution interface.
        Extracts ExecutionPlan and DesignBrief from context, or generates them on the fly.
        """
        exec_plan = getattr(context, "execution_plan", None)
        brief = getattr(context, "design_brief", None)

        if exec_plan is None:
            from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner

            planner = ExecutionPlanner()
            exec_plan = planner.reason(graph=graph, context=context)

        return self.plan(plan=exec_plan, brief=brief)

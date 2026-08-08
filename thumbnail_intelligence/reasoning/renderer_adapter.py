"""
renderer_adapter.py
===================

Renderer Adapter Engine Implementation (Phase 3.8).
Translates a renderer-independent SpatialComposition + ExecutionPlan into a renderer-specific RenderExecutionPackage.

This engine is the ONLY component that knows about target renderers (Renderer V2, ComfyUI, Flux, Imagen, Custom).
The Intelligence Engine's internal objects (DesignBrief, ReasoningContext, ExecutionPlan, SpatialComposition)
MUST NEVER be passed directly to rendering engines.

Contains ZERO rendering code (no pixel generation, no model execution). Performs pure translation.
"""

from __future__ import annotations

import logging
import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.knowledge_base.models import _utc_now_iso
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.exceptions import ReasonerValidationError
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan, ExecutionStepType
from thumbnail_intelligence.reasoning.interfaces import RendererAdapterInterface
from thumbnail_intelligence.reasoning.models import ReasonerContract, ReasonerType
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
    RenderSceneGraphNode,
    RenderTypographyInstruction,
)
from thumbnail_intelligence.reasoning.spatial_composition_models import CompositionLayerPlane, SpatialComposition

logger = logging.getLogger(__name__)


# Operation type mapping from ExecutionStepType -> RenderOperationType
_STEP_TO_RENDER_OP: Dict[ExecutionStepType, RenderOperationType] = {
    ExecutionStepType.LOAD_ASSETS: RenderOperationType.LOAD_ASSET,
    ExecutionStepType.PREPARE_CANVAS: RenderOperationType.PREPARE_CANVAS,
    ExecutionStepType.BACKGROUND_PLANNING: RenderOperationType.GENERATE_BACKGROUND,
    ExecutionStepType.BACKGROUND_GENERATION: RenderOperationType.GENERATE_BACKGROUND,
    ExecutionStepType.SUBJECT_EXTRACTION: RenderOperationType.EXTRACT_SUBJECT,
    ExecutionStepType.SUBJECT_ENHANCEMENT: RenderOperationType.ENHANCE_SUBJECT,
    ExecutionStepType.LIGHTING: RenderOperationType.APPLY_LIGHTING,
    ExecutionStepType.SHADOW: RenderOperationType.GENERATE_SHADOW,
    ExecutionStepType.COMPOSITION: RenderOperationType.COMPOSE_LAYER,
    ExecutionStepType.OBJECT_PLACEMENT: RenderOperationType.COMPOSE_LAYER,
    ExecutionStepType.TYPOGRAPHY_PLANNING: RenderOperationType.RENDER_TYPOGRAPHY,
    ExecutionStepType.TYPOGRAPHY_PLACEMENT: RenderOperationType.RENDER_TYPOGRAPHY,
    ExecutionStepType.COLOR_HARMONIZATION: RenderOperationType.APPLY_COLOR_GRADE,
    ExecutionStepType.CONTRAST_ADJUSTMENT: RenderOperationType.ADJUST_CONTRAST,
    ExecutionStepType.VALIDATION: RenderOperationType.EVALUATE_QUALITY,
    ExecutionStepType.FINAL_COMPOSITE: RenderOperationType.COMPOSITE_FINAL,
    ExecutionStepType.CLEANUP: RenderOperationType.CLEANUP_BUFFERS,
}


class BaseRendererAdapter(RendererAdapterInterface):
    """Abstract base class for all renderer translation adapters."""

    def __init__(self, target_renderer_id: str = "RendererV2") -> None:
        self._target_renderer_id = target_renderer_id
        self._contract = ReasonerContract(
            name=f"renderer_adapter_{target_renderer_id.lower()}",
            reasoner_type=ReasonerType.RENDERER_ADAPTER,
            description=f"Translates SpatialComposition into RenderExecutionPackage targeting {target_renderer_id}",
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
                "spatial_composition_planner",
            ],
            timeout_seconds=5.0,
            max_retries=0,
            supports_cache=True,
        )

    @property
    def contract(self) -> ReasonerContract:
        """Return metadata contract for registry integration."""
        return self._contract

    @property
    def target_renderer_id(self) -> str:
        """Return the target renderer engine name."""
        return self._target_renderer_id

    @abstractmethod
    def translate(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
    ) -> RenderExecutionPackage:
        """Abstract translation method."""
        ...

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> RenderExecutionPackage:
        """BaseReasoner interface implementation."""
        spatial_comp = getattr(context, "spatial_composition", None)
        exec_plan = getattr(context, "execution_plan", None)

        if spatial_comp is None or exec_plan is None:
            from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner

            scp = SpatialCompositionPlanner()
            spatial_comp = scp.reason(graph=graph, context=context)
            exec_plan = getattr(context, "execution_plan", None)
            if exec_plan is None:
                from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner

                ep = ExecutionPlanner()
                exec_plan = ep.reason(graph=graph, context=context)

        return self.translate(composition=spatial_comp, plan=exec_plan)


class RendererV2Adapter(BaseRendererAdapter):
    """Production Renderer V2 Adapter translating SpatialComposition into RenderExecutionPackage."""

    def __init__(self) -> None:
        super().__init__(target_renderer_id="RendererV2")

    def render(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
        output_path: Optional[str] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Translate SpatialComposition + ExecutionPlan and render final thumbnail end-to-end via RendererV2Pipeline."""
        from renderer_v2.pipeline import RendererV2Pipeline

        pipeline = RendererV2Pipeline()
        return pipeline.render(composition, plan, output_path=output_path, context_overrides=context_overrides)

    def translate(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
        target_renderer: Optional[str] = None,
    ) -> RenderExecutionPackage:
        """
        Primary entrypoint: Translate SpatialComposition + ExecutionPlan into RenderExecutionPackage.
        """
        if not composition or not plan:
            raise ReasonerValidationError(
                reasoner_name="renderer_adapter_v2",
                validation_errors=["Provided SpatialComposition or ExecutionPlan is None."],
            )

        canvas_w = composition.canvas.width_px
        canvas_h = composition.canvas.height_px

        # 1. Metadata
        metadata = RenderPackageMetadata(
            package_id=f"pkg_render_{uuid.uuid4().hex[:8]}",
            comp_ref=composition.composition_id,
            plan_ref=plan.metadata.plan_id,
            brief_ref=composition.brief_ref,
            target_renderer=target_renderer or self.target_renderer_id,
            schema_version="1.0.0",
            created_at=_utc_now_iso(),
        )

        # 2. Translate Placements (Normalized -> Pixel Coordinates)
        placements: List[RenderPlacementCoordinate] = []
        layer_stack: List[RenderLayerEntry] = []
        scene_nodes: Dict[str, RenderSceneGraphNode] = {}

        for elem_id, elem in composition.composition_graph.nodes.items():
            b_norm = elem.bbox
            x_px = int(round(b_norm.x * canvas_w))
            y_px = int(round(b_norm.y * canvas_h))
            w_px = max(1, int(round(b_norm.width * canvas_w)))
            h_px = max(1, int(round(b_norm.height * canvas_h)))

            pixel_box = PixelBoundingBox(
                x_px=x_px,
                y_px=y_px,
                width_px=w_px,
                height_px=h_px,
            )

            anchor_x = x_px + elem.anchor_point.x_pct * w_px
            anchor_y = y_px + elem.anchor_point.y_pct * h_px

            p_coord = RenderPlacementCoordinate(
                element_id=elem.element_id,
                element_name=elem.element_name,
                bbox_normalized=(b_norm.x, b_norm.y, b_norm.width, b_norm.height),
                bbox_pixels=pixel_box,
                anchor_x_px=anchor_x,
                anchor_y_px=anchor_y,
                rotation_deg=elem.rotation_deg,
                scale=elem.scale,
                z_index=elem.z_index,
                opacity=elem.opacity,
            )
            placements.append(p_coord)

            # Layer Stack Entry
            layer_entry = RenderLayerEntry(
                layer_id=f"layer_{elem.element_id}",
                layer_name=elem.element_name,
                layer_type=elem.element_category,
                z_index=elem.z_index,
                blend_mode="normal",
                opacity=elem.opacity,
                visible=True,
            )
            layer_stack.append(layer_entry)

            # Scene Graph Node
            scene_node = RenderSceneGraphNode(
                node_id=elem.element_id,
                node_name=elem.element_name,
                placement=p_coord,
            )
            scene_nodes[elem.element_id] = scene_node

        layer_stack.sort(key=lambda l: l.z_index)

        scene_graph = RenderSceneGraph(
            scene_id=f"scene_{uuid.uuid4().hex[:8]}",
            canvas_width_px=canvas_w,
            canvas_height_px=canvas_h,
            nodes=scene_nodes,
        )

        # 3. Translate Asset References
        asset_refs: List[RenderAssetReference] = []
        for step in plan.graph.steps.values():
            for inp in step.inputs:
                if inp.startswith("asset:"):
                    asset_refs.append(
                        RenderAssetReference(
                            asset_id=f"asset_{uuid.uuid4().hex[:6]}",
                            asset_type="image_hero" if "primary" in inp else "asset_generic",
                            source_key=inp,
                            is_required=True,
                        )
                    )

        # 4. Translate Masks
        masks: List[RenderMaskInstruction] = [
            RenderMaskInstruction(
                mask_id=f"mask_{uuid.uuid4().hex[:6]}",
                target_element_id="elem_01_primary_subject",
                mask_type="subject_matte",
                feather_px=2.0,
                invert=False,
            )
        ]

        # 5. Translate Typography Instructions
        typography_instructions: List[RenderTypographyInstruction] = []
        if composition.typography_layout:
            typo = composition.typography_layout
            text_p_coord = next(
                (p for p in placements if p.element_id == typo.text_element_id),
                placements[0] if placements else None,
            )
            if text_p_coord:
                typo_inst = RenderTypographyInstruction(
                    text_id=typo.text_element_id,
                    content=typo.text_content,
                    placement=text_p_coord,
                    font_family="Sans-Serif",
                    font_size_px=48,
                    font_weight="bold",
                    font_color_hex="#FFFFFF",
                    stroke_color_hex="#000000",
                    stroke_width_px=4,
                    drop_shadow_blur_px=8,
                    alignment=typo.alignment,
                    max_word_count=4,
                )
                typography_instructions.append(typo_inst)

        # 6. Background Instruction
        bg_instruction = RenderBackgroundInstruction(
            action="replace",
            style_prompt_direction="modern neon studio",
            dominant_colors=["#0066CC", "#FFFFFF"],
            depth_treatment="shallow",
            sourced_from_step="step_04_background_generation",
        )

        # 7. Lighting Instructions
        lighting_instructions: List[RenderLightingInstruction] = [
            RenderLightingInstruction(
                target_element_id="elem_01_primary_subject",
                mood="high_key_dramatic",
                key_light_direction="top_left",
                key_light_intensity=0.8,
                rim_light_enabled=True,
                rim_light_color_temp=5600,
                shadow_cast_enabled=True,
            )
        ]

        # 8. Translate Render Operations from ExecutionPlan steps
        render_ops: List[RenderOperation] = []
        for step_id in plan.graph.topological_order:
            step = plan.graph.steps[step_id]
            op_type = _STEP_TO_RENDER_OP.get(step.step_type, RenderOperationType.COMPOSE_LAYER)
            op = RenderOperation(
                op_id=f"op_{step.step_id}",
                op_type=op_type,
                target_layer_id=f"layer_{step.outputs[0]}" if step.outputs else "",
                input_keys=list(step.inputs),
                output_keys=list(step.outputs),
                parameters=dict(step.parameters),
                sourced_from_step_id=step.step_id,
            )
            render_ops.append(op)

        pkg = RenderExecutionPackage(
            metadata=metadata,
            scene_graph=scene_graph,
            render_operations=render_ops,
            asset_references=asset_refs,
            masks=masks,
            placement_coordinates=placements,
            typography_instructions=typography_instructions,
            background_instruction=bg_instruction,
            lighting_instructions=lighting_instructions,
            layer_stack=layer_stack,
        )

        logger.info(
            f"Successfully translated SpatialComposition into RenderExecutionPackage '{pkg.metadata.package_id}' "
            f"for target '{metadata.target_renderer}' ({len(placements)} placements, {len(render_ops)} operations)."
        )
        return pkg


class FutureComfyUIAdapter(BaseRendererAdapter):
    """Extensibility Adapter target for ComfyUI workflow backends."""

    def __init__(self) -> None:
        super().__init__(target_renderer_id="ComfyUI")

    def translate(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
    ) -> RenderExecutionPackage:
        """Translate to ComfyUI-compatible RenderExecutionPackage."""
        v2_adapter = RendererV2Adapter()
        return v2_adapter.translate(composition=composition, plan=plan, target_renderer=self.target_renderer_id)


class FutureFluxAdapter(BaseRendererAdapter):
    """Extensibility Adapter target for Flux pipeline backends."""

    def __init__(self) -> None:
        super().__init__(target_renderer_id="Flux")

    def translate(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
    ) -> RenderExecutionPackage:
        """Translate to Flux-compatible RenderExecutionPackage."""
        v2_adapter = RendererV2Adapter()
        return v2_adapter.translate(composition=composition, plan=plan, target_renderer=self.target_renderer_id)


class FutureImagenAdapter(BaseRendererAdapter):
    """Extensibility Adapter target for Imagen rendering backends."""

    def __init__(self) -> None:
        super().__init__(target_renderer_id="Imagen")

    def translate(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
    ) -> RenderExecutionPackage:
        """Translate to Imagen-compatible RenderExecutionPackage."""
        v2_adapter = RendererV2Adapter()
        return v2_adapter.translate(composition=composition, plan=plan, target_renderer=self.target_renderer_id)


class FutureCustomAdapter(BaseRendererAdapter):
    """Extensibility Adapter target for custom rendering backends."""

    def __init__(self, custom_renderer_id: str = "CustomRenderer") -> None:
        super().__init__(target_renderer_id=custom_renderer_id)

    def translate(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
    ) -> RenderExecutionPackage:
        """Translate to custom RenderExecutionPackage."""
        v2_adapter = RendererV2Adapter()
        return v2_adapter.translate(composition=composition, plan=plan, target_renderer=self.target_renderer_id)


# Concrete alias for primary production adapter
RendererAdapter = RendererV2Adapter

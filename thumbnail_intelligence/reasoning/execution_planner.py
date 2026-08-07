"""
execution_planner.py
====================

Execution Planner Engine (Phase 3.6).
Translates a DesignBrief into a deterministic, renderer-agnostic ExecutionPlan.

The ExecutionPlanner computes:
- WHAT operations happen (17 step taxonomy)
- WHEN they happen (Directed Acyclic Graph)
- IN WHAT ORDER (topological ordering and parallel stages)
- RESOURCE BUDGETS (GPU VRAM peak, CPU usage, latency, disk, cost)

Contains ZERO renderer-specific code (no ComfyUI, Stable Diffusion, SAM, YOLO, or SDXL calls).
It builds the execution graph only.
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
from thumbnail_intelligence.reasoning.execution_plan_models import (
    ExecutionGraph,
    ExecutionMetadata,
    ExecutionPlan,
    ExecutionStep,
    ExecutionStepType,
    ResourceEstimates,
    RetryPolicy,
)
from thumbnail_intelligence.reasoning.interfaces import ExecutionPlannerInterface
from thumbnail_intelligence.reasoning.models import ReasonerContract, ReasonerType

logger = logging.getLogger(__name__)


class ExecutionPlanner(ExecutionPlannerInterface):
    """
    Production ExecutionPlanner translating a DesignBrief into an ExecutionPlan DAG.
    Calculates execution stages, topological dependency order, and resource budgets.
    """

    def __init__(self) -> None:
        self._contract = ReasonerContract(
            name="execution_planner",
            reasoner_type=ReasonerType.EXECUTION_PLANNER,
            description="Translates a DesignBrief into a deterministic execution graph and schedule",
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
            ],
            timeout_seconds=5.0,
            max_retries=0,
            supports_cache=True,
        )

    @property
    def contract(self) -> ReasonerContract:
        """Return metadata contract for registration and topological resolution."""
        return self._contract

    def plan(self, brief: DesignBrief) -> ExecutionPlan:
        """
        Primary entrypoint: Translate a DesignBrief into an ExecutionPlan.

        Args:
            brief: Strongly typed DesignBrief.

        Returns:
            Deterministic ExecutionPlan containing ExecutionGraph and resource schedules.
        """
        if not brief:
            raise ReasonerValidationError(
                reasoner_name="execution_planner",
                validation_errors=["Provided DesignBrief is None or empty."],
            )

        steps: Dict[str, ExecutionStep] = {}

        # 1. LOAD_ASSETS
        steps["step_01_load_assets"] = ExecutionStep(
            step_id="step_01_load_assets",
            step_type=ExecutionStepType.LOAD_ASSETS,
            description="Load hero subject, secondary subjects, and brand asset manifests",
            inputs=["brief:composition", "brief:brand"],
            outputs=["asset:primary_subject", "asset:brand_elements"],
            dependencies=[],
            resources=ResourceEstimates(vram_mb=256.0, cpu_usage_pct=20.0, estimated_runtime_ms=150.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=3),
            sourced_from_brief_field="composition.primary_subject",
            parameters={
                "primary_subject": brief.composition.primary_subject,
                "secondary_subject": brief.composition.secondary_subject,
                "required_brand_elements": brief.brand.required_elements,
            },
        )

        # 2. PREPARE_CANVAS
        steps["step_02_prepare_canvas"] = ExecutionStep(
            step_id="step_02_prepare_canvas",
            step_type=ExecutionStepType.PREPARE_CANVAS,
            description="Initialize target canvas frame and reserve UI overlay safe zones",
            inputs=["asset:primary_subject", "brief:camera"],
            outputs=["canvas:base_frame", "canvas:safe_zones"],
            dependencies=["step_01_load_assets"],
            resources=ResourceEstimates(vram_mb=128.0, cpu_usage_pct=10.0, estimated_runtime_ms=50.0, estimated_cost=0.5),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="composition.safe_zones",
            parameters={
                "safe_zones": brief.composition.safe_zones,
                "aspect_ratio": "16:9",
                "subject_scale": brief.camera.subject_scale,
            },
        )

        # 3. BACKGROUND_PLANNING
        steps["step_03_background_planning"] = ExecutionStep(
            step_id="step_03_background_planning",
            step_type=ExecutionStepType.BACKGROUND_PLANNING,
            description="Plan background depth separation and negative space allocations",
            inputs=["canvas:base_frame", "brief:color"],
            outputs=["plan:background_depth", "plan:negative_space"],
            dependencies=["step_02_prepare_canvas"],
            resources=ResourceEstimates(vram_mb=64.0, cpu_usage_pct=15.0, estimated_runtime_ms=80.0, estimated_cost=0.5),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="composition.negative_space",
            parameters={
                "negative_space_target": brief.composition.negative_space,
                "depth_treatment": brief.composition.depth_treatment,
            },
        )

        # 4. BACKGROUND_GENERATION
        steps["step_04_background_generation"] = ExecutionStep(
            step_id="step_04_background_generation",
            step_type=ExecutionStepType.BACKGROUND_GENERATION,
            description="Synthesize background layer conforming to color palette and depth goals",
            inputs=["plan:background_depth", "brief:color"],
            outputs=["layer:background_render"],
            dependencies=["step_03_background_planning"],
            resources=ResourceEstimates(
                vram_mb=1024.0, cpu_usage_pct=40.0, estimated_runtime_ms=600.0, model_loading_overhead_ms=200.0, estimated_cost=3.5
            ),
            retry_policy=RetryPolicy(max_retries=3, retry_on_vram_oom=True),
            sourced_from_brief_field="color.primary_palette",
            parameters={
                "primary_palette": brief.color.primary_palette,
                "accent_palette": brief.color.accent_palette,
                "style_direction": brief.creator.channel_voice or "modern studio",
            },
        )

        # 5. SUBJECT_EXTRACTION
        steps["step_05_subject_extraction"] = ExecutionStep(
            step_id="step_05_subject_extraction",
            step_type=ExecutionStepType.SUBJECT_EXTRACTION,
            description="Extract subject matte and isolate primary hero element",
            inputs=["asset:primary_subject"],
            outputs=["layer:subject_matte"],
            dependencies=["step_01_load_assets"],
            resources=ResourceEstimates(vram_mb=512.0, cpu_usage_pct=30.0, estimated_runtime_ms=250.0, estimated_cost=2.0),
            retry_policy=RetryPolicy(max_retries=3),
            sourced_from_brief_field="composition.primary_subject",
            parameters={"subject_name": brief.composition.primary_subject},
        )

        # 6. SUBJECT_ENHANCEMENT
        steps["step_06_subject_enhancement"] = ExecutionStep(
            step_id="step_06_subject_enhancement",
            step_type=ExecutionStepType.SUBJECT_ENHANCEMENT,
            description="Apply subject framing crop, perspective adjustment, and detail sharpening",
            inputs=["layer:subject_matte", "brief:camera"],
            outputs=["layer:subject_enhanced"],
            dependencies=["step_05_subject_extraction"],
            resources=ResourceEstimates(vram_mb=512.0, cpu_usage_pct=25.0, estimated_runtime_ms=300.0, estimated_cost=2.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="camera.crop",
            parameters={
                "crop": brief.camera.crop,
                "perspective": brief.camera.perspective,
                "zoom": brief.camera.zoom,
            },
        )

        # 7. LIGHTING
        steps["step_07_lighting"] = ExecutionStep(
            step_id="step_07_lighting",
            step_type=ExecutionStepType.LIGHTING,
            description="Relight subject to match background illumination mood and key direction",
            inputs=["layer:background_render", "layer:subject_enhanced", "brief:lighting"],
            outputs=["layer:subject_relit"],
            dependencies=["step_04_background_generation", "step_06_subject_enhancement"],
            resources=ResourceEstimates(vram_mb=768.0, cpu_usage_pct=35.0, estimated_runtime_ms=400.0, estimated_cost=2.5),
            retry_policy=RetryPolicy(max_retries=3, retry_on_vram_oom=True),
            sourced_from_brief_field="lighting.mood",
            parameters={
                "mood": brief.lighting.mood,
                "direction": brief.lighting.direction,
                "intensity": brief.lighting.intensity,
            },
        )

        # 8. SHADOW
        steps["step_08_shadow"] = ExecutionStep(
            step_id="step_08_shadow",
            step_type=ExecutionStepType.SHADOW,
            description="Synthesize ground contact and cast shadows synchronized with relighting direction",
            inputs=["layer:subject_relit", "brief:lighting"],
            outputs=["layer:shadow_render"],
            dependencies=["step_07_lighting"],
            resources=ResourceEstimates(vram_mb=256.0, cpu_usage_pct=15.0, estimated_runtime_ms=120.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="lighting.direction",
            parameters={"light_direction": brief.lighting.direction},
        )

        # 9. COMPOSITION
        steps["step_09_composition"] = ExecutionStep(
            step_id="step_09_composition",
            step_type=ExecutionStepType.COMPOSITION,
            description="Assemble focal points and visual hierarchy nodes into spatial layout",
            inputs=["layer:subject_relit", "layer:shadow_render", "brief:composition"],
            outputs=["composite:scene_assembled"],
            dependencies=["step_06_subject_enhancement", "step_07_lighting", "step_08_shadow"],
            resources=ResourceEstimates(vram_mb=256.0, cpu_usage_pct=20.0, estimated_runtime_ms=150.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="composition.visual_hierarchy",
            parameters={"hierarchy_nodes": brief.composition.visual_hierarchy},
        )

        # 10. OBJECT_PLACEMENT
        steps["step_10_object_placement"] = ExecutionStep(
            step_id="step_10_object_placement",
            step_type=ExecutionStepType.OBJECT_PLACEMENT,
            description="Place required and optional props into designated secondary focal zones",
            inputs=["composite:scene_assembled", "brief:objects"],
            outputs=["composite:objects_placed"],
            dependencies=["step_09_composition"],
            resources=ResourceEstimates(vram_mb=256.0, cpu_usage_pct=15.0, estimated_runtime_ms=100.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="objects.required_objects",
            parameters={
                "required_objects": brief.objects.required_objects,
                "optional_objects": brief.objects.optional_objects,
            },
        )

        # 11. TYPOGRAPHY_PLANNING
        steps["step_11_typography_planning"] = ExecutionStep(
            step_id="step_11_typography_planning",
            step_type=ExecutionStepType.TYPOGRAPHY_PLANNING,
            description="Plan typography character budget, word count, and safe region constraints",
            inputs=["composite:scene_assembled", "brief:typography"],
            outputs=["plan:text_layout"],
            dependencies=["step_09_composition"],
            resources=ResourceEstimates(vram_mb=64.0, cpu_usage_pct=10.0, estimated_runtime_ms=60.0, estimated_cost=0.5),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="typography.maximum_characters",
            parameters={
                "maximum_characters": brief.typography.maximum_characters,
                "max_word_count": brief.typography.max_word_count,
                "text_regions": brief.typography.text_regions,
            },
        )

        # 12. TYPOGRAPHY_PLACEMENT
        steps["step_12_typography_placement"] = ExecutionStep(
            step_id="step_12_typography_placement",
            step_type=ExecutionStepType.TYPOGRAPHY_PLACEMENT,
            description="Place typography overlay ensuring high mobile contrast and zero face overlap",
            inputs=["plan:text_layout", "layer:background_render", "brief:typography"],
            outputs=["layer:typography_render"],
            dependencies=["step_11_typography_planning", "step_04_background_generation"],
            resources=ResourceEstimates(vram_mb=128.0, cpu_usage_pct=15.0, estimated_runtime_ms=180.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="typography.readability_targets",
            parameters={
                "readability_targets": brief.typography.readability_targets,
                "text_priority": brief.typography.text_priority,
            },
        )

        # 13. COLOR_HARMONIZATION
        steps["step_13_color_harmonization"] = ExecutionStep(
            step_id="step_13_color_harmonization",
            step_type=ExecutionStepType.COLOR_HARMONIZATION,
            description="Harmonize scene palette, accent pops, and brand color alignment",
            inputs=["composite:objects_placed", "layer:typography_render", "brief:color"],
            outputs=["composite:color_harmonized"],
            dependencies=["step_10_object_placement", "step_12_typography_placement"],
            resources=ResourceEstimates(vram_mb=128.0, cpu_usage_pct=15.0, estimated_runtime_ms=100.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="color.accent_palette",
            parameters={"accent_palette": brief.color.accent_palette, "brand_colors": brief.color.brand_colors},
        )

        # 14. CONTRAST_ADJUSTMENT
        steps["step_14_contrast_adjustment"] = ExecutionStep(
            step_id="step_14_contrast_adjustment",
            step_type=ExecutionStepType.CONTRAST_ADJUSTMENT,
            description="Apply luminance contrast targets for mobile feed legibility",
            inputs=["composite:color_harmonized", "brief:color"],
            outputs=["composite:contrast_adjusted"],
            dependencies=["step_13_color_harmonization"],
            resources=ResourceEstimates(vram_mb=64.0, cpu_usage_pct=10.0, estimated_runtime_ms=80.0, estimated_cost=0.5),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="color.contrast_targets",
            parameters={"contrast_targets": brief.color.contrast_targets},
        )

        # 15. VALIDATION
        steps["step_15_validation"] = ExecutionStep(
            step_id="step_15_validation",
            step_type=ExecutionStepType.VALIDATION,
            description="Evaluate rendered composite against quality targets and brand preservation rules",
            inputs=["composite:contrast_adjusted", "brief:validation"],
            outputs=["report:quality_scores"],
            dependencies=["step_14_contrast_adjustment"],
            resources=ResourceEstimates(vram_mb=128.0, cpu_usage_pct=20.0, estimated_runtime_ms=120.0, estimated_cost=1.0),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="validation.validation_score",
            parameters={
                "min_validation_score": brief.validation.validation_score,
                "min_readiness_score": brief.validation.readiness_score,
            },
        )

        # 16. FINAL_COMPOSITE
        steps["step_16_final_composite"] = ExecutionStep(
            step_id="step_16_final_composite",
            step_type=ExecutionStepType.FINAL_COMPOSITE,
            description="Produce final composite image artifact and embed provenance metadata",
            inputs=["composite:contrast_adjusted", "report:quality_scores"],
            outputs=["artifact:final_image"],
            dependencies=["step_15_validation"],
            resources=ResourceEstimates(vram_mb=256.0, cpu_usage_pct=25.0, estimated_runtime_ms=150.0, estimated_cost=1.5),
            retry_policy=RetryPolicy(max_retries=2),
            sourced_from_brief_field="execution_constraints.must_preserve",
            parameters={"must_preserve": brief.execution_constraints.must_preserve},
        )

        # 17. CLEANUP
        steps["step_17_cleanup"] = ExecutionStep(
            step_id="step_17_cleanup",
            step_type=ExecutionStepType.CLEANUP,
            description="Flush temporary intermediate buffers and release GPU VRAM",
            inputs=["artifact:final_image"],
            outputs=["state:cleaned_buffers"],
            dependencies=["step_16_final_composite"],
            resources=ResourceEstimates(vram_mb=0.0, cpu_usage_pct=5.0, estimated_runtime_ms=20.0, estimated_cost=0.1),
            retry_policy=RetryPolicy(max_retries=1),
            sourced_from_brief_field="metadata.brief_id",
            parameters={"brief_id": brief.metadata.brief_id},
        )

        # Compute topological sort & parallel stages
        temp_graph = ExecutionGraph(
            graph_id=f"graph_{uuid.uuid4().hex[:8]}",
            steps=steps,
        )
        topo_order = temp_graph.compute_topological_sort()
        parallel_stages = temp_graph.compute_parallel_stages()

        # Update step execution_stage indices via model_copy
        updated_steps: Dict[str, ExecutionStep] = {}
        for stage_idx, stage_nodes in enumerate(parallel_stages):
            for node_id in stage_nodes:
                updated_steps[node_id] = steps[node_id].model_copy(update={"execution_stage": stage_idx})

        # Calculate peak VRAM and total runtime/cost
        peak_vram_mb = 0.0
        for stage_nodes in parallel_stages:
            stage_vram = sum(updated_steps[n].resources.vram_mb for n in stage_nodes)
            peak_vram_mb = max(peak_vram_mb, stage_vram)

        total_runtime_ms = sum(updated_steps[n].resources.estimated_runtime_ms for n in topo_order)
        total_cost = sum(updated_steps[n].resources.estimated_cost for n in topo_order)

        graph = ExecutionGraph(
            graph_id=temp_graph.graph_id,
            steps=updated_steps,
            topological_order=topo_order,
            parallel_stages=parallel_stages,
            total_vram_peak_mb=peak_vram_mb,
            total_estimated_runtime_ms=total_runtime_ms,
            total_estimated_cost=total_cost,
        )

        metadata = ExecutionMetadata(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            brief_ref=brief.metadata.brief_id,
            schema_version="1.0.0",
            created_at=_utc_now_iso(),
            planner_id="execution_planner_v1",
            total_steps=len(steps),
            total_stages=len(parallel_stages),
        )

        quality_targets = {
            "min_validation_score": brief.validation.validation_score,
            "min_readiness_score": brief.validation.readiness_score,
            "confidence": brief.validation.confidence,
        }

        execution_constraints = {
            "must_preserve": brief.execution_constraints.must_preserve,
            "allowed_transformations": brief.execution_constraints.allowed_transformations,
            "forbidden_transformations": brief.execution_constraints.forbidden_transformations,
        }

        plan = ExecutionPlan(
            metadata=metadata,
            graph=graph,
            quality_targets=quality_targets,
            execution_constraints=execution_constraints,
        )

        logger.info(
            f"Successfully generated ExecutionPlan '{plan.metadata.plan_id}' for Brief '{brief.metadata.brief_id}' "
            f"({len(steps)} steps, {len(parallel_stages)} stages, peak_vram={peak_vram_mb:.1f}MB, runtime={total_runtime_ms:.1f}ms)."
        )
        return plan

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> ExecutionPlan:
        """
        BaseReasoner execution interface.
        Extracts DesignBrief from context or generates it on the fly, then plans execution.
        """
        brief = getattr(context, "design_brief", None)
        if brief is None:
            from thumbnail_intelligence.reasoning.design_brief_generator import DesignBriefGenerator

            brief_gen = DesignBriefGenerator()
            brief = brief_gen.reason(graph=graph, context=context)

        return self.plan(brief=brief)

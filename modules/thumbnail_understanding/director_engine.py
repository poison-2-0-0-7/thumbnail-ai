"""
director_engine.py
===================

AI Thumbnail Director & Professional Improvement Planner (Phase 15 & Phase 16).

Transforms structured visual understanding (scene graph, hierarchy, psychology,
weaknesses, layers) into high-level creative direction and a concrete, multi-action
Professional Improvement Plan with explicit expected CTR gains, risk levels, and fallbacks.
"""

from __future__ import annotations

from thumbnail_understanding.schemas import (
    ActionType,
    AIThumbnailDirectorPlan,
    CompositionIntelligence,
    DecomposedScene,
    ElementType,
    ImprovementAction,
    ProfessionalImprovementPlan,
    SceneGraph,
    ThumbnailPsychologyAssessment,
    VisualHierarchy,
    WeaknessAnalysis,
)


class AIThumbnailDirector:
    """Creative Director and Professional Improvement Planner engine."""

    @classmethod
    def generate_plan(
        cls,
        scene_graph: SceneGraph,
        hierarchy: VisualHierarchy,
        composition: CompositionIntelligence,
        psychology: ThumbnailPsychologyAssessment,
        weaknesses: WeaknessAnalysis,
        decomposed_scene: DecomposedScene,
    ) -> tuple[AIThumbnailDirectorPlan, ProfessionalImprovementPlan]:
        """
        Produce AIThumbnailDirectorPlan and ProfessionalImprovementPlan.
        """
        keep_targets: list[str] = []
        change_targets: list[str] = []
        remove_targets: list[str] = []
        emphasize_targets: list[str] = []

        actions: list[ImprovementAction] = []
        action_idx = 1

        # 1. Primary Subject Preservation Directive
        if scene_graph.hero_element_id:
            hero = next((e for e in scene_graph.elements if e.element_id == scene_graph.hero_element_id), None)
            if hero:
                keep_targets.append(hero.element_id)
                emphasize_targets.append(hero.element_id)

                actions.append(
                    ImprovementAction(
                        action_id=f"act_{action_idx:02d}_preserve_hero",
                        action=ActionType.KEEP,
                        target_element_id=hero.element_id,
                        reason=f"Preserve creator face identity '{hero.label}' as main emotional hook",
                        priority=1,
                        confidence=0.95,
                        expected_impact="High identity preservation and viewer connection",
                        expected_ctr_gain=0.08,
                        identity_risk="low",
                        visual_risk="low",
                        dependencies=[],
                        preserve_requirements=["face_pixels", "facial_expression"],
                        edit_method="mask_protection",
                        fallback_action="keep",
                    )
                )
                action_idx += 1

                # Enhance hero subject lighting/contrast if separation is low
                if hierarchy.subject_separation_score < 0.5:
                    actions.append(
                        ImprovementAction(
                            action_id=f"act_{action_idx:02d}_enhance_hero_pop",
                            action=ActionType.ENHANCE,
                            target_element_id=hero.element_id,
                            reason="Enhance primary face separation and contrast against background",
                            priority=2,
                            confidence=0.90,
                            expected_impact="Increased visual hierarchy and eye draw",
                            expected_ctr_gain=0.10,
                            identity_risk="low",
                            visual_risk="low",
                            dependencies=[f"act_01_preserve_hero"],
                            preserve_requirements=["face_pixels"],
                            edit_method="relight_and_contrast_boost",
                            fallback_action="keep",
                        )
                    )
                    action_idx += 1

        # 2. Background Replacement Directive (if clutter or weakness detected)
        bg_layer = next((l for l in decomposed_scene.layers if l.category.value == "background"), None)
        bg_id = bg_layer.layer_id if bg_layer else "layer_00_background"

        if composition.clutter_score > 0.3 or any(w.weakness_type == "clutter" for w in weaknesses.findings):
            change_targets.append(bg_id)
            actions.append(
                ImprovementAction(
                    action_id=f"act_{action_idx:02d}_replace_background",
                    action=ActionType.REPLACE,
                    target_element_id=bg_id,
                    reason="Replace noisy room background with high-contrast, vibrant, de-cluttered backdrop",
                    priority=2,
                    confidence=0.88,
                    expected_impact="Eliminates visual noise and amplifies hero subject prominence",
                    expected_ctr_gain=0.15,
                    identity_risk="low",
                    visual_risk="medium",
                    dependencies=[f"act_01_preserve_hero"],
                    preserve_requirements=["subject_silhouette"],
                    edit_method="regional_inpainting_or_controlnet_bg",
                    fallback_action="enhance",
                )
            )
            action_idx += 1

        # 3. Removal of low-value distracting objects
        distractions = [e for e in scene_graph.elements if e.role.value == "distraction" or e.replace_score > 0.7]
        for dist in distractions:
            remove_targets.append(dist.element_id)
            actions.append(
                ImprovementAction(
                    action_id=f"act_{action_idx:02d}_remove_{dist.category}",
                    action=ActionType.REMOVE,
                    target_element_id=dist.element_id,
                    reason=f"Remove distracting low-value object '{dist.label}'",
                    priority=3,
                    confidence=0.85,
                    expected_impact="Cleans up composition and focuses attention",
                    expected_ctr_gain=0.05,
                    identity_risk="low",
                    visual_risk="low",
                    dependencies=[],
                    preserve_requirements=[],
                    edit_method="object_removal_inpaint",
                    fallback_action="keep",
                )
            )
            action_idx += 1

        # 4. Text / Typography Enhancement
        text_elems = [e for e in scene_graph.elements if e.element_type == ElementType.TEXT]
        for txt in text_elems:
            keep_targets.append(txt.element_id)
            if composition.text_overlaps_subject:
                actions.append(
                    ImprovementAction(
                        action_id=f"act_{action_idx:02d}_reposition_text",
                        action=ActionType.REPOSITION,
                        target_element_id=txt.element_id,
                        reason=f"Reposition text '{txt.label}' to clean negative space away from hero face",
                        priority=1,
                        confidence=0.92,
                        expected_impact="Resolves text/subject occlusion and improves legibility",
                        expected_ctr_gain=0.12,
                        identity_risk="low",
                        visual_risk="low",
                        dependencies=[],
                        preserve_requirements=["text_content"],
                        edit_method="layout_reposition",
                        fallback_action="keep",
                    )
                )
                action_idx += 1

        total_ctr = sum(a.expected_ctr_gain for a in actions)

        director_plan = AIThumbnailDirectorPlan(
            creative_direction=(
                f"Redesign thumbnail for video by preserving creator identity ({len(keep_targets)} elements), "
                f"replacing background clutter ({len(change_targets)} elements), and focusing visual hierarchy."
            ),
            story_analysis=psychology.overall_psychology_summary,
            elements_to_keep=keep_targets,
            elements_to_change=change_targets,
            elements_to_remove=remove_targets,
            elements_to_emphasize=emphasize_targets,
            composition_strategy=f"Dominant subject '{scene_graph.hero_element_id}' with clean high-contrast backdrop",
            redesign_aggressiveness="moderate",
            expected_ctr_improvement=f"+{total_ctr * 100:.1f}% estimated CTR potential gain",
        )

        improvement_plan = ProfessionalImprovementPlan(
            actions=actions,
            primary_goal="Maximize thumbnail CTR through hero preservation and visual hierarchy optimization",
        )

        return director_plan, improvement_plan

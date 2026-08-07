"""
design_brief_generator.py
==========================

DesignBrief Generator Implementation (Phase 3.5).
Converts a fully validated reasoning package (ValidatedReasoningPackage) into a strongly typed,
renderer-independent DesignBrief.

This engine performs pure deterministic TRANSLATION.
It performs NO speculative reasoning.
It performs NO optimization.
It performs NO image generation.
It contains NO renderer-specific logic (no SD prompts, no ComfyUI nodes, no SDXL parameters).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.knowledge_base.models import EvidenceReference, _utc_now_iso
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.design_brief_models import (
    AudienceBrief,
    BrandBrief,
    BriefMetadata,
    CameraBrief,
    ColorBrief,
    CompositionBrief,
    CreatorBrief,
    DesignBrief,
    ExecutionConstraintsBrief,
    LightingBrief,
    NarrativeBrief,
    ObjectsBrief,
    TypographyBrief,
    ValidationBrief,
)
from thumbnail_intelligence.reasoning.exceptions import ReasonerValidationError
from thumbnail_intelligence.reasoning.interfaces import BaseReasoner, DesignBriefGeneratorInterface
from thumbnail_intelligence.reasoning.models import (
    ReasonerContract,
    ReasonerType,
)
from thumbnail_intelligence.reasoning.strategy_models import StrategyDecision
from thumbnail_intelligence.reasoning.validator import StrategicReasoningValidator
from thumbnail_intelligence.reasoning.validator_models import ValidatedReasoningPackage

logger = logging.getLogger(__name__)


class DesignBriefGenerator(DesignBriefGeneratorInterface):
    """
    Production DesignBriefGenerator converting ValidatedReasoningPackage into a DesignBrief.
    Enforces deterministic translation, strict schema validation, evidence propagation,
    and renderer independence.
    """

    def __init__(self, validator: Optional[StrategicReasoningValidator] = None) -> None:
        self._contract = ReasonerContract(
            name="design_brief_generator",
            reasoner_type=ReasonerType.DESIGN_BRIEF_GENERATOR,
            description="Translates a validated strategic reasoning package into a deterministic DesignBrief",
            dependencies=[
                "narrative_reasoner",
                "audience_reasoner",
                "creator_reasoner",
                "brand_reasoner",
                "priority_reasoner",
                "risk_reasoner",
                "strategy_ranker",
                "validator",
            ],
            timeout_seconds=5.0,
            max_retries=0,
            supports_cache=True,
        )
        self._validator = validator or StrategicReasoningValidator()

    @property
    def contract(self) -> ReasonerContract:
        """Return metadata contract for registration and coordinator topology."""
        return self._contract

    def generate(
        self,
        package: ValidatedReasoningPackage,
        strict_validation: bool = False,
    ) -> DesignBrief:
        """
        Primary entrypoint: Translate a ValidatedReasoningPackage into a DesignBrief.

        Args:
            package: Fully validated reasoning package from StrategicReasoningValidator.
            strict_validation: If True, raises ReasoningValidationError when package is not ready for brief generation.

        Returns:
            Strongly typed, renderer-independent DesignBrief.
        """
        if strict_validation and not package.ready_for_design_brief:
            msg = (
                f"Reasoning package '{package.package_id}' failed validation readiness check "
                f"(readiness={package.validation.readiness_score:.2f}, blocking_errors={len(package.validation.blocking_errors)})."
            )
            raise ReasonerValidationError(
                reasoner_name="design_brief_generator",
                validation_errors=[msg],
            )

        context = package.context
        strategy_decision = package.strategy_decision or getattr(context, "strategies", None)
        validation_report = package.validation

        # 1. Metadata
        metadata = BriefMetadata(
            brief_id=f"brief_{uuid.uuid4().hex[:8]}",
            schema_version="1.0.0",
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
            generator_id="design_brief_generator_v1",
        )

        # 2. Narrative
        narrative_res = getattr(context, "narrative", None)
        prim_nar = getattr(narrative_res, "primary_narrative", None)
        nar_arc = getattr(narrative_res, "narrative_arc", None)
        v_candidates = getattr(narrative_res, "visual_focus_candidates", [])

        primary_story = (
            getattr(prim_nar, "premise", "")
            or getattr(narrative_res, "story_hook", "")
            or getattr(prim_nar, "title", "Grounded visual storytelling premise")
        )
        supporting_story = (
            getattr(narrative_res, "narrative_angle", "")
            or getattr(prim_nar, "hook", "")
        )
        emotional_goal = (
            getattr(narrative_res, "emotional_tone", "")
            or getattr(prim_nar, "emotional_tone", "")
            or getattr(nar_arc, "primary_driver", "")
            or "curiosity"
        )
        story_focus = (
            v_candidates[0].element_name if v_candidates and hasattr(v_candidates[0], "element_name") else "Hero Subject"
        )
        raw_nar_type = getattr(narrative_res, "narrative_type", "discovery")
        nar_type = raw_nar_type.value if hasattr(raw_nar_type, "value") else str(raw_nar_type)

        nar_arc_name = getattr(nar_arc, "arc_name", "Mystery Arc") if nar_arc else "Mystery Arc"

        narrative_brief = NarrativeBrief(
            primary_story=primary_story,
            supporting_story=supporting_story,
            emotional_goal=emotional_goal,
            story_focus=story_focus,
            narrative_type=str(nar_type),
            narrative_arc=nar_arc_name,
        )

        # 3. Audience
        aud_res = getattr(context, "audience", None)
        prim_aud = getattr(aud_res, "primary_audience", None)
        sec_auds = getattr(aud_res, "secondary_audiences", [])

        primary_audience_str = (
            getattr(prim_aud, "audience_segment", "General Audience")
            if prim_aud
            else "General Audience"
        )
        secondary_aud_list = [
            a.audience_segment for a in sec_auds if hasattr(a, "audience_segment")
        ]

        c_trigs = getattr(prim_aud, "curiosity_triggers", []) if prim_aud else []
        curiosity_trig = ", ".join(c_trigs) if c_trigs else getattr(aud_res, "viewer_motivation", "intrigue")

        v_intent = getattr(aud_res, "viewer_intent", None)
        if v_intent is None and prim_aud:
            v_intent = getattr(prim_aud, "intent", "entertainment")
        v_intent_str = v_intent.value if hasattr(v_intent, "value") else str(v_intent or "entertainment")

        cog_load = getattr(aud_res, "optimal_cognitive_load", None)
        if cog_load is None and prim_aud:
            cog_load = getattr(prim_aud, "cognitive_load", "medium")
        cog_load_str = cog_load.value if hasattr(cog_load, "value") else str(cog_load or "medium")

        audience_brief = AudienceBrief(
            primary_audience=primary_audience_str,
            secondary_audience=secondary_aud_list,
            curiosity_trigger=curiosity_trig,
            viewer_intent=str(v_intent_str),
            cognitive_load=str(cog_load_str),
        )

        # 4. Creator
        creator_res = getattr(context, "creator_intent", None)
        prim_style = (
            getattr(creator_res, "primary_creator_style", None)
            or getattr(creator_res, "primary_style", None)
        )
        creator_id = (
            getattr(creator_res, "creator_identity", "")
            or getattr(prim_style, "persona_name", "Creator Identity")
        )
        sig_elems = (
            (getattr(prim_style, "signature_elements", []) if prim_style else [])
            + getattr(creator_res, "visual_constraints", [])
        )
        hist_cons = getattr(creator_res, "brand_consistency", 1.0)
        c_archetype = getattr(prim_style, "creator_archetype", None) or getattr(creator_res, "creator_archetype", "educator")
        c_archetype_str = c_archetype.value if hasattr(c_archetype, "value") else str(c_archetype or "educator")
        c_voice = getattr(prim_style, "channel_voice", "") or getattr(creator_res, "creator_style", "")

        creator_brief = CreatorBrief(
            creator_identity=creator_id,
            style_constraints=list(set(sig_elems)),
            historical_consistency=hist_cons,
            creator_archetype=c_archetype_str,
            channel_voice=c_voice,
        )

        # 5. Brand
        brand_res = getattr(context, "brand_constraints", None)
        b_directives = (
            getattr(brand_res, "required_preservations", None)
            or getattr(brand_res, "preservation_directives", [])
        )
        req_elems = [p.element_name for p in b_directives if hasattr(p, "element_name")]
        forb_changes = (
            getattr(brand_res, "brand_constraints", [])
            + getattr(brand_res, "prohibited_elements", [])
        )
        pres_rules = [
            f"{p.element_name}: {p.required_treatment}"
            for p in b_directives
            if hasattr(p, "element_name") and hasattr(p, "required_treatment")
        ]
        prim_brand_interp = getattr(brand_res, "primary_brand_interpretation", None) or getattr(brand_res, "primary_interpretation", None)
        b_pillars = getattr(prim_brand_interp, "brand_pillars", []) if prim_brand_interp else []

        brand_brief = BrandBrief(
            required_elements=req_elems,
            forbidden_changes=list(set(forb_changes)),
            brand_preservation_rules=pres_rules,
            brand_pillars=b_pillars,
        )

        # 6. Composition
        prio_res = getattr(context, "visual_priorities", None)
        prim_hier = getattr(prio_res, "primary_hierarchy_candidate", None) or getattr(prio_res, "primary_hierarchy", None)
        v_hierarchy = getattr(prio_res, "visual_hierarchy", None) or getattr(prio_res, "hierarchy_nodes", [])

        primary_subj = (
            getattr(prim_hier, "primary_focus", "")
            if prim_hier
            else (v_hierarchy[0].element_name if v_hierarchy and hasattr(v_hierarchy[0], "element_name") else "Host Face")
        )
        secondary_subj = (
            getattr(prim_hier, "secondary_focus", "")
            if prim_hier
            else (v_hierarchy[1].element_name if len(v_hierarchy) > 1 and hasattr(v_hierarchy[1], "element_name") else "Supporting Object")
        )

        hierarchy_dicts = [
            n.model_dump(mode="json") if hasattr(n, "model_dump") else dict(n)
            for n in v_hierarchy
        ]

        composition_brief = CompositionBrief(
            primary_subject=primary_subj,
            secondary_subject=secondary_subj,
            visual_hierarchy=hierarchy_dicts,
            negative_space="balanced_uncluttered",
            safe_zones=["timestamp_bottom_right", "title_top_left"],
            depth_treatment="shallow_subject_forward",
        )

        # 7. Typography
        win_strat = getattr(strategy_decision, "winning_strategy", None) if strategy_decision else None
        text_priority = "high"
        text_regions = ["upper_third", "top_left_quadrant"]
        max_chars = 25
        readability_targets = "high_contrast_mobile_first"
        max_words = 4

        typography_brief = TypographyBrief(
            text_priority=text_priority,
            text_regions=text_regions,
            maximum_characters=max_chars,
            readability_targets=readability_targets,
            max_word_count=max_words,
        )

        # 8. Color
        vis_ident = getattr(creator_res, "visual_identity", None)
        prim_palette = getattr(vis_ident, "dominant_color_palette", ["#0066CC", "#FFFFFF"]) if vis_ident else ["#0066CC", "#FFFFFF"]
        accent_palette = ["#FFCC00", "#FF3300"]
        contrast_targets = "4.5:1 minimum luminance separation"
        brand_colors = getattr(vis_ident, "dominant_color_palette", []) if vis_ident else []

        color_brief = ColorBrief(
            primary_palette=prim_palette,
            accent_palette=accent_palette,
            contrast_targets=contrast_targets,
            brand_colors=brand_colors,
        )

        # 9. Lighting
        lighting_brief = LightingBrief(
            mood="high_key_dramatic",
            direction="top_left_key_soft_rim",
            intensity="punchy",
        )

        # 10. Camera
        camera_brief = CameraBrief(
            crop="medium_close_up",
            perspective="eye_level",
            zoom="subject_focused",
            subject_scale="large_mobile_first",
        )

        # 11. Objects
        req_objects = [
            n.element_name
            for n in v_hierarchy
            if hasattr(n, "element_name") and hasattr(n, "tier") and str(n.tier).lower() != "suppressed"
        ]
        opt_objects = [
            vf.element_name
            for vf in v_candidates
            if hasattr(vf, "element_name") and str(getattr(vf, "visual_priority", "")).upper() != "PRIMARY"
        ]
        forb_objects = list(forb_changes)

        objects_brief = ObjectsBrief(
            required_objects=list(set(req_objects)),
            optional_objects=list(set(opt_objects)),
            forbidden_objects=list(set(forb_objects)),
        )

        # 12. Execution Constraints
        must_pres = list(req_elems)
        allowed_trans = [
            "expression_enhancement",
            "warm_rim_lighting",
            "background_blur",
        ]
        forb_trans = [
            "identity_distortion",
            "pose_flip",
            "brand_color_mutation",
        ]

        exec_constraints_brief = ExecutionConstraintsBrief(
            must_preserve=must_pres,
            allowed_transformations=allowed_trans,
            forbidden_transformations=forb_trans,
        )

        # 13. Validation
        strat_id = getattr(win_strat, "candidate_id", "strat_default") if win_strat else "strat_default"
        ev_refs = getattr(validation_report, "evidence_references", []) or getattr(context, "evidence_references", [])

        validation_brief = ValidationBrief(
            strategy_id=strat_id,
            evidence_references=ev_refs,
            confidence=validation_report.confidence,
            validation_score=validation_report.consistency_score,
            readiness_score=validation_report.readiness_score,
            ready_for_design_brief=package.ready_for_design_brief,
            detected_conflicts_count=len(validation_report.detected_conflicts),
            blocking_errors_count=len(validation_report.blocking_errors),
        )

        design_brief = DesignBrief(
            metadata=metadata,
            narrative=narrative_brief,
            audience=audience_brief,
            creator=creator_brief,
            brand=brand_brief,
            composition=composition_brief,
            typography=typography_brief,
            color=color_brief,
            lighting=lighting_brief,
            camera=camera_brief,
            objects=objects_brief,
            execution_constraints=exec_constraints_brief,
            validation=validation_brief,
        )

        logger.info(
            f"Successfully generated DesignBrief '{design_brief.metadata.brief_id}' "
            f"(readiness={design_brief.validation.readiness_score:.2f}, confidence={design_brief.validation.confidence:.2f})."
        )
        return design_brief

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> DesignBrief:
        """
        BaseReasoner execution interface.
        Extracts or generates ValidatedReasoningPackage from context and emits DesignBrief.
        """
        package = getattr(context, "validated_package", None)
        if package is None:
            # Run validator dynamically if context is not yet packaged
            package = self._validator.validate(context=context, graph=graph)

        return self.generate(package=package, strict_validation=False)

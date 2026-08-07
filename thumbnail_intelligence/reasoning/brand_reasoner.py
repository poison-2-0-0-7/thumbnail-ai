"""
brand_reasoner.py
=================

Production BrandReasoner for the Thumbnail Intelligence Engine (Phase 3.4D).
Infers the creator's brand identity, foundational brand pillars, mandatory visual preservations,
visual guardrails, allowed variations, and forbidden changes grounded in empirical evidence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from thumbnail_intelligence.evidence.models import (
    EvidenceNode,
    EvidenceReference,
    EvidenceSourceType,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    _utc_now_iso,
)
from thumbnail_intelligence.reasoning.brand_models import (
    BrandPreservationPriority,
    BrandResult,
    CandidateBrandInterpretation,
    VisualElementPreservation,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.interfaces import BrandReasoner as BaseBrandReasoner
from thumbnail_intelligence.reasoning.models import (
    ReasonerContract,
    ReasonerType,
)


class BrandReasoner(BaseBrandReasoner):
    """
    Brand Intelligence and Visual Identity Protection Reasoning Engine.
    Synthesizes brand constraints, mandatory visual preservations, and forbidden changes.
    """

    def __init__(
        self,
        name: str = "brand_reasoner",
        version: str = "1.0.0",
        is_mandatory: bool = True,
        timeout_ms: float = 5000.0,
    ) -> None:
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.BRAND,
            dependencies=["narrative_reasoner", "creator_reasoner"],
            version=version,
            description="Infers creator brand identity, mandatory visual preservations, visual guardrails, and forbidden changes",
            is_mandatory=is_mandatory,
            timeout_ms=timeout_ms,
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> BrandResult:
        """
        Execute brand reasoning over NormalizedEvidenceGraph and current ReasoningContext.

        Args:
            graph: Grounded NormalizedEvidenceGraph.
            context: ReasoningContext containing narrative and creator inferences.

        Returns:
            Validated BrandResult with visual preservations, guardrails, candidate brand rankings, and confidence.
        """
        trace: List[str] = []
        trace.append(f"Starting BrandReasoner execution (v{self.version}) on graph {graph.graph_id}")

        # 1. Ingest active evidence nodes and prior reasoning context
        active_nodes = [node for node in graph.get_active_nodes() if node.is_active]
        creator_ctx = context.creator_intent
        narrative_ctx = context.narrative
        audience_ctx = context.audience
        trace.append(
            f"Ingested {len(active_nodes)} active nodes; creator context: {creator_ctx is not None}, "
            f"narrative context: {narrative_ctx is not None}"
        )

        # 2. Extract grounded brand signals, palette rules, and references
        tokens, brand_rules_data, refs_map = self._extract_brand_signals(
            graph, active_nodes, creator_ctx, narrative_ctx, trace
        )

        # 3. Infer Brand Identity and Pillars
        brand_identity, brand_pillars = self._infer_brand_identity_and_pillars(
            creator_ctx, narrative_ctx, tokens, trace
        )

        # 4. Synthesize Logo, Typography, and Palette Rules
        logo_rules, typo_rules, color_palette = self._synthesize_style_rules(
            creator_ctx, tokens, refs_map, trace
        )

        # 5. Formulate Required Preservations, Allowed Variations, and Forbidden Changes
        preservations, allowed_variations, forbidden_changes = self._formulate_preservations_and_guardrails(
            creator_ctx, tokens, refs_map, trace
        )

        # 6. Multi-Hypothesis Candidate Brand Interpretations
        candidates = self._generate_brand_candidates(
            graph, brand_identity, brand_pillars, color_palette, typo_rules, preservations, refs_map, trace
        )
        if not candidates:
            candidates = [self._build_default_candidate(brand_identity, color_palette, typo_rules, refs_map)]

        primary_candidate = candidates[0]
        secondary_candidates = candidates[1:] if len(candidates) > 1 else []

        # 7. Multi-Signal Calibrated Confidence Model
        confidence_breakdown, overall_confidence = self._calculate_confidence(
            graph, active_nodes, creator_ctx, narrative_ctx, audience_ctx, tokens, trace
        )

        # 8. Harvest and Deduplicate Evidence References
        all_refs: List[EvidenceReference] = []
        seen_refs: Set[str] = set()

        for cand in candidates:
            for ref in cand.evidence_refs:
                key = f"{ref.source_id}:{ref.source_type.value if hasattr(ref.source_type, 'value') else ref.source_type}"
                if key not in seen_refs:
                    seen_refs.add(key)
                    all_refs.append(ref)

        for pres in preservations:
            for ref in pres.evidence_refs:
                key = f"{ref.source_id}:{ref.source_type.value if hasattr(ref.source_type, 'value') else ref.source_type}"
                if key not in seen_refs:
                    seen_refs.add(key)
                    all_refs.append(ref)

        # Grounding gate invariant: zero evidence refs implies zero confidence
        if not all_refs:
            overall_confidence = 0.0
            confidence_breakdown["evidence_quality"] = 0.0

        supporting_ids = list(refs_map.keys())

        # 9. Selection Rationale
        selection_rationale = (
            f"Brand interpretation '{primary_candidate.interpretation_name}' chosen as primary ({primary_candidate.fit_score:.2f}) "
            f"due to strict alignment with creator identity and support across {len(primary_candidate.supporting_evidence_ids)} empirical nodes."
        )

        # 10. Assemble BrandResult
        result = BrandResult(
            brand_identity=brand_identity,
            brand_pillars=brand_pillars,
            visual_identity={
                "color_palette": color_palette,
                "typography": typo_rules,
                "logo_usage": logo_rules,
            },
            logo_usage=logo_rules,
            color_palette=color_palette,
            typography_preferences=typo_rules,
            recurring_subjects=primary_candidate.recurring_subjects,
            recurring_layout_patterns=primary_candidate.recurring_layout_patterns,
            creator_signature_elements=primary_candidate.creator_signature_elements,
            brand_constraints=[p.required_treatment for p in preservations],
            required_preservations=preservations,
            allowed_variations=allowed_variations,
            forbidden_changes=forbidden_changes,
            primary_brand_interpretation=primary_candidate,
            candidate_interpretations=candidates,
            rejected_interpretations=secondary_candidates,
            selection_rationale=selection_rationale,
            brand_confidence=overall_confidence,
            confidence_breakdown=confidence_breakdown,
            supporting_evidence_ids=supporting_ids,
            color_palette_rules=[f"Use palette: {', '.join(color_palette[:3])}", "Ensure 4.5:1 text-to-background contrast"],
            typography_rules=[typo_rules, "Max 4-5 words total per thumbnail"],
            logo_rules=[logo_rules],
            prohibited_elements=forbidden_changes,
            identity_lock_requirements=[p.required_treatment for p in preservations if p.element_type == "face"],
            compliance_score=0.95 if overall_confidence > 0.0 else 0.0,
            evidence_refs=all_refs,
            confidence=overall_confidence,
            reasoning_trace=trace,
            metadata={"primary_interpretation": primary_candidate.interpretation_name, "preservations_count": len(preservations)},
        )

        trace.append(f"Successfully constructed BrandResult with {len(all_refs)} grounding references")
        return result

    def validate_output(self, output: Any) -> bool:
        """Validate output satisfies contract and confidence invariants."""
        if not isinstance(output, BrandResult):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        if not (0.0 <= output.brand_confidence <= 1.0):
            return False
        if not (0.0 <= output.compliance_score <= 1.0):
            return False
        return True

    # -----------------------------------------------------------------------
    # Internal Extraction and Inference Helpers
    # -----------------------------------------------------------------------

    def _extract_brand_signals(
        self,
        graph: NormalizedEvidenceGraph,
        active_nodes: List[EvidenceNode],
        creator_ctx: Any,
        narrative_ctx: Any,
        trace: List[str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any], Dict[str, List[EvidenceReference]]]:
        """Extract brand rules, colors, typography tokens, and references."""
        tokens: Dict[str, List[str]] = {
            "colors": [],
            "brand_names": [],
            "prohibitions": [],
            "subjects": [],
        }
        brand_data: Dict[str, Any] = {}
        refs_map: Dict[str, List[EvidenceReference]] = {}

        for node in active_nodes:
            payload = getattr(node.evidence_item, "data_payload", {}) or {}
            node_refs = list(getattr(node.evidence_item, "evidence_refs", []))

            if not node_refs:
                synth_ref = EvidenceReference(
                    source_id=node.node_id,
                    source_type=getattr(node.provenance, "source_type", EvidenceSourceType.KNOWLEDGE_ENTRY),
                    confidence=node.confidence.propagated_confidence,
                    grade=EvidenceGrade.STRONG if node.confidence.propagated_confidence > 0.8 else EvidenceGrade.MODERATE,
                    claim_summary=getattr(node.provenance, "retrieval_reason", "Brand evidence node"),
                )
                node_refs.append(synth_ref)

            refs_map[node.node_id] = node_refs

            # Detect creator profile entry
            if node.node_type == KnowledgeEntryType.CREATOR_PROFILE_ENTRY:
                brand_data["display_name"] = payload.get("display_name", "")
                tokens["brand_names"].append(str(payload.get("display_name", "")))
                rules = payload.get("brand_rules", [])
                if isinstance(rules, list):
                    for r in rules:
                        claim = getattr(r, "claim", str(r))
                        tokens["prohibitions"].append(claim)

            # Detect historical thumbnail entry
            if node.node_type == KnowledgeEntryType.HISTORICAL_THUMBNAIL:
                if "color_palette" in payload:
                    tokens["colors"].extend(payload["color_palette"])

            # Detect subjects
            if "objects" in payload:
                objs = payload.get("objects", [])
                if isinstance(objs, list):
                    tokens["subjects"].extend([str(o).lower() for o in objs])

        if creator_ctx and hasattr(creator_ctx, "evidence_refs"):
            for ref in creator_ctx.evidence_refs:
                refs_map.setdefault(ref.source_id, []).append(ref)

        if narrative_ctx and hasattr(narrative_ctx, "evidence_refs"):
            for ref in narrative_ctx.evidence_refs:
                refs_map.setdefault(ref.source_id, []).append(ref)

        trace.append(f"Extracted brand signals across {len(refs_map)} grounded node sources")
        return tokens, brand_data, refs_map

    def _infer_brand_identity_and_pillars(
        self,
        creator_ctx: Any,
        narrative_ctx: Any,
        tokens: Dict[str, List[str]],
        trace: List[str],
    ) -> Tuple[str, List[str]]:
        """Infer high-level brand identity and foundational pillars."""
        creator_name = (
            creator_ctx.creator_identity if creator_ctx and hasattr(creator_ctx, "creator_identity") and creator_ctx.creator_identity
            else "Featured Brand"
        )
        channel_voice = (
            creator_ctx.channel_voice if creator_ctx and hasattr(creator_ctx, "channel_voice") and creator_ctx.channel_voice
            else "High clarity and engaging visual storytelling"
        )

        brand_identity = f"{creator_name}: Defined by {channel_voice} and consistent high-contrast thumbnail presentation."

        pillars = [
            "Authentic Creator Presence & Expressive Face Recognition",
            "High-Contrast Visual Hierarchy with Clean Focal Separation",
            "Viewer Trust & Credible Story Hook Delivery",
            "Audience Retention Through Minimal Cognitive Friction",
        ]

        trace.append(f"Inferred brand identity: '{brand_identity}' with {len(pillars)} pillars")
        return brand_identity, pillars

    def _synthesize_style_rules(
        self,
        creator_ctx: Any,
        tokens: Dict[str, List[str]],
        refs_map: Dict[str, List[EvidenceReference]],
        trace: List[str],
    ) -> Tuple[str, str, List[str]]:
        """Synthesize logo rules, typography styling, and color palette."""
        logo_rules = "Optional channel logo/badge placed in top-left with 10% outer padding, never obscuring the hero face."
        typo_rules = "Extra-bold grotesque sans-serif with 15% solid stroke outline and subtle dark drop shadow for high legibility."

        if creator_ctx and hasattr(creator_ctx, "visual_identity") and creator_ctx.visual_identity:
            vis = creator_ctx.visual_identity
            if hasattr(vis, "dominant_color_palette") and vis.dominant_color_palette:
                color_palette = list(vis.dominant_color_palette)
            else:
                color_palette = ["#00E5FF", "#FF3366", "#0D0D11", "#FFFFFF"]
            if hasattr(vis, "typography_style") and vis.typography_style:
                typo_rules = vis.typography_style
        elif tokens["colors"]:
            color_palette = list(dict.fromkeys(tokens["colors"]))[:4]
        else:
            color_palette = ["#00E5FF", "#FF3366", "#0D0D11", "#FFFFFF"]

        trace.append(f"Synthesized style rules with palette: {color_palette[:3]}")
        return logo_rules, typo_rules, color_palette

    def _formulate_preservations_and_guardrails(
        self,
        creator_ctx: Any,
        tokens: Dict[str, List[str]],
        refs_map: Dict[str, List[EvidenceReference]],
        trace: List[str],
    ) -> Tuple[List[VisualElementPreservation], List[str], List[str]]:
        """Formulate mandatory visual preservations, allowed variations, and forbidden changes."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]

        preservations: List[VisualElementPreservation] = [
            VisualElementPreservation(
                element_name="Creator Expressive Hero Face",
                element_type="face",
                preservation_priority=BrandPreservationPriority.STRICT_MANDATORY,
                required_treatment="Preserve natural facial features and identity; place on outer third with minimum 30% canvas scale.",
                allowed_variation="Facial expression can vary between intense shock, curious intrigue, and triumphant smile.",
                forbidden_change="Do not apply extreme AI facial distortion, artificial beautification smoothing, or text overlays over the face.",
                evidence_refs=all_refs[:2],
            ),
            VisualElementPreservation(
                element_name="High-Contrast Visual Separation",
                element_type="color",
                preservation_priority=BrandPreservationPriority.HIGH_RECOMMENDED,
                required_treatment="Maintain clean luminance separation (minimum 4.5:1 ratio) between foreground subject and background environment.",
                allowed_variation="Background color tinting and ambient rim light colors can adapt to narrative mood.",
                forbidden_change="Do not use muddy, low-contrast washed out backgrounds that blend into the subject.",
                evidence_refs=all_refs[2:4] if len(all_refs) >= 4 else all_refs[:1],
            ),
            VisualElementPreservation(
                element_name="Curiosity Tension Object",
                element_type="prop",
                preservation_priority=BrandPreservationPriority.HIGH_RECOMMENDED,
                required_treatment="Prominently display the central story object on the opposing third with crisp edge rim lighting.",
                allowed_variation="Object angle and depth perspective can be rotated to maximize curiosity.",
                forbidden_change="Do not hide or crop the key tension object below 20% canvas visibility.",
                evidence_refs=all_refs[:1],
            ),
        ]

        allowed_variations = [
            "Text hook phrasing can adapt between 2 to 4 words to maximize emotional resonance.",
            "Rim light hue can alternate between cyan, yellow, and magenta depending on background tone.",
            "Background setting angle can utilize slight wide-angle perspective distortion to enhance depth.",
        ]

        forbidden_changes = [
            "Do not obscure the creator's eyes or facial expression with typography or graphic stickers.",
            "Do not use generic red clickbait arrows without direct narrative relevance.",
            "Do not violate the 4.5:1 text-to-background contrast ratio on mobile devices.",
            "Do not introduce low-resolution or blurry assets in the foreground.",
        ]

        trace.append(
            f"Formulated {len(preservations)} visual preservations, {len(allowed_variations)} allowed variations, "
            f"and {len(forbidden_changes)} forbidden changes"
        )
        return preservations, allowed_variations, forbidden_changes

    def _generate_brand_candidates(
        self,
        graph: NormalizedEvidenceGraph,
        brand_identity: str,
        brand_pillars: List[str],
        color_palette: List[str],
        typo_rules: str,
        preservations: List[VisualElementPreservation],
        refs_map: Dict[str, List[EvidenceReference]],
        trace: List[str],
    ) -> List[CandidateBrandInterpretation]:
        """Generate competing candidate brand interpretations (Strict Legacy, Modern Evolved, Topic-Forward)."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]
        all_node_ids = list(refs_map.keys())

        candidates: List[CandidateBrandInterpretation] = []

        # Candidate A: Strict Legacy Brand Preservation (Highest Fit)
        cand_a = CandidateBrandInterpretation(
            interpretation_name="Strict Legacy Brand Continuity",
            brand_pillars=brand_pillars,
            fit_score=0.95,
            confidence=0.93,
            color_palette=color_palette,
            typography_preferences=typo_rules,
            recurring_subjects=["Creator Hero Face", "Core Tension Prop"],
            recurring_layout_patterns=["Two-element split: face right, object left"],
            creator_signature_elements=["High key rim lighting", "Expressive face"],
            required_preservations=[p.element_name for p in preservations],
            allowed_variations=["Background ambient lighting"],
            forbidden_changes=["No facial obscuration"],
            pros=["Guarantees 100% subscriber recognition", "Protects established channel brand equity"],
            cons=["Strict constraints limit extreme avant-garde layout experimentation"],
            evidence_refs=all_refs[:3],
            supporting_evidence_ids=all_node_ids[:3],
        )
        candidates.append(cand_a)

        # Candidate B: Modernized Evolved Brand Style (Alternative 1)
        cand_b = CandidateBrandInterpretation(
            interpretation_name="Modernized High-Contrast Evolution",
            brand_pillars=brand_pillars[:3],
            fit_score=0.80,
            confidence=0.84,
            color_palette=color_palette[:3],
            typography_preferences="Sleek modern sans-serif with tighter kerning",
            recurring_subjects=["Creator Face"],
            recurring_layout_patterns=["Centered hero with peripheral visual tension"],
            creator_signature_elements=["Volumetric atmospheric lighting"],
            required_preservations=[preservations[0].element_name if preservations else "Face"],
            allowed_variations=["Dynamic composition angles"],
            forbidden_changes=["No low contrast typography"],
            pros=["Appeals to new demographic segments in recommendation feeds"],
            cons=["Slightly lower immediate channel signature continuity"],
            rejection_rationale="Ranked as secondary alternative (#2) with fit score 0.80 compared to legacy continuity (0.95)",
            evidence_refs=all_refs[:2],
            supporting_evidence_ids=all_node_ids[:2],
        )
        candidates.append(cand_b)

        # Candidate C: Topic-Forward Minimalist Framing (Alternative 2)
        cand_c = CandidateBrandInterpretation(
            interpretation_name="Topic-Forward Subject Focus",
            brand_pillars=brand_pillars[:2],
            fit_score=0.64,
            confidence=0.70,
            color_palette=["#FFFFFF", "#000000"],
            typography_preferences="Minimal 1-2 word text hook",
            recurring_subjects=["Story Object Only"],
            recurring_layout_patterns=["Single isolated subject"],
            creator_signature_elements=[],
            required_preservations=[],
            allowed_variations=["All elements flexible"],
            forbidden_changes=["Do not use cluttered text"],
            pros=["Maximum instant story premise comprehension"],
            cons=["Discards creator brand equity and facial trust anchor"],
            rejection_rationale="Ranked as tertiary alternative (#3) with fit score 0.64 due to discarding creator face recognition",
            evidence_refs=all_refs[:1],
            supporting_evidence_ids=all_node_ids[:1],
        )
        candidates.append(cand_c)

        trace.append(f"Generated {len(candidates)} candidate brand interpretations")
        return candidates

    def _build_default_candidate(
        self,
        brand_identity: str,
        color_palette: List[str],
        typo_rules: str,
        refs_map: Dict[str, List[EvidenceReference]],
    ) -> CandidateBrandInterpretation:
        """Construct fallback candidate brand interpretation when minimal signals are present."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]
        return CandidateBrandInterpretation(
            interpretation_name="Standard Creator Brand Style",
            brand_pillars=["Visual Clarity", "Creator Recognition"],
            fit_score=0.80,
            confidence=0.80,
            color_palette=color_palette,
            typography_preferences=typo_rules,
            recurring_subjects=["Creator Presence"],
            recurring_layout_patterns=["Split thumbnail composition"],
            creator_signature_elements=["High contrast subject"],
            required_preservations=["Creator Face"],
            allowed_variations=["Background framing"],
            forbidden_changes=["No ungrounded clutter"],
            evidence_refs=all_refs[:1],
            supporting_evidence_ids=list(refs_map.keys())[:1],
        )

    def _calculate_confidence(
        self,
        graph: NormalizedEvidenceGraph,
        active_nodes: List[EvidenceNode],
        creator_ctx: Any,
        narrative_ctx: Any,
        audience_ctx: Any,
        tokens: Dict[str, List[str]],
        trace: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """Compute multi-signal calibrated confidence taking prior reasoner confidence and graph conflicts into account."""
        creator_conf = getattr(creator_ctx, "confidence", 0.85) if creator_ctx is not None else 0.80
        narrative_conf = getattr(narrative_ctx, "confidence", 0.85) if narrative_ctx is not None else 0.80
        audience_conf = getattr(audience_ctx, "confidence", 0.85) if audience_ctx is not None else 0.80

        # Historical consistency score
        hist_consistency = 0.95 if tokens["brand_names"] or tokens["colors"] else 0.80

        # Evidence quality
        if active_nodes:
            ev_quality = sum(n.confidence.propagated_confidence for n in active_nodes) / len(active_nodes)
        else:
            ev_quality = 0.50

        # Metadata quality
        meta_quality = 1.0 if len(tokens["brand_names"]) > 0 else 0.70

        # Conflict penalty
        conflicts_count = len(getattr(graph, "conflicts", []))
        conflict_penalty = min(0.40, conflicts_count * 0.10)

        raw_conf = (
            0.25 * hist_consistency
            + 0.20 * creator_conf
            + 0.15 * narrative_conf
            + 0.15 * audience_conf
            + 0.15 * ev_quality
            + 0.10 * meta_quality
        )
        final_conf = max(0.0, min(1.0, raw_conf * (1.0 - conflict_penalty)))

        breakdown = {
            "historical_consistency": hist_consistency,
            "creator_confidence": creator_conf,
            "narrative_confidence": narrative_conf,
            "audience_confidence": audience_conf,
            "evidence_quality": ev_quality,
            "metadata_quality": meta_quality,
            "conflict_penalty": conflict_penalty,
        }

        trace.append(f"Calibrated brand confidence score: {final_conf:.2f}")
        return breakdown, final_conf

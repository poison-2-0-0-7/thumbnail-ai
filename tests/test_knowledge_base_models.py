"""
Unit tests for Knowledge Base foundational data models.
Tests schema validation, serialization round-trips, frozen immutability,
and domain constraint enforcement.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    ArchetypeMatch,
    BrandConstraint,
    ChannelProfile,
    CompetitorProfile,
    CompetitorStatus,
    CreatorProfile,
    DesignPattern,
    DesignReason,
    DesignReasonType,
    DifferentiationSummary,
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
    IdentityConstraint,
    KnowledgeEntry,
    KnowledgeEntryType,
    PatternScope,
    ThumbnailPattern,
    VisualPattern,
)


def test_evidence_reference_creation_and_serialization() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
        source_id="creator_0",
        source_field="bbox",
        excerpt_or_value="[0.1, 0.2, 0.4, 0.5]",
        confidence=0.95,
    )
    assert ref.source_id == "creator_0"
    assert ref.confidence == 0.95
    assert ref.schema_version == "1.0.0"

    # Serialization round-trip
    d = ref.to_dict()
    assert d["source_id"] == "creator_0"
    assert d["source_type"] == "scene_graph_element"

    json_str = ref.to_json()
    assert "creator_0" in json_str
    reconstructed = EvidenceReference.from_json(json_str)
    assert reconstructed.source_id == ref.source_id
    assert reconstructed.confidence == ref.confidence


def test_evidence_reference_validation_failures() -> None:
    # Empty source_id
    with pytest.raises(ValidationError):
        EvidenceReference(
            source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
            source_id="",
        )

    # Out of range confidence
    with pytest.raises(ValidationError):
        EvidenceReference(
            source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
            source_id="elem_1",
            confidence=1.5,
        )


def test_design_reason_grounding_gate() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.PSYCHOLOGY_DRIVER,
        source_id="curiosity_driver_1",
        excerpt_or_value="Curiosity gap score 0.85",
        confidence=0.9,
    )
    reason = DesignReason(
        reason_id="reason_001",
        claim="Increase hero face scale ratio to 0.35 to match brand anchor",
        reason_type=DesignReasonType.BRAND_CONSISTENCY,
        confidence=0.88,
        evidence=[ref],
        target_element_id="creator_0",
    )
    assert len(reason.evidence) == 1
    assert reason.reason_id == "reason_001"

    # Grounding gate: empty evidence list must be rejected
    with pytest.raises(ValidationError):
        DesignReason(
            reason_id="reason_002",
            claim="Ungrounded claim",
            reason_type=DesignReasonType.CTR_EVIDENCE,
            evidence=[],
        )


def test_knowledge_entry_model() -> None:
    entry = KnowledgeEntry(
        entry_id="entry_hash_001",
        entry_type=KnowledgeEntryType.ARCHETYPE_EXAMPLE,
        embedding=[0.1] * 512,
        embedding_model="OpenCLIP-ViT-B-32",
        archetype_id="big_face_reaction",
        niche="gaming",
        facets={"resolution": "1920x1080", "has_face": True},
    )
    assert entry.entry_id == "entry_hash_001"
    assert len(entry.embedding) == 512
    assert entry.facets["has_face"] is True

    # Empty entry_id validation
    with pytest.raises(ValidationError):
        KnowledgeEntry(
            entry_id="   ",
            entry_type=KnowledgeEntryType.ARCHETYPE_EXAMPLE,
        )


def test_creator_and_channel_profiles() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.BRAND_RULE,
        source_id="brand_rule_01",
        excerpt_or_value="Face ratio consistent 0.35",
    )
    rule = DesignReason(
        reason_id="rule_01",
        claim="Preserve large face placement",
        reason_type=DesignReasonType.BRAND_CONSISTENCY,
        evidence=[ref],
    )
    creator = CreatorProfile(
        creator_id="creator_mrbeast",
        display_name="MrBeast",
        channel_ids=["channel_main", "channel_gaming", "channel_shorts"],
        primary_niche="entertainment",
        brand_rules=[rule],
        cross_channel_consistency_score=0.92,
    )
    assert creator.creator_id == "creator_mrbeast"
    assert len(creator.channel_ids) == 3
    assert len(creator.brand_rules) == 1

    channel = ChannelProfile(
        channel_id="channel_main",
        creator_id="creator_mrbeast",
        niche="entertainment",
        style_embedding_ref="data/creator_style_profiles/channel_main/style_embedding.json",
        profile_established=True,
        sample_count=45,
        archetype_affinity={"big_face_reaction": 0.8},
        dominant_hook_types=["reaction", "challenge"],
        brand_stability_score=0.89,
    )
    assert channel.channel_id == "channel_main"
    assert channel.profile_established is True
    assert channel.sample_count == 45


def test_competitor_profile_model() -> None:
    comp = CompetitorProfile(
        competitor_id="comp_veritasium",
        channel_name="Veritasium",
        niche="science",
        style_embedding=[0.2] * 512,
        dominant_archetypes=["curiosity_gap", "expert_authority"],
        dominant_hook_types=["question", "mystery"],
        color_palette_signature=["#1A1A1A", "#FFFFFF", "#0088FF"],
        text_density_avg=0.15,
        sample_count=30,
        status=CompetitorStatus.ACTIVE,
    )
    assert comp.competitor_id == "comp_veritasium"
    assert comp.status == CompetitorStatus.ACTIVE
    assert len(comp.dominant_archetypes) == 2


def test_archetype_and_match_models() -> None:
    archetype = Archetype(
        archetype_id="big_face_reaction",
        name="Big Face Reaction",
        description="Close up expressive facial reaction",
        defining_scene_graph_pattern={"hero_role": "hero", "hero_bbox_area_min": 0.35},
        typical_hook_types=["reaction", "shock"],
        typical_emotion="surprise",
        niches_observed_in=["entertainment", "gaming"],
        centroid_embedding=[0.0] * 512,
        example_count=120,
    )
    assert archetype.archetype_id == "big_face_reaction"
    assert archetype.defining_scene_graph_pattern["hero_bbox_area_min"] == 0.35

    match = ArchetypeMatch(
        video_id="video_123",
        archetype_id="big_face_reaction",
        match_confidence=0.88,
        matched_via="both",
        runner_up_archetype_ids=["curiosity_gap"],
    )
    assert match.video_id == "video_123"
    assert match.match_confidence == 0.88


def test_brand_and_identity_constraints() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.CREATOR_PROFILE_FIELD,
        source_id="creator_style_01",
        excerpt_or_value="Palette: #FF0000, #FFFFFF",
    )
    brand = BrandConstraint(
        constraint_id="bc_001",
        channel_id="channel_123",
        palette_ref="creator_palette_primary",
        prohibited_elements=["corporate_stock_icons"],
        mandatory_elements=["channel_logo"],
        evidence_refs=[ref],
    )
    assert brand.constraint_id == "bc_001"
    assert "channel_logo" in brand.mandatory_elements

    identity = IdentityConstraint(
        constraint_id="ic_001",
        creator_id="creator_001",
        locked_instances=["creator_0", "logo_0"],
        pose_change_allowed=False,
        face_similarity_threshold=0.92,
        evidence_refs=[ref],
    )
    assert identity.constraint_id == "ic_001"
    assert identity.pose_change_allowed is False
    assert identity.face_similarity_threshold == 0.92


def test_visual_and_design_patterns() -> None:
    vp = VisualPattern(
        pattern_id="rim_light_subject_edge",
        name="Rim Light Subject Edge",
        description="High intensity rim light",
        category="lighting",
        visual_techniques=["edge_lighting"],
        centroid_embedding=[0.0] * 512,
        curated=True,
        evidence_grade=EvidenceGrade.STRONG,
    )
    assert vp.pattern_id == "rim_light_subject_edge"
    assert vp.evidence_grade == EvidenceGrade.STRONG

    dp = DesignPattern(
        pattern_id="curiosity_gap_partial_reveal",
        pattern_scope=PatternScope.AUDIENCE_PSYCHOLOGY,
        name="Curiosity Gap Partial Reveal",
        description="Partial reveal teasing outcome",
        applicable_element_types=["object", "text"],
        frequency_in_niche={"entertainment": 0.7},
    )
    assert dp.pattern_id == "curiosity_gap_partial_reveal"
    assert dp.pattern_scope == PatternScope.AUDIENCE_PSYCHOLOGY

    tp = ThumbnailPattern(
        pattern_id="pattern_composite_01",
        name="High Energy Reaction Composite",
        description="Reaction face with rim lighting and short text hook",
        archetype_id="big_face_reaction",
        visual_pattern_ids=["rim_light_subject_edge"],
        design_pattern_ids=["curiosity_gap_partial_reveal"],
        confidence=0.85,
    )
    assert tp.pattern_id == "pattern_composite_01"
    assert len(tp.visual_pattern_ids) == 1


def test_frozen_model_immutability() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
        source_id="elem_0",
        excerpt_or_value="Face detected",
    )
    with pytest.raises(ValidationError):
        # Mutating frozen model must raise error
        ref.source_id = "elem_1"  # type: ignore

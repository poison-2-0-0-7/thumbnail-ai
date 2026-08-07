"""Tests for Thumbnail Understanding V2 Pydantic Schemas."""

from __future__ import annotations

import json
import pytest
from models import BoundingBox, ResolvedDecision
from thumbnail_understanding import (
    ActionType,
    AIThumbnailDirectorPlan,
    CompositionIntelligence,
    DecomposedScene,
    EditabilityStatus,
    ElementRelationship,
    ElementRole,
    ElementType,
    ImprovementAction,
    LayerCategory,
    ProfessionalImprovementPlan,
    PsychologyDriver,
    SceneElement,
    SceneGraph,
    SceneLayer,
    SpatialRelation,
    ThumbnailPsychologyAssessment,
    ThumbnailUnderstanding,
    VisualHierarchy,
    WeaknessAnalysis,
    WeaknessFinding,
)


def test_scene_element_creation_and_serialization():
    elem = SceneElement(
        element_id="elem_001",
        element_type=ElementType.PERSON,
        category="person",
        label="Creator Face",
        semantic_description="Primary creator speaking to camera",
        bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.9),
        importance_rank=1,
        role=ElementRole.HERO,
        preserve_score=0.95,
        identity_relevance=1.0,
        is_creator=True,
        emotion="surprised",
        emotion_confidence=0.9,
    )
    assert elem.element_id == "elem_001"
    assert elem.role == ElementRole.HERO
    assert elem.is_creator is True

    # JSON roundtrip
    dumped = elem.model_dump_json()
    reloaded = SceneElement.model_validate_json(dumped)
    assert reloaded == elem


def test_scene_graph_and_relationships():
    hero = SceneElement(
        element_id="hero_face",
        element_type=ElementType.PERSON,
        category="person",
        label="Hero Person",
        bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.8),
        importance_rank=1,
        role=ElementRole.HERO,
    )
    obj = SceneElement(
        element_id="prop_mic",
        element_type=ElementType.PROP,
        category="microphone",
        label="Golden Microphone",
        bbox=BoundingBox(x_min=0.4, y_min=0.5, x_max=0.5, y_max=0.7),
        importance_rank=2,
        role=ElementRole.PROP,
    )
    rel = ElementRelationship(
        subject_element_id="hero_face",
        relation=SpatialRelation.HOLDING,
        object_element_id="prop_mic",
        confidence=0.95,
    )
    graph = SceneGraph(
        elements=[hero, obj],
        relationships=[rel],
        hero_element_id="hero_face",
        primary_subject_ids=["hero_face"],
    )
    assert graph.hero_element_id == "hero_face"
    assert len(graph.relationships) == 1
    assert graph.relationships[0].relation == SpatialRelation.HOLDING


def test_full_thumbnail_understanding_serialization():
    understanding = ThumbnailUnderstanding(
        video_id="eWzsmjA1vOo",
        source_thumbnail_path="data/thumbnails/eWzsmjA1vOo.jpg",
        scene_graph=SceneGraph(
            hero_element_id="face_1",
            elements=[
                SceneElement(
                    element_id="face_1",
                    element_type=ElementType.PERSON,
                    category="person",
                    label="Hero Face",
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
                    importance_rank=1,
                    role=ElementRole.HERO,
                )
            ]
        ),
        hierarchy=VisualHierarchy(
            reading_order=["face_1"],
            first_attention_target="face_1",
            dominant_subject_id="face_1",
        ),
        composition=CompositionIntelligence(
            rule_of_thirds_score=0.85,
            subject_placement="center",
        ),
        psychology=ThumbnailPsychologyAssessment(
            ctr_potential_score=0.75,
            curiosity_gap_score=0.8,
            drivers=[
                PsychologyDriver(
                    driver="curiosity",
                    strength=0.8,
                    confidence=0.9,
                    supporting_evidence="Surprised facial expression",
                )
            ],
        ),
        weaknesses=WeaknessAnalysis(
            findings=[
                WeaknessFinding(
                    weakness_type="background_distraction",
                    severity="medium",
                    evidence="Noisy room background",
                    recommended_correction="Replace background with vibrant clean gradient",
                )
            ]
        ),
        decomposed_scene=DecomposedScene(
            layers=[
                SceneLayer(
                    layer_id="bg_layer",
                    category=LayerCategory.BACKGROUND,
                    depth_priority=0,
                ),
                SceneLayer(
                    layer_id="hero_layer",
                    category=LayerCategory.PRIMARY_SUBJECT,
                    element_ref_id="face_1",
                    depth_priority=1,
                ),
            ],
            layer_count=2,
        ),
        director_plan=AIThumbnailDirectorPlan(
            creative_direction="High impact redesign focusing on face emotion and clean background",
            redesign_aggressiveness="moderate",
        ),
        improvement_plan=ProfessionalImprovementPlan(
            actions=[
                ImprovementAction(
                    action_id="act_01",
                    action=ActionType.REPLACE,
                    target_element_id="bg_layer",
                    reason="Remove noisy background",
                    expected_ctr_gain=0.15,
                )
            ]
        ),
        analyzed_at="2026-08-04T02:00:00Z",
    )

    dumped = understanding.model_dump_json()
    reloaded = ThumbnailUnderstanding.model_validate_json(dumped)
    assert reloaded.video_id == "eWzsmjA1vOo"
    assert reloaded.hierarchy.dominant_subject_id == "face_1"
    assert reloaded.improvement_plan.actions[0].expected_ctr_gain == 0.15


def test_resolved_decision_v2_fields():
    from models import TargetElement, DecisionAction, DecisionSource
    decision = ResolvedDecision(
        decision_id="dec_01",
        target=TargetElement(element_id="bg", element_type="background", label="Background"),
        action=DecisionAction.REPLACE,
        confidence=0.9,
        source=DecisionSource.RULE,
        rationale="Replace noisy background",
        priority_rank=1,
        expected_ctr_gain=0.12,
        risk="medium",
        depends_on_decision_ids=["dec_00"],
    )
    assert decision.expected_ctr_gain == 0.12
    assert decision.risk == "medium"
    assert decision.depends_on_decision_ids == ["dec_00"]

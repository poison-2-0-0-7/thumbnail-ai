"""Tests for Thumbnail Understanding Engine V2 components."""

from __future__ import annotations

import pytest
from pathlib import Path
from models import (
    BoundingBox,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    GeminiReasoning,
    OCRResult,
    TextRegion,
    ThumbnailIntelligence,
    VideoMetadata,
)
from thumbnail_understanding import (
    ElementType,
    SpatialRelation,
    ThumbnailUnderstandingEngine,
)
from thumbnail_understanding.hierarchy_calculator import HierarchyCalculator
from thumbnail_understanding.relationship_reasoner import RelationshipReasoner
from thumbnail_understanding.scene_grounding import SceneGrounder


@pytest.fixture
def dummy_intelligence():
    return ThumbnailIntelligence(
        video_id="test_vid_001",
        thumbnail_path="data/thumbnails/test_vid_001.jpg",
        ocr=OCRResult(
            visible_text="SECRET TIPS",
            text_regions=[
                TextRegion(
                    text="SECRET TIPS",
                    confidence=0.95,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.2),
                )
            ],
            word_count=2,
            text_coverage_ratio=0.08,
            average_confidence=0.95,
        ),
        faces=FaceAnalysis(
            face_count=1,
            has_face=True,
            faces=[
                FaceDetail(
                    bbox=BoundingBox(x_min=0.5, y_min=0.2, x_max=0.9, y_max=0.8),
                    detection_confidence=0.98,
                    is_largest=True,
                    emotion="surprised",
                    emotion_confidence=0.92,
                    position_label="right-third",
                )
            ],
        ),
        objects=[
            DetectedObject(
                label="phone",
                confidence=0.85,
                bbox=BoundingBox(x_min=0.6, y_min=0.5, x_max=0.75, y_max=0.7),
            )
        ],
        colors=ColorProfile(
            dominant_colors=["#ff0000", "#000000"],
            warm_or_cool="warm",
        ),
        composition=CompositionAnalysis(
            rule_of_thirds_score=0.8,
            subject_placement="right-third",
            negative_space_ratio=0.35,
            clutter_score=0.25,
            visual_hierarchy_score=0.85,
            text_overlaps_subject=False,
            balance_score=0.75,
        ),
        reasoning=GeminiReasoning(
            ctr_potential_score=0.82,
            curiosity_gap_score=0.88,
            emotional_impact="Surprise and curiosity",
            visual_storytelling_notes="Creator pointing at secret tech product",
            content_mismatch_detected=False,
            strengths=["Strong facial reaction", "Bold readable text"],
            weaknesses=["Background room is a bit plain"],
            redesign_recommendations=["Replace background with gradient lighting"],
            elements_to_preserve=["Creator face", "Headline text"],
        ),
        analyzed_at="2026-08-04T00:00:00Z",
    )


def test_scene_grounding(dummy_intelligence):
    elements = SceneGrounder.ground_elements(dummy_intelligence)
    assert len(elements) >= 4  # bg, face, obj, text

    face_elem = next(e for e in elements if e.element_type == ElementType.PERSON)
    assert face_elem.role.value in ("hero", "primary")
    assert face_elem.emotion == "surprised"
    assert face_elem.is_creator is True

    text_elem = next(e for e in elements if e.element_type == ElementType.TEXT)
    assert "SECRET TIPS" in text_elem.label


def test_hierarchy_calculation(dummy_intelligence):
    elements = SceneGrounder.ground_elements(dummy_intelligence)
    scene_graph, hierarchy = HierarchyCalculator.compute_hierarchy(elements, dummy_intelligence.composition)

    assert scene_graph.hero_element_id is not None
    assert len(hierarchy.reading_order) >= 3
    assert hierarchy.focal_strength_score > 0.0


def test_relationship_reasoner(dummy_intelligence):
    elements = SceneGrounder.ground_elements(dummy_intelligence)
    relationships = RelationshipReasoner.analyze_relationships(elements)

    assert len(relationships) > 0
    bg_rel = next((r for r in relationships if r.relation == SpatialRelation.IN_FRONT_OF), None)
    assert bg_rel is not None


def test_understanding_engine_end_to_end(tmp_path, dummy_intelligence):
    engine = ThumbnailUnderstandingEngine(output_dir=tmp_path)
    understanding = engine.understand(dummy_intelligence)

    assert understanding.video_id == "test_vid_001"
    assert len(understanding.scene_graph.elements) >= 4
    assert understanding.decomposed_scene.layer_count >= 4
    assert len(understanding.improvement_plan.actions) >= 1
    assert (tmp_path / "test_vid_001.json").is_file()

    # Verify reloading from disk
    reloaded = engine.load_understanding("test_vid_001")
    assert reloaded is not None
    assert reloaded.video_id == "test_vid_001"

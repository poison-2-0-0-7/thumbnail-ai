import sys
from pathlib import Path
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from design_blueprint_components.copywriter import (
    author_headline_candidates,
    extract_keywords,
    score_curiosity,
    score_mobile_readability,
    select_hook_types,
)
from models import (
    BoundingBox,
    ColorDirection,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    GeminiReasoning,
    LayoutDirection,
    OCRResult,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    ThumbnailIntelligence,
    VideoMetadata,
)


@pytest.fixture
def dummy_intelligence():
    return ThumbnailIntelligence(
        video_id="test1234567",
        thumbnail_path="data/thumbnails/test1234567.jpg",
        ocr=OCRResult(text_regions=[]),
        faces=FaceAnalysis(faces=[]),
        objects=[DetectedObject(label="laptop", confidence=0.9, bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5))],
        colors=ColorProfile(dominant_colors=[], brightness=0.5, contrast=0.5, saturation=0.5, warm_or_cool="neutral"),
        composition=CompositionAnalysis(clutter_score=0.2, negative_space_ratio=0.4, rule_of_thirds_score=0.8, subject_placement="center", visual_hierarchy_score=0.7),
        reasoning=GeminiReasoning(
            ctr_potential_score=0.7,
            curiosity_gap_score=0.8,
            content_mismatch_detected=False,
            emotional_impact="high",
            visual_storytelling_notes="notes",
            elements_to_preserve=["Python Coding"],
        ),
        analyzed_at="2026-08-01T00:00:00Z",
    )


@pytest.fixture
def dummy_spec():
    return RedesignSpecification(
        video_id="test1234567",
        source_thumbnail_path="data/thumbnails/test1234567.jpg",
        color_direction=ColorDirection(target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="neutral"),
        subject_treatment=SubjectTreatment(has_subject=False),
        text_overlay=TextOverlaySpec(include_text=True),
        layout_direction=LayoutDirection(target_negative_space_ratio=0.4, target_clutter_score=0.2),
        elements_to_preserve=["Python Coding"],
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.8,
        source_content_mismatch_detected=False,
        generated_at="2026-08-01T00:00:00Z",
    )


@pytest.fixture
def dummy_metadata():
    return VideoMetadata(
        video_id="test1234567",
        title="10 Secret Python Coding Hacks You Must Know",
        description="Learn amazing Python tips",
        url="https://youtube.com/watch?v=test1234567",
        uploader="Test",
        uploader_id="test_id",
        channel_id="UC123456789",
    )


def test_extract_keywords(dummy_metadata, dummy_intelligence):
    kw = extract_keywords(dummy_metadata, dummy_intelligence)
    assert "subject" in kw
    assert kw["subject"] == "Python Coding"


def test_select_hook_types(dummy_intelligence, dummy_metadata):
    primary, secondaries = select_hook_types(dummy_intelligence, dummy_metadata)
    assert primary == "curiosity"
    assert len(secondaries) == 2
    assert primary not in secondaries


def test_score_curiosity():
    score1 = score_curiosity("The Secret Behind Python")
    score2 = score_curiosity("Normal Python Guide")
    assert score1 > score2


def test_score_mobile_readability():
    assert score_mobile_readability(30) == 1.0
    assert score_mobile_readability(50) == 0.5
    assert score_mobile_readability(65) == 0.0


def test_author_headline_candidates(dummy_intelligence, dummy_spec, dummy_metadata):
    headline, score, hook, emotion, candidates = author_headline_candidates(
        dummy_intelligence, dummy_spec, dummy_metadata
    )
    assert headline
    assert 0.0 <= score <= 1.0
    assert hook == "curiosity"
    assert len(candidates) >= 3
    h2, s2, hk2, em2, c2 = author_headline_candidates(dummy_intelligence, dummy_spec, dummy_metadata)
    assert headline == h2
    assert score == s2
    assert [c.text for c in candidates] == [c.text for c in c2]

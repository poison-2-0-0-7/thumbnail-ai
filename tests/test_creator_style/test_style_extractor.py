"""Tests for Module 10 StyleExtractor component."""

from __future__ import annotations

import pytest
from modules.creator_style.style_extractor import StyleExtractor
from modules.models import (
    BoundingBox,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    OCRResult,
    TextRegion,
    ThumbnailIntelligence,
)


@pytest.fixture
def mock_intelligence() -> ThumbnailIntelligence:
    return ThumbnailIntelligence(
        video_id="test_video_001",
        thumbnail_path="data/thumbnails/test_video_001.jpg",
        status="success",
        analyzed_at="2026-08-01T00:00:00Z",
        colors=ColorProfile(
            dominant_colors=["#FF0000", "#00FF00"],
            brightness=0.6,
            contrast=0.7,
            saturation=0.8,
            warm_or_cool="warm",
            harmony_score=0.85,
        ),
        composition=CompositionAnalysis(
            subject_placement="center",
            negative_space_ratio=0.35,
            balance_score=0.75,
            symmetry_score=0.50,
            rule_of_thirds_score=0.80,
        ),
        faces=FaceAnalysis(
            has_face=True,
            face_count=1,
            faces=[
                FaceDetail(
                    bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
                    detection_confidence=0.95,
                    is_largest=True,
                )
            ],
        ),
        ocr=OCRResult(
            visible_text="TEST HEADLINE",
            text_regions=[
                TextRegion(
                    text="TEST HEADLINE",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.3),
                )
            ],
            word_count=2,
            text_coverage_ratio=0.16,
            average_confidence=0.9,
        ),
        objects=[
            DetectedObject(
                label="person",
                confidence=0.92,
                bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
            )
        ],
        overall_score=0.88,
    )



def test_style_extractor_signature_derivation(mock_intelligence):
    sig = StyleExtractor.extract_signature(
        video_id="test_video_001",
        channel_id="UC123456789",
        intelligence=mock_intelligence,
    )

    assert sig.video_id == "test_video_001"
    assert sig.channel_id == "UC123456789"
    assert sig.dominant_colors == ["#FF0000", "#00FF00"]
    assert sig.warm_or_cool == "warm"
    assert sig.subject_placement == "center"
    assert sig.face_scale_ratio is not None
    assert pytest.approx(sig.face_scale_ratio, 0.01) == 0.16  # 0.4 * 0.4 = 0.16
    assert sig.text_coverage_ratio == 0.16
    assert sig.text_region_count == 1
    assert "person" in sig.object_classes_present

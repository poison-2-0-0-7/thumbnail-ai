"""
test_headline_planner.py
========================

Unit tests for HeadlinePlanner in planner_components.
"""

from models import (
    AssetExtractionManifest,
    BoundingBox,
    ColorDirection,
    ColorProfile,
    CompositionAnalysis,
    FaceAnalysis,
    HeadlineSource,
    LayoutDirection,
    OCRResult,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    TextRegion,
    ThumbnailIntelligence,
    TypographyAsset,
)
from planner_components.headline_planner import HeadlinePlanner

VALID_HASH = "a" * 64


def make_spec(include_text: bool) -> RedesignSpecification:
    return RedesignSpecification(
        video_id="vid1",
        source_thumbnail_path="thumb.jpg",
        color_direction=ColorDirection(),
        subject_treatment=SubjectTreatment(),
        text_overlay=TextOverlaySpec(include_text=include_text),
        layout_direction=LayoutDirection(),
        source_ctr_potential_score=0.8,
        source_curiosity_gap_score=0.8,
        source_content_mismatch_detected=False,
        generated_at="2026-08-01T00:00:00Z",
    )


def test_headline_planner_no_include_text():
    spec = make_spec(False)
    planner = HeadlinePlanner()
    text, source = planner.plan_headline(spec)
    assert text == ""
    assert source == HeadlineSource.NONE


def test_headline_planner_with_typography_asset():
    spec = make_spec(True)
    manifest = AssetExtractionManifest(
        video_id="vid1",
        source_thumbnail_path="thumb.jpg",
        source_hash=VALID_HASH,
        intelligence_hash=VALID_HASH,
        typography=[
            TypographyAsset(
                text_region_index=0,
                text="ULTIMATE GUIDE",
                crop_path="crop.png",
                bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
                source_text_region_index=0,
            )
        ],
        engine_version="1.0.0",
        extracted_at="2026-08-01T00:00:00Z",
    )
    planner = HeadlinePlanner()
    text, source = planner.plan_headline(spec, extraction_manifest=manifest)
    assert text == "ULTIMATE GUIDE"
    assert source == HeadlineSource.PRESERVED_OCR


def test_headline_planner_with_ocr_intelligence():
    spec = make_spec(True)
    intelligence = ThumbnailIntelligence(
        video_id="vid1",
        thumbnail_path="thumb.jpg",
        ocr=OCRResult(
            text_regions=[TextRegion(text="NEW LESSONS", confidence=0.9, bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10))],
            detected_text_count=1,
            average_confidence=0.9,
        ),
        faces=FaceAnalysis(detected_face_count=0, faces=[]),
        colors=ColorProfile(dominant_colors=[], color_temperature="neutral", contrast_level="medium", brightness_level="medium"),
        composition=CompositionAnalysis(visual_center=(640, 360), rule_of_thirds_alignment=True, clutter_score=0.1, negative_space_ratio=0.5, focal_point=(640, 360)),
        analyzed_at="2026-08-01T00:00:00Z",
    )
    planner = HeadlinePlanner()
    text, source = planner.plan_headline(spec, intelligence=intelligence)
    assert text == "NEW LESSONS"
    assert source == HeadlineSource.PRESERVED_OCR

"""
test_rule_engine.py
===================

Unit tests for RuleEngine and rules subpackage (Phase 2).
"""

from pathlib import Path
import pytest

from modules.decision_components.confidence import (
    calculate_overall_confidence,
    combine_rule_and_llm_confidence,
    recalibrate_llm_confidence,
)
from modules.decision_components.io import DecisionInputBundle
from modules.decision_components.rule_engine import RuleEngine
from modules.models import (
    BoundingBox,
    ColorDirection,
    ColorProfile,
    CompositionAnalysis,
    DecisionAction,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    GenerationParameters,
    LayoutDirection,
    ModelSettings,
    OCRResult,
    PromptPackage,
    QualityParameters,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    TextRegion,
    ThumbnailIntelligence,
)


@pytest.fixture
def sample_bundle() -> DecisionInputBundle:
    video_id = "v_rule_test"

    intel = ThumbnailIntelligence(
        video_id=video_id,
        thumbnail_path="thumb.jpg",
        ocr=OCRResult(
            text_regions=[
                TextRegion(
                    text="OLD CLUTTER",
                    confidence=0.8,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.3, y_max=0.2),
                )
            ]
        ),
        faces=FaceAnalysis(
            face_count=1,
            faces=[
                FaceDetail(
                    bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
                    detection_confidence=0.95,
                    is_largest=True,
                )
            ],
            has_face=True,
        ),
        objects=[
            DetectedObject(
                label="car",
                confidence=0.9,
                bbox=BoundingBox(x_min=0.5, y_min=0.5, x_max=0.8, y_max=0.8),
            )
        ],
        colors=ColorProfile(
            dominant_colors=["#0000ff"],
            palette_hex=["#0000ff"],
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
        ),
        composition=CompositionAnalysis(clutter_score=0.8),
        status="success",
        analyzed_at="2026-07-30T00:00:00Z",
    )

    spec = RedesignSpecification(
        video_id=video_id,
        source_thumbnail_path="thumb.jpg",
        elements_to_preserve=["creator face"],
        text_overlay=TextOverlaySpec(include_text=False),
        color_direction=ColorDirection(warm_or_cool="warm"),
        subject_treatment=SubjectTreatment(),
        layout_direction=LayoutDirection(),
        source_ctr_potential_score=0.4,
        source_curiosity_gap_score=0.5,
        source_content_mismatch_detected=False,
        status="success",
        generated_at="2026-07-30T00:00:00Z",
    )

    prompt_pkg = PromptPackage(
        video_id=video_id,
        positive_prompt="Thumbnail",
        negative_prompt="blurry",
        subject_instructions="creator face",
        background_instructions="background",
        typography_instructions="text",
        composition_instructions="comp",
        lighting_instructions="light",
        color_instructions="color",
        generation_parameters=GenerationParameters(),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        status="success",
        generated_at="2026-07-30T00:00:00Z",
    )

    index = {
        "m4_face_0": {
            "element_id": "m4_face_0",
            "element_type": "face",
            "label": "creator face",
            "bbox": BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
        },
        "m4_text_0": {
            "element_id": "m4_text_0",
            "element_type": "text",
            "label": "OLD CLUTTER",
            "bbox": BoundingBox(x_min=0.0, y_min=0.0, x_max=0.3, y_max=0.2),
        },
    }

    return DecisionInputBundle(
        video_id=video_id,
        intelligence=intel,
        redesign_spec=spec,
        prompt_package=prompt_pkg,
        asset_extraction=None,
        cross_reference_index=index,
    )


def test_rule_engine_evaluate(sample_bundle: DecisionInputBundle):
    engine = RuleEngine()
    candidates = engine.evaluate(sample_bundle)

    assert len(candidates) >= 4

    actions = [c.action for c in candidates]
    assert DecisionAction.KEEP in actions
    assert DecisionAction.REMOVE in actions
    assert DecisionAction.REPLACE in actions
    assert DecisionAction.ENHANCE in actions
    assert DecisionAction.ADD in actions


def test_confidence_helpers():
    assert recalibrate_llm_confidence(0.98) == 0.9
    assert combine_rule_and_llm_confidence(0.8, 0.8) > 0.8
    assert calculate_overall_confidence([0.9, 0.8], soft_warning_count=1) < 0.85

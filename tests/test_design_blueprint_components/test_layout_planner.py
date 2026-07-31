import sys
from pathlib import Path
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from design_blueprint_components.layout_planner import (
    calculate_iou,
    derive_camera_distance,
    resolve_layout,
)
from models import (
    BoundingBox,
    ColorDirection,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    GeminiReasoning,
    LayoutDirection,
    ObjectDirective,
    OCRResult,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    ThumbnailIntelligence,
)


def test_calculate_iou():
    b1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    b2 = BoundingBox(x_min=0.25, y_min=0.0, x_max=0.75, y_max=0.5)
    iou = calculate_iou(b1, b2)
    assert 0.3 < iou < 0.4

    b3 = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)
    assert calculate_iou(b1, b3) == 0.0


def test_derive_camera_distance():
    sub_large = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    assert derive_camera_distance(sub_large, False) == "wide"

    sub_small = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3)
    assert derive_camera_distance(sub_small, True) == "close_up"


def test_resolve_layout_overlapping_conflict():
    face_box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    text_box = BoundingBox(x_min=0.15, y_min=0.15, x_max=0.55, y_max=0.45)

    intel = ThumbnailIntelligence(
        video_id="v1",
        thumbnail_path="thumb.jpg",
        ocr=OCRResult(text_regions=[]),
        faces=FaceAnalysis(faces=[FaceDetail(bbox=face_box, detection_confidence=0.9)]),
        objects=[DetectedObject(label="laptop", confidence=0.8, bbox=text_box)],
        colors=ColorProfile(dominant_colors=[], brightness=0.5, contrast=0.5, saturation=0.5, warm_or_cool="neutral"),
        composition=CompositionAnalysis(clutter_score=0.1),
        reasoning=GeminiReasoning(
            ctr_potential_score=0.7, curiosity_gap_score=0.5, content_mismatch_detected=False,
            emotional_impact="high", visual_storytelling_notes="notes", elements_to_preserve=[]
        ),
        analyzed_at="2026-08-01T00:00:00Z",
    )

    spec = RedesignSpecification(
        video_id="v1",
        source_thumbnail_path="thumb.jpg",
        color_direction=ColorDirection(target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="neutral"),
        subject_treatment=SubjectTreatment(has_subject=True, target_bbox=face_box),
        text_overlay=TextOverlaySpec(include_text=True, placement_zone=text_box),
        layout_direction=LayoutDirection(),
        object_directives=[ObjectDirective(label="laptop", action="include")],
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.5,
        source_content_mismatch_detected=False,
        generated_at="2026-08-01T00:00:00Z",
    )

    text_pos, sub_pos, obj_strat, cam_dist, priority, conflicts = resolve_layout(intel, spec)

    assert conflicts >= 1
    assert text_pos.include_text is True
    assert text_pos.placement_zone_px is not None
    assert sub_pos is not None
    assert "face" in priority
    assert "headline" in priority

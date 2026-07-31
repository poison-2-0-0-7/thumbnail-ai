import sys
from pathlib import Path
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from design_blueprint_engine import (
    build_design_blueprint,
    load_design_blueprint,
    save_design_blueprint,
)
from design_blueprint_exceptions import InvalidRedesignSpecError
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
    VideoMetadata,
)


@pytest.fixture
def sample_data():
    video_id = "v_blueprint_test"
    face_box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)

    intel = ThumbnailIntelligence(
        video_id=video_id,
        thumbnail_path="thumb.jpg",
        ocr=OCRResult(text_regions=[]),
        faces=FaceAnalysis(faces=[FaceDetail(bbox=face_box, detection_confidence=0.9)]),
        objects=[DetectedObject(label="laptop", confidence=0.8, bbox=face_box)],
        colors=ColorProfile(dominant_colors=["#ffffff"], brightness=0.5, contrast=0.5, saturation=0.5, warm_or_cool="warm"),
        composition=CompositionAnalysis(clutter_score=0.2, negative_space_ratio=0.4),
        reasoning=GeminiReasoning(
            ctr_potential_score=0.7, curiosity_gap_score=0.8, content_mismatch_detected=False,
            emotional_impact="high", visual_storytelling_notes="notes", elements_to_preserve=["Laptop"]
        ),
        analyzed_at="2026-08-01T00:00:00Z",
    )

    spec = RedesignSpecification(
        video_id=video_id,
        source_thumbnail_path="thumb.jpg",
        color_direction=ColorDirection(target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="warm"),
        subject_treatment=SubjectTreatment(has_subject=True, target_bbox=face_box),
        text_overlay=TextOverlaySpec(include_text=True, placement_zone=face_box),
        layout_direction=LayoutDirection(target_negative_space_ratio=0.4, target_clutter_score=0.2),
        object_directives=[ObjectDirective(label="laptop", action="preserve")],
        elements_to_preserve=["Laptop"],
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.8,
        source_content_mismatch_detected=False,
        generated_at="2026-08-01T00:00:00Z",
    )

    metadata = VideoMetadata(
        video_id=video_id,
        title="Secrets of Laptop Design Exposed",
        description="Detailed review",
        url="https://youtube.com/watch?v=v_blueprint_test",
        uploader="Tester",
        uploader_id="tester_id",
        channel_id="UC999999999",
    )

    return intel, spec, metadata


def test_build_design_blueprint(sample_data):
    intel, spec, metadata = sample_data
    bp = build_design_blueprint(intel, spec, metadata)

    assert bp.video_id == spec.video_id
    assert bp.headline
    assert bp.headline_score > 0
    assert bp.conflicts_resolved >= 1
    assert bp.face_strategy in ["smile", "neutral", "shock", "exaggerate", "preserve"]
    assert bp.camera_distance in ["close_up", "medium", "wide"]
    assert bp.status in ["success", "partial"]


def test_invalid_spec_raises(sample_data):
    intel, spec, metadata = sample_data
    err_spec = spec.model_copy(update={"status": "error"})

    with pytest.raises(InvalidRedesignSpecError):
        build_design_blueprint(intel, err_spec, metadata)


def test_save_and_load_blueprint(tmp_path: Path, sample_data):
    intel, spec, metadata = sample_data
    bp = build_design_blueprint(intel, spec, metadata)

    saved_path = save_design_blueprint(bp, blueprint_dir=tmp_path)
    assert saved_path.exists()

    loaded = load_design_blueprint(bp.video_id, blueprint_dir=tmp_path)
    assert loaded is not None
    assert loaded.video_id == bp.video_id
    assert loaded.headline == bp.headline
    assert loaded.conflicts_resolved == bp.conflicts_resolved

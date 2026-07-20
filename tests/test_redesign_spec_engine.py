"""Pure unit tests for Module 5's deterministic redesign rule engine."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    CLUTTER_HIGH_THRESHOLD,
    MIN_NEGATIVE_SPACE_RATIO,
)
from models import (  # noqa: E402
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
)
from redesign_spec_engine import (  # noqa: E402
    InvalidIntelligenceError,
    RedesignSpecCacheError,
    _build_overall_rationale,
    _derive_color_direction,
    _derive_layout_direction,
    _derive_object_directives,
    _derive_subject_treatment,
    _derive_text_overlay,
    build_redesign_specification,
    load_cached_redesign_spec,
    save_redesign_spec,
)

VALID_VIDEO_ID = "abcdEFGH123"
FACE_BOX = BoundingBox(x_min=0.10, y_min=0.20, x_max=0.30, y_max=0.50)
OBJECT_BOX = BoundingBox(x_min=0.60, y_min=0.20, x_max=0.90, y_max=0.80)


def _reasoning(**updates: object) -> GeminiReasoning:
    values: dict[str, object] = {
        "ctr_potential_score": 0.65,
        "curiosity_gap_score": 0.45,
        "emotional_impact": "engaging",
        "visual_storytelling_notes": "Clear story.",
        "content_mismatch_detected": False,
        "strengths": ["clear subject"],
        "weaknesses": [],
        "redesign_recommendations": ["simplify"],
        "elements_to_preserve": ["existing text"],
    }
    values.update(updates)
    return GeminiReasoning(**values)


def _intelligence(**updates: object) -> ThumbnailIntelligence:
    values: dict[str, object] = {
        "video_id": VALID_VIDEO_ID,
        "thumbnail_path": "data/thumbnails/abcdEFGH123.jpg",
        "ocr": OCRResult(),
        "faces": FaceAnalysis(),
        "objects": [],
        "colors": ColorProfile(brightness=0.5, contrast=0.5, saturation=0.5),
        "composition": CompositionAnalysis(),
        "reasoning": _reasoning(),
        "analyzed_at": "2026-01-01T00:00:00+00:00",
    }
    values.update(updates)
    return ThumbnailIntelligence(**values)


class TestDeriveColorDirection:
    def test_clamps_values_to_configured_ranges(self) -> None:
        result = _derive_color_direction(
            ColorProfile(brightness=0.1, contrast=0.9, saturation=0.2), []
        )
        assert (result.target_brightness, result.target_contrast, result.target_saturation) == (
            0.35,
            0.8,
            0.3,
        )

    @pytest.mark.parametrize("keyword", ["too warm", "too cool", "color temperature", "washed out"])
    def test_keyword_flips_warm_and_cool(self, keyword: str) -> None:
        result = _derive_color_direction(ColorProfile(warm_or_cool="warm"), [keyword])
        assert result.warm_or_cool == "cool"

    def test_neutral_temperature_is_not_flipped(self) -> None:
        result = _derive_color_direction(ColorProfile(warm_or_cool="neutral"), ["too cool"])
        assert result.warm_or_cool == "neutral"


class TestDeriveSubjectTreatment:
    def test_no_faces_has_no_subject(self) -> None:
        result = _derive_subject_treatment(FaceAnalysis(), CompositionAnalysis())
        assert result.has_subject is False
        assert result.target_bbox is None

    def test_selects_largest_face_and_requests_tighter_crop(self) -> None:
        small = FaceDetail(bbox=BoundingBox(x_min=0, y_min=0, x_max=0.1, y_max=0.1), detection_confidence=0.9)
        large = FaceDetail(bbox=FACE_BOX, detection_confidence=0.6)
        result = _derive_subject_treatment(
            FaceAnalysis(face_count=2, has_face=True, faces=[small, large]),
            CompositionAnalysis(rule_of_thirds_score=0.8),
        )
        assert result.target_bbox == FACE_BOX
        assert result.crop_tighter is True

    def test_low_rule_of_thirds_nudges_bbox(self) -> None:
        result = _derive_subject_treatment(
            FaceAnalysis(face_count=1, has_face=True, faces=[FaceDetail(bbox=FACE_BOX, detection_confidence=0.9)]),
            CompositionAnalysis(rule_of_thirds_score=0.1),
        )
        assert result.target_bbox != FACE_BOX
        assert (result.target_bbox.x_min + result.target_bbox.x_max) / 2 == pytest.approx(1 / 3)


class TestDeriveTextOverlay:
    def test_no_ocr_text_does_not_request_overlay(self) -> None:
        result = _derive_text_overlay(OCRResult(), _derive_subject_treatment(FaceAnalysis(), CompositionAnalysis()))
        assert result.include_text is False
        assert result.placement_zone is None

    def test_existing_text_uses_combined_ocr_region_and_avoids_subject(self) -> None:
        ocr = OCRResult(text_regions=[TextRegion(text="Hello", confidence=0.9, bbox=BoundingBox(x_min=0.7, y_min=0, x_max=0.9, y_max=0.1))])
        subject = _derive_subject_treatment(FaceAnalysis(face_count=1, has_face=True, faces=[FaceDetail(bbox=FACE_BOX, detection_confidence=0.9)]), CompositionAnalysis())
        result = _derive_text_overlay(ocr, subject)
        assert result.include_text is True
        assert result.placement_zone == ocr.text_regions[0].bbox
        assert result.avoid_zones == [subject.target_bbox]

    def test_preserved_text_requests_overlay_without_inventing_copy(self) -> None:
        result = _derive_text_overlay(
            OCRResult(),
            _derive_subject_treatment(FaceAnalysis(), CompositionAnalysis()),
            ("existing text",),
        )
        assert result.include_text is True
        assert result.placement_zone is None


class TestDeriveLayoutAndObjects:
    def test_high_clutter_and_low_negative_space_apply_targets(self) -> None:
        composition = CompositionAnalysis(clutter_score=0.8, negative_space_ratio=0.1)
        result = _derive_layout_direction(composition, _derive_subject_treatment(FaceAnalysis(), composition), [])
        assert result.target_clutter_score == pytest.approx(0.56)
        assert result.target_negative_space_ratio == MIN_NEGATIVE_SPACE_RATIO

    def test_object_becomes_focal_zone_without_face(self) -> None:
        obj = DetectedObject(label="person", confidence=0.8, bbox=OBJECT_BOX)
        result = _derive_layout_direction(CompositionAnalysis(), _derive_subject_treatment(FaceAnalysis(), CompositionAnalysis()), [obj])
        assert result.focal_zone == OBJECT_BOX

    def test_directives_are_sorted_and_change_with_clutter(self) -> None:
        low = DetectedObject(label="bowl", confidence=0.4, bbox=OBJECT_BOX)
        high = DetectedObject(label="person", confidence=0.9, bbox=FACE_BOX)
        result = _derive_object_directives([low, high], CompositionAnalysis(clutter_score=CLUTTER_HIGH_THRESHOLD + 0.1))
        assert [directive.label for directive in result] == ["person", "bowl"]
        assert {directive.action for directive in result} == {"remove"}


class TestBuildRedesignSpecification:
    def test_builds_complete_specification_and_passes_through_preserved_elements(self) -> None:
        intelligence = _intelligence(
            composition=CompositionAnalysis(clutter_score=0.9, negative_space_ratio=0.1),
            reasoning=_reasoning(elements_to_preserve=["faces", "text"]),
        )
        result = build_redesign_specification(intelligence)
        assert result.video_id == VALID_VIDEO_ID
        assert result.elements_to_preserve == ["faces", "text"]
        assert result.status == "success"
        assert "clutter_reduction" in result.overall_rationale

    @pytest.mark.parametrize("updates", [{"reasoning": None}, {"status": "error"}])
    def test_rejects_unusable_intelligence(self, updates: dict[str, object]) -> None:
        with pytest.raises(InvalidIntelligenceError):
            build_redesign_specification(_intelligence(**updates))

    def test_rationale_is_fixed_format(self) -> None:
        assert _build_overall_rationale(_reasoning(), ["crop_tighter"]) == (
            "ctr_potential_score=0.65; curiosity_gap_score=0.45; "
            "content_mismatch_detected=False; rules applied: crop_tighter."
        )


class TestRedesignSpecPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        spec = build_redesign_specification(_intelligence())
        save_redesign_spec(spec, tmp_path)
        assert load_cached_redesign_spec(VALID_VIDEO_ID, tmp_path) == spec
        assert not (tmp_path / f"{VALID_VIDEO_ID}.tmp").exists()

    def test_missing_or_corrupted_cache_returns_none(self, tmp_path: Path) -> None:
        assert load_cached_redesign_spec(VALID_VIDEO_ID, tmp_path) is None
        (tmp_path / f"{VALID_VIDEO_ID}.json").write_text("not json", encoding="utf-8")
        assert load_cached_redesign_spec(VALID_VIDEO_ID, tmp_path) is None

    def test_write_error_raises_cache_exception(self, tmp_path: Path) -> None:
        spec = build_redesign_specification(_intelligence())
        blocked_target = tmp_path / "not-a-directory"
        blocked_target.write_text("block", encoding="utf-8")
        with pytest.raises(RedesignSpecCacheError):
            save_redesign_spec(spec, blocked_target)

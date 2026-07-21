"""Unit tests for Module 6's deterministic prompt compiler."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import BASE_NEGATIVE_PROMPT_TERMS, BASE_QUALITY_TAGS, SAFETY_CONSTRAINTS  # noqa: E402
from models import (  # noqa: E402
    BoundingBox,
    ColorDirection,
    LayoutDirection,
    ObjectDirective,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
)
from prompt_compiler import (  # noqa: E402
    InvalidRedesignSpecError,
    PromptCompilerError,
    PromptPackageCacheError,
    _compile_background_instructions,
    _compile_color_instructions,
    _compile_composition_instructions,
    _compile_lighting_instructions,
    _compile_negative_prompt,
    _compile_object_placement,
    _compile_positive_prompt,
    _compile_rendering_constraints,
    _compile_subject_instructions,
    _compile_typography_instructions,
    _derive_seed,
    _zone_label,
    compile_prompt_package,
    load_cached_prompt_package,
    save_prompt_package,
)


VIDEO_ID = "abcdEFGH123"
TOP_LEFT = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2)
CENTER = BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6)
RIGHT_BOTTOM = BoundingBox(x_min=0.8, y_min=0.8, x_max=1.0, y_max=1.0)


def _spec(**updates: object) -> RedesignSpecification:
    values: dict[str, object] = {
        "video_id": VIDEO_ID,
        "source_thumbnail_path": "data/thumbnails/abcdEFGH123.jpg",
        "color_direction": ColorDirection(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        "subject_treatment": SubjectTreatment(has_subject=False),
        "text_overlay": TextOverlaySpec(),
        "layout_direction": LayoutDirection(),
        "object_directives": [],
        "elements_to_preserve": [],
        "overall_rationale": "test rationale",
        "source_ctr_potential_score": 0.6,
        "source_curiosity_gap_score": 0.4,
        "source_content_mismatch_detected": False,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    values.update(updates)
    return RedesignSpecification(**values)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [(None, "center"), (CENTER, "center"), (TOP_LEFT, "top-left"), (RIGHT_BOTTOM, "bottom-right")],
)
def test_zone_label_is_a_deterministic_three_by_three_grid(
    bbox: BoundingBox | None, expected: str
) -> None:
    assert _zone_label(bbox) == expected


@pytest.mark.parametrize(
    ("clutter", "phrase"),
    [(0.1, "minimal and uncluttered"), (0.5, "moderately simple"), (0.9, "acceptable")],
)
def test_background_compilation_uses_only_configured_target_values(
    clutter: float, phrase: str
) -> None:
    result = _compile_background_instructions(
        LayoutDirection(target_negative_space_ratio=0.25, target_clutter_score=clutter)
    )
    assert "25%" in result
    assert phrase in result


def test_subject_and_typography_instructions_do_not_invent_content() -> None:
    assert "without inventing" in _compile_subject_instructions(SubjectTreatment(has_subject=False))
    result = _compile_typography_instructions(
        TextOverlaySpec(include_text=True, placement_zone=TOP_LEFT, avoid_zones=[CENTER])
    )
    assert "do not invent new copy" in result
    assert "top-left" in result
    assert "center" in result


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        (ColorDirection(target_brightness=0.8), "bright, well-lit"),
        (ColorDirection(target_brightness=0.2), "moody, low-key"),
        (ColorDirection(target_contrast=0.8), "high-contrast"),
        (ColorDirection(target_saturation=0.2, warm_or_cool="warm"), "muted, desaturated"),
    ],
)
def test_lighting_and_color_compilation_have_fixed_thresholds(
    color: ColorDirection, expected: str
) -> None:
    combined = _compile_lighting_instructions(color) + _compile_color_instructions(color)
    assert expected in combined


def test_negative_prompt_adds_only_module_five_removal_directives() -> None:
    result = _compile_negative_prompt(
        [ObjectDirective(label="bowl", action="remove"), ObjectDirective(label="person", action="preserve")]
    )
    assert all(term in result for term in BASE_NEGATIVE_PROMPT_TERMS)
    assert "no bowl" in result
    assert "no person" not in result


def test_compilers_preserve_module_five_order_and_targets() -> None:
    directives = [
        ObjectDirective(label="person", action="preserve"),
        ObjectDirective(label="bowl", action="remove"),
    ]
    assert _compile_object_placement(directives) == [
        "person: preserve",
        "bowl: remove",
    ]
    composition = _compile_composition_instructions(
        LayoutDirection(
            focal_zone=RIGHT_BOTTOM,
            target_clutter_score=0.42,
            target_negative_space_ratio=0.33,
        )
    )
    assert "bottom-right" in composition
    assert "0.42" in composition
    assert "0.33" in composition


def test_positive_prompt_order_and_rendering_constraints_are_fixed() -> None:
    assert _compile_positive_prompt("A", "B", "C", "D", "E", "F", []) == "A B C D E F"
    spec = _spec(
        elements_to_preserve=["existing text"],
        text_overlay=TextOverlaySpec(include_text=False),
    )
    constraints = _compile_rendering_constraints(spec)
    assert any("existing text" in constraint for constraint in constraints)
    assert "Do not add any text to the rendered image." in constraints


def test_seed_is_stable_across_calls_and_distinguishes_video_ids() -> None:
    assert _derive_seed(VIDEO_ID) == _derive_seed(VIDEO_ID)
    assert _derive_seed(VIDEO_ID) != _derive_seed("zzzzzzzzzzz")


def test_compiler_builds_complete_package_from_specification_only() -> None:
    package = compile_prompt_package(
        _spec(
            subject_treatment=SubjectTreatment(has_subject=True, target_bbox=CENTER, crop_tighter=True),
            color_direction=ColorDirection(warm_or_cool="warm"),
            object_directives=[ObjectDirective(label="phone", action="include")],
        )
    )
    assert package.status == "success"
    assert package.generation_parameters.seed == _derive_seed(VIDEO_ID)
    assert package.quality_parameters.quality_tags == list(BASE_QUALITY_TAGS)
    assert package.quality_parameters.upscale_requested is True
    assert package.model_settings.style_preset == "warm-photographic"
    assert package.object_placement == ["phone: include"]
    assert package.safety_constraints == list(SAFETY_CONSTRAINTS)


def test_compiler_rejects_error_specification() -> None:
    with pytest.raises(InvalidRedesignSpecError):
        compile_prompt_package(_spec(status="error", error_message="unusable"))


def test_compiled_content_is_repeatable_for_identical_input() -> None:
    first = compile_prompt_package(_spec())
    second = compile_prompt_package(_spec())
    assert first.positive_prompt == second.positive_prompt
    assert first.negative_prompt == second.negative_prompt
    assert first.generation_parameters == second.generation_parameters
    assert first.quality_parameters == second.quality_parameters
    assert first.model_settings == second.model_settings
    assert first.rendering_constraints == second.rendering_constraints


def test_prompt_package_persistence_round_trip_and_bad_cache(tmp_path: Path) -> None:
    package = compile_prompt_package(_spec())
    save_prompt_package(package, tmp_path)
    assert load_cached_prompt_package(VIDEO_ID, tmp_path) == package
    (tmp_path / f"{VIDEO_ID}.json").write_text("not json", encoding="utf-8")
    assert load_cached_prompt_package(VIDEO_ID, tmp_path) is None


def test_prompt_package_persistence_wraps_write_errors(tmp_path: Path) -> None:
    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("block", encoding="utf-8")
    with pytest.raises(PromptPackageCacheError):
        save_prompt_package(compile_prompt_package(_spec()), blocked_path)


def test_prompt_compiler_exceptions_share_the_module_base_class() -> None:
    assert issubclass(InvalidRedesignSpecError, PromptCompilerError)
    assert issubclass(PromptPackageCacheError, PromptCompilerError)

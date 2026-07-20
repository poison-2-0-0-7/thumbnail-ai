"""Deterministic Module 5 redesign-specification engine.

This module converts Module 4's :class:`ThumbnailIntelligence` report into
an execution-ready :class:`RedesignSpecification`.  It never opens image
pixels, calls a network service, invokes an AI/LLM, or invents creative
content.  Module 4 remains the pipeline's only reasoning stage.

Specifications are stored as atomic JSON files in ``data/redesign_specs``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_MODULES_DIR = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger

from config import (
    BRIGHTNESS_TARGET_RANGE,
    CLUTTER_HIGH_THRESHOLD,
    CLUTTER_REDUCTION_FACTOR,
    COLOR_TEMPERATURE_FLIP_KEYWORDS,
    CONTRAST_TARGET_RANGE,
    DEFAULT_REDESIGN_SPEC_DIR,
    LOG_DIR,
    MIN_NEGATIVE_SPACE_RATIO,
    MIN_SUBJECT_AREA_RATIO,
    MODULE5_LOG_PATH,
    REDESIGN_SPEC_FILENAME_TEMPLATE,
    RULE_OF_THIRDS_LOW_THRESHOLD,
    SATURATION_TARGET_RANGE,
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
    ObjectDirective,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    ThumbnailIntelligence,
)


_LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """Attach the Module 5 rotating Loguru sink."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE5_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


class RedesignSpecError(Exception):
    """Base exception for Module 5 failures."""


class InvalidIntelligenceError(RedesignSpecError):
    """Raised when a Module 4 report cannot support deterministic redesign."""


class RedesignSpecCacheError(RedesignSpecError):
    """Raised when a redesign-specification cache cannot be written."""


def _bbox_area(bbox: BoundingBox) -> float:
    """Return the normalized area of ``bbox``."""
    return max(0.0, bbox.x_max - bbox.x_min) * max(0.0, bbox.y_max - bbox.y_min)


def _clamp_to_range(value: float, target_range: tuple[float, float]) -> float:
    """Keep a value inside its configured acceptable band."""
    low, high = target_range
    return max(low, min(high, value))


def _derive_color_direction(
    colors: ColorProfile,
    weaknesses: list[str],
) -> ColorDirection:
    """Derive bounded color targets and a rule-based temperature direction."""
    joined_weaknesses = " ".join(weaknesses).lower()
    should_flip = any(
        keyword in joined_weaknesses for keyword in COLOR_TEMPERATURE_FLIP_KEYWORDS
    )
    temperature = colors.warm_or_cool
    if should_flip and temperature in {"warm", "cool"}:
        temperature = "cool" if temperature == "warm" else "warm"

    return ColorDirection(
        target_brightness=_clamp_to_range(colors.brightness, BRIGHTNESS_TARGET_RANGE),
        target_contrast=_clamp_to_range(colors.contrast, CONTRAST_TARGET_RANGE),
        target_saturation=_clamp_to_range(colors.saturation, SATURATION_TARGET_RANGE),
        warm_or_cool=temperature,
        rationale=(
            "Color targets clamp source brightness, contrast, and saturation "
            "to their configured acceptable ranges"
            + ("; color-temperature weakness triggered a temperature flip." if should_flip else ".")
        ),
    )


def _nudge_to_nearest_thirds_intersection(bbox: BoundingBox) -> BoundingBox:
    """Move a bbox center to its nearest rule-of-thirds intersection."""
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min
    center_x = (bbox.x_min + bbox.x_max) / 2
    center_y = (bbox.y_min + bbox.y_max) / 2
    target_x = min((1 / 3, 2 / 3), key=lambda value: abs(value - center_x))
    target_y = min((1 / 3, 2 / 3), key=lambda value: abs(value - center_y))
    return BoundingBox(
        x_min=target_x - width / 2,
        y_min=target_y - height / 2,
        x_max=target_x + width / 2,
        y_max=target_y + height / 2,
    )


def _derive_subject_treatment(
    faces: FaceAnalysis,
    composition: CompositionAnalysis,
) -> SubjectTreatment:
    """Select the largest face and apply crop/placement rules."""
    if not faces.faces:
        return SubjectTreatment(
            has_subject=False,
            target_position_label=composition.subject_placement,
            rationale="No detected face is available for subject treatment.",
        )

    largest_face = max(faces.faces, key=lambda face: _bbox_area(face.bbox))
    source_bbox = largest_face.bbox
    target_bbox = source_bbox
    nudged = composition.rule_of_thirds_score < RULE_OF_THIRDS_LOW_THRESHOLD
    if nudged:
        target_bbox = _nudge_to_nearest_thirds_intersection(source_bbox)

    crop_tighter = _bbox_area(source_bbox) < MIN_SUBJECT_AREA_RATIO
    return SubjectTreatment(
        has_subject=True,
        target_bbox=target_bbox,
        target_position_label=composition.subject_placement,
        crop_tighter=crop_tighter,
        rationale=(
            "Largest detected face selected"
            + (" and nudged to the nearest rule-of-thirds intersection" if nudged else "")
            + ("; crop tighter because subject area is below the minimum." if crop_tighter else ".")
        ),
    )


def _derive_text_overlay(
    ocr: OCRResult,
    subject: SubjectTreatment,
    elements_to_preserve: tuple[str, ...] = (),
) -> TextOverlaySpec:
    """Preserve existing text placement without creating new copy."""
    preserve_text = any("text" in element.lower() for element in elements_to_preserve)
    if not ocr.text_regions and not preserve_text:
        return TextOverlaySpec(
            rationale="No existing OCR text was detected; no text is requested."
        )

    if not ocr.text_regions:
        return TextOverlaySpec(
            include_text=True,
            avoid_zones=[subject.target_bbox] if subject.target_bbox is not None else [],
            rationale="Module 4 marked existing text for preservation; no new copy is created.",
        )

    x_min = min(region.bbox.x_min for region in ocr.text_regions)
    y_min = min(region.bbox.y_min for region in ocr.text_regions)
    x_max = max(region.bbox.x_max for region in ocr.text_regions)
    y_max = max(region.bbox.y_max for region in ocr.text_regions)
    return TextOverlaySpec(
        include_text=True,
        placement_zone=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        avoid_zones=[subject.target_bbox] if subject.target_bbox is not None else [],
        rationale="Existing OCR text is preserved as placement-only guidance.",
    )


def _derive_layout_direction(
    composition: CompositionAnalysis,
    subject: SubjectTreatment,
    objects: Optional[list[DetectedObject]] = None,
) -> LayoutDirection:
    """Derive composition targets and the focal-zone priority ordering."""
    high_clutter = composition.clutter_score > CLUTTER_HIGH_THRESHOLD
    low_negative_space = composition.negative_space_ratio < MIN_NEGATIVE_SPACE_RATIO
    focal_zone = subject.target_bbox
    if focal_zone is None and objects:
        focal_zone = max(objects, key=lambda item: item.confidence).bbox

    return LayoutDirection(
        target_negative_space_ratio=(
            MIN_NEGATIVE_SPACE_RATIO if low_negative_space else composition.negative_space_ratio
        ),
        target_clutter_score=(
            composition.clutter_score * CLUTTER_REDUCTION_FACTOR
            if high_clutter
            else composition.clutter_score
        ),
        focal_zone=focal_zone,
        rationale=(
            "Layout targets preserve source values"
            + (" except clutter is reduced." if high_clutter else ".")
            + (" Negative space is raised to the configured minimum." if low_negative_space else "")
        ),
    )


def _derive_object_directives(
    objects: list[DetectedObject],
    composition: CompositionAnalysis,
) -> list[ObjectDirective]:
    """Flag objects for removal only when high clutter requires simplification."""
    action = "remove" if composition.clutter_score > CLUTTER_HIGH_THRESHOLD else "preserve"
    ordered_objects = sorted(
        objects,
        key=lambda item: (-item.confidence, -_bbox_area(item.bbox), item.label),
    )
    rationale = (
        "High clutter requires simplifying detected objects."
        if action == "remove"
        else "Clutter is within the acceptable range; preserve detected objects."
    )
    return [
        ObjectDirective(label=item.label, action=action, rationale=rationale)
        for item in ordered_objects
    ]


def _build_overall_rationale(reasoning: GeminiReasoning, fired_rules: list[str]) -> str:
    """Build the fixed-format, non-generative overall rationale."""
    rules = ", ".join(fired_rules) if fired_rules else "none"
    return (
        f"ctr_potential_score={reasoning.ctr_potential_score:.2f}; "
        f"curiosity_gap_score={reasoning.curiosity_gap_score:.2f}; "
        f"content_mismatch_detected={reasoning.content_mismatch_detected}; "
        f"rules applied: {rules}."
    )


def build_redesign_specification(
    intelligence: ThumbnailIntelligence,
) -> RedesignSpecification:
    """Deterministically derive a redesign specification from Module 4 output."""
    if intelligence.status == "error" or intelligence.reasoning is None:
        raise InvalidIntelligenceError(
            "ThumbnailIntelligence must have non-error status and reasoning to build "
            "a redesign specification"
        )

    started_at = time.monotonic()
    reasoning = intelligence.reasoning
    color_direction = _derive_color_direction(intelligence.colors, reasoning.weaknesses)
    subject_treatment = _derive_subject_treatment(intelligence.faces, intelligence.composition)
    text_overlay = _derive_text_overlay(
        intelligence.ocr,
        subject_treatment,
        tuple(reasoning.elements_to_preserve),
    )
    layout_direction = _derive_layout_direction(
        intelligence.composition, subject_treatment, intelligence.objects
    )
    object_directives = _derive_object_directives(
        intelligence.objects, intelligence.composition
    )

    fired_rules: list[str] = []
    if intelligence.composition.clutter_score > CLUTTER_HIGH_THRESHOLD:
        fired_rules.append("clutter_reduction")
    if intelligence.composition.negative_space_ratio < MIN_NEGATIVE_SPACE_RATIO:
        fired_rules.append("minimum_negative_space")
    if (
        subject_treatment.has_subject
        and intelligence.composition.rule_of_thirds_score < RULE_OF_THIRDS_LOW_THRESHOLD
    ):
        fired_rules.append("rule_of_thirds_nudge")
    if subject_treatment.crop_tighter:
        fired_rules.append("crop_tighter")
    if any(
        keyword in " ".join(reasoning.weaknesses).lower()
        for keyword in COLOR_TEMPERATURE_FLIP_KEYWORDS
    ):
        fired_rules.append("color_temperature_flip")

    specification = RedesignSpecification(
        video_id=intelligence.video_id,
        source_thumbnail_path=intelligence.thumbnail_path,
        color_direction=color_direction,
        subject_treatment=subject_treatment,
        text_overlay=text_overlay,
        layout_direction=layout_direction,
        object_directives=object_directives,
        elements_to_preserve=reasoning.elements_to_preserve,
        overall_rationale=_build_overall_rationale(reasoning, fired_rules),
        source_ctr_potential_score=reasoning.ctr_potential_score,
        source_curiosity_gap_score=reasoning.curiosity_gap_score,
        source_content_mismatch_detected=reasoning.content_mismatch_detected,
        duration_seconds=time.monotonic() - started_at,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "Redesign specification complete for video_id={id} ({dur:.4f}s)",
        id=intelligence.video_id,
        dur=specification.duration_seconds,
    )
    return specification


def _redesign_spec_path(video_id: str, spec_dir: Path) -> Path:
    """Return the canonical JSON path for a redesign specification."""
    return spec_dir / REDESIGN_SPEC_FILENAME_TEMPLATE.format(video_id=video_id)


def save_redesign_spec(
    spec: RedesignSpecification,
    spec_dir: Path = DEFAULT_REDESIGN_SPEC_DIR,
) -> None:
    """Persist a redesign specification atomically as JSON."""
    target = _redesign_spec_path(spec.video_id, spec_dir)
    tmp = target.with_suffix(".tmp")
    try:
        spec_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.debug("Saved redesign specification for video_id={id} -> {path}", id=spec.video_id, path=target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        logger.error("Failed to save redesign specification for video_id={id}: {exc}", id=spec.video_id, exc=exc)
        raise RedesignSpecCacheError(
            f"Could not write redesign specification to {target}: {exc}"
        ) from exc


def load_cached_redesign_spec(
    video_id: str,
    spec_dir: Path = DEFAULT_REDESIGN_SPEC_DIR,
) -> Optional[RedesignSpecification]:
    """Load a valid cached specification, or return ``None`` on a cache miss."""
    path = _redesign_spec_path(video_id, spec_dir)
    if not path.exists():
        logger.debug("Redesign specification cache miss for video_id={id}", id=video_id)
        return None
    try:
        cached = RedesignSpecification.model_validate_json(path.read_text(encoding="utf-8"))
        logger.debug("Redesign specification cache hit for video_id={id}: {path}", id=video_id, path=path)
        return cached
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Cached redesign specification for video_id={id} is unreadable ({reason}) — treating as cache miss",
            id=video_id,
            reason=exc,
        )
        return None

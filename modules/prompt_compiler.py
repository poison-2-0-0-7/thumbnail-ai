"""Deterministic Module 6 prompt-compilation engine.

This module compiles a Module 5 :class:`RedesignSpecification` into a
:class:`PromptPackage` for Module 7. It is a compiler, not a reasoning or
generation system: every value is a fixed template, direct pass-through, or
deterministic rule-table transformation of the supplied specification.

It makes no network calls, invokes no AI/LLM, opens no image pixels, and does
not invent copy, subjects, colours, or object directives. Prompt packages are
persisted atomically in ``data/prompt_packages``.
"""

from __future__ import annotations

import hashlib
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
    BASE_NEGATIVE_PROMPT_TERMS,
    BASE_QUALITY_TAGS,
    BASE_RENDERING_CONSTRAINTS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_GENERATION_HEIGHT,
    DEFAULT_GENERATION_WIDTH,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MIN_RESOLUTION_PX,
    DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_PROMPT_WEIGHT,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_SAMPLER,
    DEFAULT_STYLE_PRESET,
    LOG_DIR,
    MODULE6_LOG_PATH,
    PROMPT_PACKAGE_FILENAME_TEMPLATE,
    SAFETY_CONSTRAINTS,
    SEED_HASH_MODULUS,
    ZONE_THIRD_THRESHOLD,
)
from models import (
    BoundingBox,
    ColorDirection,
    GenerationParameters,
    LayoutDirection,
    ModelSettings,
    ObjectDirective,
    PromptPackage,
    QualityParameters,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
)


_LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """Attach the Module 6 rotating Loguru sink, matching Modules 4 and 5."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE6_LOG_PATH), rotation="10 MB", retention="30 days",
        format=_LOG_FORMAT, level="DEBUG", enqueue=True,
    )


_configure_logger()


class PromptCompilerError(Exception):
    """Base exception for Module 6 failures."""


class InvalidRedesignSpecError(PromptCompilerError):
    """Raised when a Module 5 specification cannot be compiled."""


class PromptPackageCacheError(PromptCompilerError):
    """Raised when a prompt-package cache cannot be written."""


def _zone_label(bbox: Optional[BoundingBox]) -> str:
    """Return a coarse 3x3 zone for a bounding box, or ``center``."""
    if bbox is None:
        return "center"

    center_x = (bbox.x_min + bbox.x_max) / 2
    center_y = (bbox.y_min + bbox.y_max) / 2
    high_threshold = 1 - ZONE_THIRD_THRESHOLD
    horizontal = (
        "left" if center_x < ZONE_THIRD_THRESHOLD
        else "right" if center_x > high_threshold else "center"
    )
    vertical = (
        "top" if center_y < ZONE_THIRD_THRESHOLD
        else "bottom" if center_y > high_threshold else "middle"
    )
    if horizontal == "center" and vertical == "middle":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"


def _compile_subject_instructions(subject: SubjectTreatment) -> str:
    """Compile source-derived subject placement and crop guidance."""
    if not subject.has_subject:
        return (
            "No primary subject was specified; compose the thumbnail without "
            "inventing a dominant foreground figure."
        )
    sentence = (
        f"Place the primary subject in the {_zone_label(subject.target_bbox)} of the frame, "
        f"matching the source position label '{subject.target_position_label}'."
    )
    if subject.crop_tighter:
        sentence += " Crop tighter around the subject so it occupies a larger share of the frame."
    return sentence


def _compile_background_instructions(layout: LayoutDirection) -> str:
    """Compile deterministic background and negative-space guidance."""
    sentence = f"Maintain approximately {layout.target_negative_space_ratio:.0%} negative space in the background."
    if layout.target_clutter_score <= 0.3:
        return sentence + " Keep the background minimal and uncluttered."
    if layout.target_clutter_score <= 0.6:
        return sentence + " Keep the background moderately simple with limited competing elements."
    return sentence + " Background detail is acceptable within the configured clutter target."


def _compile_typography_instructions(text_overlay: TextOverlaySpec) -> str:
    """Compile placement-only typography guidance; never invent copy."""
    if not text_overlay.include_text:
        return "Do not render any text overlay on the thumbnail."
    sentence = "Preserve existing text placement; do not invent new copy or wording."
    if text_overlay.placement_zone is not None:
        sentence += f" Reserve the {_zone_label(text_overlay.placement_zone)} region of the frame for text."
    if text_overlay.avoid_zones:
        zones = ", ".join(_zone_label(zone) for zone in text_overlay.avoid_zones)
        sentence += f" Avoid placing text over the following region(s): {zones}."
    return sentence


def _compile_composition_instructions(layout: LayoutDirection) -> str:
    """Compile focal-point and composition targets."""
    return (
        f"Anchor the composition's focal point in the {_zone_label(layout.focal_zone)} of the frame, "
        f"targeting a clutter score of {layout.target_clutter_score:.2f} and "
        f"a negative-space ratio of {layout.target_negative_space_ratio:.2f}."
    )


def _compile_lighting_instructions(color: ColorDirection) -> str:
    """Compile lighting guidance from Module 5 brightness and contrast targets."""
    brightness = "bright, well-lit" if color.target_brightness >= 0.6 else "moody, low-key" if color.target_brightness <= 0.4 else "balanced"
    contrast = "high-contrast" if color.target_contrast >= 0.6 else "soft-contrast" if color.target_contrast <= 0.4 else "moderate-contrast"
    return (
        f"Use {brightness} lighting with a {contrast} look "
        f"(target brightness={color.target_brightness:.2f}, target contrast={color.target_contrast:.2f})."
    )


def _compile_color_instructions(color: ColorDirection) -> str:
    """Compile colour guidance from the already-derived Module 5 targets."""
    saturation = "vivid, highly saturated" if color.target_saturation >= 0.6 else "muted, desaturated" if color.target_saturation <= 0.4 else "moderately saturated"
    return (
        f"Render {saturation} colors with a {color.warm_or_cool} color temperature "
        f"(target saturation={color.target_saturation:.2f})."
    )


def _compile_object_placement(directives: list[ObjectDirective]) -> list[str]:
    """Pass object directives through in their deliberate Module 5 order."""
    return [f"{directive.label}: {directive.action}" for directive in directives]


def _compile_rendering_constraints(spec: RedesignSpecification) -> list[str]:
    """Return fixed constraints plus direct preservation instructions."""
    constraints = list(BASE_RENDERING_CONSTRAINTS)
    if spec.elements_to_preserve:
        constraints.append("Preserve the following elements exactly: " + ", ".join(spec.elements_to_preserve) + ".")
    if not spec.text_overlay.include_text:
        constraints.append("Do not add any text to the rendered image.")
    return constraints


def _compile_positive_prompt(
    subject_instructions: str,
    background_instructions: str,
    composition_instructions: str,
    lighting_instructions: str,
    color_instructions: str,
    typography_instructions: str,
    elements_to_preserve: list[str],
) -> str:
    """Assemble instruction segments in a fixed order."""
    compiled = [
        subject_instructions,
        background_instructions,
        composition_instructions,
        lighting_instructions,
        color_instructions,
        typography_instructions,
    ]
    if elements_to_preserve:
        compiled.append("Preserve: " + ", ".join(elements_to_preserve) + ".")
    return " ".join(compiled)


def _compile_negative_prompt(directives: list[ObjectDirective]) -> str:
    """Combine fixed exclusions with objects Module 5 marked for removal."""
    terms = list(BASE_NEGATIVE_PROMPT_TERMS)
    terms.extend(f"no {item.label}" for item in directives if item.action == "remove")
    return ", ".join(terms)


def _derive_seed(video_id: str) -> int:
    """Derive a stable cross-process seed from the video ID."""
    return int(hashlib.sha256(video_id.encode("utf-8")).hexdigest(), 16) % SEED_HASH_MODULUS


def _compile_generation_parameters(video_id: str) -> GenerationParameters:
    return GenerationParameters(
        width=DEFAULT_GENERATION_WIDTH, height=DEFAULT_GENERATION_HEIGHT,
        aspect_ratio=DEFAULT_ASPECT_RATIO, seed=_derive_seed(video_id),
        guidance_scale=DEFAULT_GUIDANCE_SCALE, inference_steps=DEFAULT_INFERENCE_STEPS,
        sampler=DEFAULT_SAMPLER,
    )


def _compile_quality_parameters(subject: SubjectTreatment) -> QualityParameters:
    return QualityParameters(
        quality_tags=list(BASE_QUALITY_TAGS), min_resolution_px=DEFAULT_MIN_RESOLUTION_PX,
        upscale_requested=subject.crop_tighter,
    )


def _compile_model_settings(color: ColorDirection) -> ModelSettings:
    style = DEFAULT_STYLE_PRESET if color.warm_or_cool == "neutral" else f"{color.warm_or_cool}-{DEFAULT_STYLE_PRESET}"
    return ModelSettings(
        model_name=DEFAULT_MODEL_NAME, style_preset=style,
        negative_prompt_weight=DEFAULT_NEGATIVE_PROMPT_WEIGHT,
    )


def compile_prompt_package(spec: RedesignSpecification) -> PromptPackage:
    """Compile one usable Module 5 specification without external calls.

    ``duration_seconds`` and ``generated_at`` are observability metadata only;
    all prompt and generation fields are deterministic for identical input.
    """
    if spec.status == "error":
        raise InvalidRedesignSpecError("RedesignSpecification must have non-error status to compile a prompt package")

    started_at = time.monotonic()
    subject = _compile_subject_instructions(spec.subject_treatment)
    background = _compile_background_instructions(spec.layout_direction)
    typography = _compile_typography_instructions(spec.text_overlay)
    composition = _compile_composition_instructions(spec.layout_direction)
    lighting = _compile_lighting_instructions(spec.color_direction)
    color = _compile_color_instructions(spec.color_direction)
    package = PromptPackage(
        video_id=spec.video_id,
        positive_prompt=_compile_positive_prompt(
            subject, background, composition, lighting, color, typography,
            spec.elements_to_preserve,
        ),
        negative_prompt=_compile_negative_prompt(spec.object_directives),
        subject_instructions=subject, background_instructions=background,
        typography_instructions=typography, composition_instructions=composition,
        lighting_instructions=lighting, color_instructions=color,
        object_placement=_compile_object_placement(spec.object_directives),
        rendering_constraints=_compile_rendering_constraints(spec),
        safety_constraints=list(SAFETY_CONSTRAINTS),
        generation_parameters=_compile_generation_parameters(spec.video_id),
        quality_parameters=_compile_quality_parameters(spec.subject_treatment),
        model_settings=_compile_model_settings(spec.color_direction),
        duration_seconds=time.monotonic() - started_at,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info("Prompt package compiled for video_id={id} ({dur:.4f}s)", id=spec.video_id, dur=package.duration_seconds)
    return package


def _prompt_package_path(video_id: str, package_dir: Path) -> Path:
    return package_dir / PROMPT_PACKAGE_FILENAME_TEMPLATE.format(video_id=video_id)


def save_prompt_package(package: PromptPackage, package_dir: Path = DEFAULT_PROMPT_PACKAGE_DIR) -> None:
    """Persist a prompt package atomically."""
    target = _prompt_package_path(package.video_id, package_dir)
    tmp = target.with_suffix(".tmp")
    try:
        package_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(package.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.debug("Saved prompt package for video_id={id} -> {path}", id=package.video_id, path=target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        logger.error("Failed to save prompt package for video_id={id}: {exc}", id=package.video_id, exc=exc)
        raise PromptPackageCacheError(f"Could not write prompt package to {target}: {exc}") from exc


def load_cached_prompt_package(video_id: str, package_dir: Path = DEFAULT_PROMPT_PACKAGE_DIR) -> Optional[PromptPackage]:
    """Load a valid cached prompt package, or return ``None`` on a miss."""
    path = _prompt_package_path(video_id, package_dir)
    if not path.exists():
        logger.debug("Prompt package cache miss for video_id={id}", id=video_id)
        return None
    try:
        package = PromptPackage.model_validate_json(path.read_text(encoding="utf-8"))
        logger.debug("Prompt package cache hit for video_id={id}: {path}", id=video_id, path=path)
        return package
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Cached prompt package for video_id={id} is unreadable ({reason}) — treating as cache miss", id=video_id, reason=exc)
        return None

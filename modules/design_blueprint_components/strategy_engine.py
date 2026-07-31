"""Design Decision Strategy Engine for Module 5.5.

Derives design strategy fields (face_strategy, background_strategy, lighting,
color_palette, branding_constraints) from ThumbnailIntelligence, RedesignSpecification,
and VideoMetadata via rule tables.
"""

from __future__ import annotations

from typing import Literal

from config import SAFETY_CONSTRAINTS
from models import (
    RedesignSpecification,
    ThumbnailIntelligence,
)

FaceStrategy = Literal["smile", "neutral", "shock", "exaggerate", "remove", "preserve"]
BackgroundStrategy = Literal["keep", "replace", "blur", "darken", "simplify"]


def derive_face_strategy(
    intelligence: ThumbnailIntelligence,
    spec: RedesignSpecification,
    hook_type: str,
) -> FaceStrategy:
    """Derive face treatment strategy deterministically."""
    if not intelligence.faces.faces:
        return "neutral"

    face = intelligence.faces.faces[0]
    smile = face.smile_detected if hasattr(face, "smile_detected") and face.smile_detected is not None else False

    if hook_type == "shock":
        if spec.subject_treatment.crop_tighter:
            return "exaggerate"
        return "shock"

    if smile and hook_type != "shock":
        return "preserve"

    if spec.subject_treatment.crop_tighter:
        return "exaggerate"

    return "smile" if smile else "neutral"


def derive_background_strategy(
    intelligence: ThumbnailIntelligence,
    spec: RedesignSpecification,
) -> BackgroundStrategy:
    """Derive background simplification and modification strategy."""
    comp = intelligence.composition
    color = intelligence.colors

    if comp.clutter_score > 0.6:
        return "simplify"
    if comp.negative_space_ratio < 0.25:
        return "blur"
    if color.brightness < 0.35:
        return "darken"
    return "keep"


def derive_lighting(intelligence: ThumbnailIntelligence, spec: RedesignSpecification) -> str:
    """Derive lighting phrase from brightness, contrast, and color direction."""
    b = spec.color_direction.target_brightness
    c = spec.color_direction.target_contrast
    temp = spec.color_direction.warm_or_cool

    if b >= 0.6 and c >= 0.6:
        phrase = "vivid high-contrast studio lighting"
    elif b <= 0.4 and c >= 0.6:
        phrase = "dramatic moody chiaroscuro lighting"
    elif b <= 0.4:
        phrase = "soft low-key ambient lighting"
    elif c <= 0.4:
        phrase = "even diffused fill lighting"
    else:
        phrase = "balanced natural studio lighting"

    if temp != "neutral":
        phrase += f" with {temp} color tone"
    return phrase


def derive_color_palette(spec: RedesignSpecification) -> list[str]:
    """Derive color palette listing for prompt compiler convenience."""
    palette: list[str] = []
    if spec.color_direction.warm_or_cool != "neutral":
        palette.append(f"{spec.color_direction.warm_or_cool} tones")
    palette.append(f"brightness={spec.color_direction.target_brightness:.2f}")
    palette.append(f"contrast={spec.color_direction.target_contrast:.2f}")
    palette.append(f"saturation={spec.color_direction.target_saturation:.2f}")
    return palette


def derive_branding_constraints(spec: RedesignSpecification) -> list[str]:
    """Combine elements to preserve with safety constraints."""
    constraints: list[str] = list(spec.elements_to_preserve)
    for safety in SAFETY_CONSTRAINTS:
        if "watermark" in safety.lower() or "copyrighted" in safety.lower():
            constraints.append(safety)
    return constraints

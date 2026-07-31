"""Layout Planner subsystem for Module 5.5.

Arbitrates between face, text, object, and focal zones computed by Modules 4 and 5,
resolves overlapping bounding box conflicts, enforces safe margins, ranks visual priority,
and derives camera distance and object directives deterministically.
"""

from __future__ import annotations

from typing import Literal, Optional

from config import (
    DEFAULT_GENERATION_HEIGHT,
    DEFAULT_GENERATION_WIDTH,
    MODULE55_MAX_ZONE_OVERLAP,
    MODULE55_SAFE_MARGIN_RATIO,
)
from models import (
    BoundingBox,
    CanvasTransform,
    ObjectDirective,
    ObjectLayoutDirective,
    RedesignSpecification,
    TextPlacement,
    ThumbnailIntelligence,
    VisualBoundingBox,
)

CameraDistance = Literal["close_up", "medium", "wide"]


def calculate_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute Intersection-over-Union between two normalized bounding boxes."""
    x_left = max(b1.x_min, b2.x_min)
    y_top = max(b1.y_min, b2.y_min)
    x_right = min(b1.x_max, b2.x_max)
    y_bottom = min(b1.y_max, b2.y_max)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    b1_area = (b1.x_max - b1.x_min) * (b1.y_max - b1.y_min)
    b2_area = (b2.x_max - b2.x_min) * (b2.y_max - b2.y_min)

    union_area = b1_area + b2_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def apply_safe_margin(bbox: BoundingBox, margin_ratio: float = MODULE55_SAFE_MARGIN_RATIO) -> BoundingBox:
    """Clamp bbox to safe margin inset from frame edges."""
    return BoundingBox(
        x_min=max(margin_ratio, min(1.0 - margin_ratio, bbox.x_min)),
        y_min=max(margin_ratio, min(1.0 - margin_ratio, bbox.y_min)),
        x_max=max(margin_ratio, min(1.0 - margin_ratio, bbox.x_max)),
        y_max=max(margin_ratio, min(1.0 - margin_ratio, bbox.y_max)),
    )


def bbox_to_pixel(
    bbox: Optional[BoundingBox],
    width: int = DEFAULT_GENERATION_WIDTH,
    height: int = DEFAULT_GENERATION_HEIGHT,
) -> Optional[VisualBoundingBox]:
    """Convert normalized BoundingBox to VisualBoundingBox in pixel space."""
    if bbox is None:
        return None
    x = int(round(bbox.x_min * width))
    y = int(round(bbox.y_min * height))
    w = max(1, int(round((bbox.x_max - bbox.x_min) * width)))
    h = max(1, int(round((bbox.y_max - bbox.y_min) * height)))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = min(width - x, w)
    h = min(height - y, h)
    return VisualBoundingBox(x=x, y=y, width=w, height=h)


def derive_camera_distance(
    subject_bbox: Optional[BoundingBox],
    crop_tighter: bool,
) -> CameraDistance:
    """Derive camera distance from resolved subject area and crop requirement."""
    if subject_bbox is None:
        return "medium"

    area = (subject_bbox.x_max - subject_bbox.x_min) * (subject_bbox.y_max - subject_bbox.y_min)
    if area >= 0.35:
        return "wide"
    if area <= 0.15 or crop_tighter:
        return "close_up"
    return "medium"


def resolve_layout(
    intelligence: ThumbnailIntelligence,
    spec: RedesignSpecification,
) -> tuple[
    TextPlacement,
    Optional[BoundingBox],
    list[ObjectLayoutDirective],
    CameraDistance,
    list[str],
    int,
]:
    """Perform pairwise conflict resolution across all layout elements.

    Returns:
        (text_position, subject_position, object_strategy, camera_distance, visual_priority, conflicts_resolved)
    """
    conflicts_count = 0

    # 1. Subject zone
    subject_bbox: Optional[BoundingBox] = None
    if spec.subject_treatment.has_subject and spec.subject_treatment.target_bbox:
        subject_bbox = apply_safe_margin(spec.subject_treatment.target_bbox)

    # 2. Text placement zone
    include_text = spec.text_overlay.include_text
    raw_text_bbox = spec.text_overlay.placement_zone

    if raw_text_bbox is None and include_text:
        # Default top-left or top-third text zone if requested but absent
        raw_text_bbox = BoundingBox(x_min=0.05, y_min=0.05, x_max=0.55, y_max=0.35)

    text_bbox = apply_safe_margin(raw_text_bbox) if raw_text_bbox else None

    # Quadrants for shifting lower precedence elements
    quadrants = [
        BoundingBox(x_min=0.05, y_min=0.05, x_max=0.55, y_max=0.35),  # top-left
        BoundingBox(x_min=0.45, y_min=0.05, x_max=0.95, y_max=0.35),  # top-right
        BoundingBox(x_min=0.05, y_min=0.55, x_max=0.55, y_max=0.95),  # bottom-left
        BoundingBox(x_min=0.45, y_min=0.55, x_max=0.95, y_max=0.95),  # bottom-right
    ]

    # Check conflict between face and headline text
    if subject_bbox and text_bbox:
        iou = calculate_iou(subject_bbox, text_bbox)
        if iou > MODULE55_MAX_ZONE_OVERLAP:
            conflicts_count += 1
            # Face has higher precedence; shift text zone to nearest free quadrant
            for quad in quadrants:
                if calculate_iou(subject_bbox, quad) <= MODULE55_MAX_ZONE_OVERLAP:
                    text_bbox = quad
                    break

    # Convert text_bbox to pixel space TextPlacement
    avoid_zones_px: list[VisualBoundingBox] = []
    if subject_bbox:
        px_sub = bbox_to_pixel(subject_bbox)
        if px_sub:
            avoid_zones_px.append(px_sub)

    for avoid in spec.text_overlay.avoid_zones:
        px_avoid = bbox_to_pixel(avoid)
        if px_avoid:
            avoid_zones_px.append(px_avoid)

    text_position = TextPlacement(
        include_text=include_text,
        placement_zone_px=bbox_to_pixel(text_bbox) if include_text else None,
        avoid_zones_px=avoid_zones_px if include_text else [],
    )

    # 3. Object directives refinement
    object_strategy: list[ObjectLayoutDirective] = []
    for rank, obj_directive in enumerate(spec.object_directives, start=1):
        scale = 1.0
        # Check conflict with face or text
        matching_obj = next((o for o in intelligence.objects if o.label == obj_directive.label and o.bbox), None)
        if matching_obj and matching_obj.bbox:
            obj_bbox = apply_safe_margin(matching_obj.bbox)
            if subject_bbox and calculate_iou(subject_bbox, obj_bbox) > MODULE55_MAX_ZONE_OVERLAP:
                conflicts_count += 1
                scale = 0.7
            if text_bbox and calculate_iou(text_bbox, obj_bbox) > MODULE55_MAX_ZONE_OVERLAP:
                conflicts_count += 1
                scale = 0.8

        object_strategy.append(
            ObjectLayoutDirective(
                label=obj_directive.label,
                action=obj_directive.action,
                scale_factor=scale,
                emphasis_rank=rank,
                rationale=obj_directive.rationale or f"Rank {rank} object directive",
            )
        )

    # 4. Camera distance
    camera_distance = derive_camera_distance(
        subject_bbox,
        spec.subject_treatment.crop_tighter if spec.subject_treatment else False,
    )

    # 5. Visual priority ranking
    visual_priority: list[str] = []
    if include_text:
        visual_priority.append("headline")
    if subject_bbox:
        visual_priority.append("face")
    if any(o.action != "remove" for o in object_strategy):
        visual_priority.append("primary_object")
    visual_priority.append("background")

    return (
        text_position,
        subject_bbox,
        object_strategy,
        camera_distance,
        visual_priority,
        conflicts_count,
    )

"""Deterministic Module 5.5 Design Blueprint Engine.

Assembles Copywriter, Layout Planner, and Strategy Engine outputs into an
execution-ready DesignBlueprint artifact. Operates without any external network calls
or LLM reasoning.
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
    DEFAULT_DESIGN_BLUEPRINT_DIR,
    DESIGN_BLUEPRINT_FILENAME_TEMPLATE,
    LOG_DIR,
    MODULE55_LOG_PATH,
)
from design_blueprint_components.copywriter import author_headline_candidates
from design_blueprint_components.layout_planner import resolve_layout
from design_blueprint_components.strategy_engine import (
    derive_background_strategy,
    derive_branding_constraints,
    derive_color_palette,
    derive_face_strategy,
    derive_lighting,
)
from design_blueprint_exceptions import (
    DesignBlueprintCacheError,
    DesignBlueprintError,
    InvalidRedesignSpecError,
)
from models import (
    DesignBlueprint,
    RedesignSpecification,
    ThumbnailIntelligence,
    VideoMetadata,
)

_LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """Attach the Module 5.5 rotating Loguru sink."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE55_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


def build_design_blueprint(
    intelligence: ThumbnailIntelligence,
    redesign_spec: RedesignSpecification,
    metadata: VideoMetadata,
) -> DesignBlueprint:
    """Build a frozen DesignBlueprint deterministically from input artifacts."""
    if redesign_spec.status == "error":
        raise InvalidRedesignSpecError(
            f"RedesignSpecification for video_id={redesign_spec.video_id} has error status"
        )

    started_at = time.monotonic()
    partial_reasons: list[str] = []

    if not intelligence.faces.faces:
        partial_reasons.append("No face detected in source thumbnail; defaulted face_strategy to neutral.")

    # 1. Author and score headlines
    headline, score, hook_type, emotion, variants = author_headline_candidates(
        intelligence, redesign_spec, metadata
    )

    if len(variants) < 3:
        partial_reasons.append("Fewer than 3 headline variants were generated.")

    # 2. Resolve layout geometry and conflicts
    (
        text_position,
        subject_position,
        object_strategy,
        camera_distance,
        visual_priority,
        conflicts_resolved,
    ) = resolve_layout(intelligence, redesign_spec)

    # 3. Derive design strategies
    face_strat = derive_face_strategy(intelligence, redesign_spec, hook_type)
    bg_strat = derive_background_strategy(intelligence, redesign_spec)
    lighting_str = derive_lighting(intelligence, redesign_spec)
    palette = derive_color_palette(redesign_spec)
    branding = derive_branding_constraints(redesign_spec)

    status = "partial" if partial_reasons else "success"

    blueprint = DesignBlueprint(
        video_id=redesign_spec.video_id,
        headline=headline,
        headline_variants=variants,
        headline_score=score,
        hook_type=hook_type,
        emotion=emotion,
        face_strategy=face_strat,
        object_strategy=object_strategy,
        background_strategy=bg_strat,
        text_position=text_position,
        subject_position=subject_position,
        camera_distance=camera_distance,
        lighting=lighting_str,
        color_palette=palette,
        visual_priority=visual_priority,
        branding_constraints=branding,
        conflicts_resolved=conflicts_resolved,
        status=status,
        partial_failure_reasons=partial_reasons,
        duration_seconds=round(time.monotonic() - started_at, 4),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Design blueprint built video_id={vid} headline_score={score:.4f} conflicts={conflicts} status={status}",
        vid=blueprint.video_id,
        score=blueprint.headline_score,
        conflicts=blueprint.conflicts_resolved,
        status=blueprint.status,
    )

    return blueprint


def _design_blueprint_path(video_id: str, blueprint_dir: Path) -> Path:
    """Return canonical JSON path for a design blueprint."""
    return blueprint_dir / DESIGN_BLUEPRINT_FILENAME_TEMPLATE.format(video_id=video_id)


def save_design_blueprint(
    blueprint: DesignBlueprint,
    *,
    blueprint_dir: Path = DEFAULT_DESIGN_BLUEPRINT_DIR,
) -> Path:
    """Persist a DesignBlueprint atomically to disk."""
    target = _design_blueprint_path(blueprint.video_id, blueprint_dir)
    tmp = target.with_suffix(".tmp")
    try:
        blueprint_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.debug(
            "Saved design blueprint for video_id={id} -> {path}",
            id=blueprint.video_id,
            path=target,
        )
        return target
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        logger.error(
            "Failed to save design blueprint for video_id={id}: {exc}",
            id=blueprint.video_id,
            exc=exc,
        )
        raise DesignBlueprintCacheError(
            f"Could not write design blueprint to {target}: {exc}"
        ) from exc


def load_design_blueprint(
    video_id: str,
    *,
    blueprint_dir: Path = DEFAULT_DESIGN_BLUEPRINT_DIR,
) -> Optional[DesignBlueprint]:
    """Load cached DesignBlueprint, or return None on cache miss."""
    path = _design_blueprint_path(video_id, blueprint_dir)
    if not path.exists():
        logger.debug("Design blueprint cache miss for video_id={id}", id=video_id)
        return None
    try:
        blueprint = DesignBlueprint.model_validate_json(path.read_text(encoding="utf-8"))
        logger.debug("Design blueprint cache hit for video_id={id}: {path}", id=video_id, path=path)
        return blueprint
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Cached design blueprint for video_id={id} is unreadable ({reason}) — treating as cache miss",
            id=video_id,
            reason=exc,
        )
        return None

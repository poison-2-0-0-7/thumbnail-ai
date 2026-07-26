"""Configuration loading for AI Vision Stack V2.1.

The YAML file is the single source of truth for model checkpoints and
per-stage settings. This module validates that file but does not duplicate
checkpoint definitions in Python.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .exceptions import VisionStackConfigError
from .models import VisionStackConfig


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VISION_STACK_CONFIG_PATH: Path = PROJECT_ROOT / "vision_stack.yaml"
VISION_STACK_CONFIG_ENV: str = "THUMBNAIL_AI_VISION_STACK_CONFIG"
VISION_STACK_VERSION: str = "2.0.0"
VISION_STACK_MODEL_ORDER: tuple[str, ...] = (
    "grounding_dino",
    "florence2",
    "paddleocr",
    "openclip",
    "insightface",
    "bisenet",
    "birefnet",
    "sam2",
    "depth_anything",
    "teed",
)
VISION_STACK_STAGE_LATENCY_KEYS: tuple[str, ...] = (
    "grounding_dino",
    "florence2",
    "paddleocr",
    "openclip",
    "identity_engine",
    "birefnet",
    "sam2",
    "depth_teed",
    "cpu_heuristics",
)


def _read_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise VisionStackConfigError(f"Vision stack config file not found: {config_path}")
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VisionStackConfigError(f"Invalid YAML in vision stack config: {config_path}") from exc
    if not isinstance(raw_config, dict):
        raise VisionStackConfigError(f"Vision stack config must be a mapping: {config_path}")
    return raw_config


def load_vision_stack_config(config_path: Path | None = None) -> VisionStackConfig:
    """Load and validate the V2.1 root vision-stack YAML configuration."""
    env_path = os.getenv(VISION_STACK_CONFIG_ENV)
    resolved_path = Path(config_path or env_path or VISION_STACK_CONFIG_PATH)
    raw_config = _read_yaml_config(resolved_path)
    section = raw_config.get("vision_stack")
    if not isinstance(section, dict):
        raise VisionStackConfigError("Vision stack config must contain a mapping named 'vision_stack'")
    try:
        return VisionStackConfig.model_validate(section)
    except ValidationError as exc:
        raise VisionStackConfigError("Vision stack config failed schema validation") from exc

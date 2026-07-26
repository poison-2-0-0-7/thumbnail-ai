"""Boot-time loader for AI Vision Stack V2.1."""

from __future__ import annotations

from pathlib import Path

from .config import load_vision_stack_config
from .models import VisionStackConfig
from .registry import ModelRegistry


def load_config(config_path: Path | None = None) -> VisionStackConfig:
    """Load and validate the YAML root configuration."""
    return load_vision_stack_config(config_path)


def build_registry(config: VisionStackConfig | None = None) -> ModelRegistry:
    """Populate a registry from a validated YAML-derived configuration."""
    stack_config = config or load_config()
    registry = ModelRegistry()
    registry.register_stack(stack_config)
    return registry

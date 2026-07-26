"""Boot-time loader for AI Vision Stack V2.1.

The loader validates configuration-derived checkpoint metadata only. It never
imports AI runtimes, instantiates model wrappers, or loads weights.
"""

from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT, load_vision_stack_config
from .exceptions import VisionStackCheckpointError
from .models import (
    ModelLoadingMetadata,
    RuntimeBootstrapMetadata,
    VisionModelConfig,
    VisionStackConfig,
)
from .registry import ModelRegistry


DEFAULT_CHECKPOINT_ROOT: Path = PROJECT_ROOT / "models" / "vision_stack"

_AUXILIARY_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "grounding_dino": ("GroundingDINO_SwinT_OGC.py",),
    "sam2": ("sam2.1_hiera_l.yaml",),
    "insightface": ("det_10g.onnx", "w600k_r50.onnx"),
    "paddleocr": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}


def load_config(config_path: Path | None = None) -> VisionStackConfig:
    """Load and validate the YAML root configuration."""
    return load_vision_stack_config(config_path)


def build_registry(
    config: VisionStackConfig | None = None,
    metadata: RuntimeBootstrapMetadata | None = None,
) -> ModelRegistry:
    """Populate a registry from a validated YAML-derived configuration."""
    stack_config = config or load_config()
    registry = ModelRegistry()
    metadata_by_name = (
        {
            model_metadata.model_name: model_metadata.model_dump(mode="json")
            for model_metadata in metadata.models
        }
        if metadata
        else {}
    )
    for name, model_config in stack_config.model_items():
        registry.register(name, model_config, metadata=metadata_by_name.get(name))
    return registry


class ModelLoader:
    """Resolve and validate configured model checkpoint artifacts."""

    def __init__(self, checkpoint_root: Path | None = None) -> None:
        self.checkpoint_root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)

    def bootstrap_metadata(
        self,
        config: VisionStackConfig,
        *,
        validate_exists: bool = True,
    ) -> RuntimeBootstrapMetadata:
        """Return structured boot metadata for every configured model."""
        models = tuple(
            self.resolve_model(name, model_config, validate_exists=validate_exists)
            for name, model_config in config.model_items()
        )
        missing = {
            model.model_name: tuple(str(path) for path in model.missing_paths)
            for model in models
            if model.missing_paths
        }
        if missing:
            raise VisionStackCheckpointError(f"Missing checkpoint artifacts: {missing}")
        return RuntimeBootstrapMetadata(checkpoint_root=self.checkpoint_root, models=models)

    def resolve_model(
        self,
        name: str,
        model_config: VisionModelConfig,
        *,
        validate_exists: bool = True,
    ) -> ModelLoadingMetadata:
        """Resolve one model checkpoint and required sidecar files."""
        checkpoint_path = self._resolve_checkpoint_path(model_config.checkpoint)
        required_paths = self._required_paths(name, checkpoint_path)
        missing_paths = tuple(path for path in required_paths if validate_exists and not path.exists())
        return ModelLoadingMetadata(
            model_name=name,
            checkpoint_identifier=model_config.checkpoint,
            checkpoint_path=checkpoint_path,
            required_paths=required_paths,
            missing_paths=missing_paths,
            precision=model_config.precision,
            device=model_config.device,
            backend=model_config.backend,
            fallback=model_config.fallback,
            cache_enabled=model_config.cache_enabled,
            cpu_fallback_available=model_config.device == "cpu"
            or model_config.fallback.value.startswith("cpu"),
        )

    def _resolve_checkpoint_path(self, checkpoint: str) -> Path:
        candidate = Path(checkpoint)
        if candidate.is_absolute():
            return candidate
        return self.checkpoint_root / candidate

    def _required_paths(self, model_name: str, checkpoint_path: Path) -> tuple[Path, ...]:
        required = [checkpoint_path]
        required.extend(
            checkpoint_path.parent / filename
            for filename in _AUXILIARY_REQUIRED_FILES.get(model_name, ())
        )
        return tuple(dict.fromkeys(required))


def load_runtime_metadata(
    config: VisionStackConfig | None = None,
    *,
    checkpoint_root: Path | None = None,
    validate_exists: bool = True,
) -> RuntimeBootstrapMetadata:
    """Load config and produce validated runtime bootstrap metadata."""
    stack_config = config or load_config()
    return ModelLoader(checkpoint_root).bootstrap_metadata(
        stack_config,
        validate_exists=validate_exists,
    )

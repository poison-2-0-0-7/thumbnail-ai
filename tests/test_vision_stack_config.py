"""Tests for AI Vision Stack V2.1 configuration and model registry."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    VISION_STACK_CONFIG_ENV,
    VISION_STACK_CONFIG_PATH,
    load_vision_stack_config,
)
from vision_stack.exceptions import VisionStackConfigError, VisionStackRegistryError  # noqa: E402
from vision_stack.models import (  # noqa: E402
    VisionModelBackend,
    VisionModelFallback,
    VisionModelLifecycleState,
    VisionModelPrecision,
    VisionStackConfig,
)
from vision_stack.registry import ModelRegistry  # noqa: E402


def test_default_vision_stack_config_matches_v2_1_architecture() -> None:
    config = load_vision_stack_config()

    assert isinstance(config, VisionStackConfig)
    assert config.grounding_dino.checkpoint == "groundingdino_swint_ogc.pth"
    assert config.grounding_dino.backend == VisionModelBackend.PYTORCH
    assert config.grounding_dino.fallback == VisionModelFallback.SKIP_STAGE
    assert config.sam2.checkpoint == "sam2.1_hiera_large.pt"
    assert config.sam2.fallback == VisionModelFallback.CPU_TILED_PROCESSING
    assert config.openclip.backend == VisionModelBackend.OPEN_CLIP
    assert config.openclip.checkpoint == "ViT-B-32/laion2b_s34b_b79k"
    assert config.insightface.backend == VisionModelBackend.ONNXRUNTIME
    assert config.paddleocr.backend == VisionModelBackend.PADDLE
    assert config.paddleocr.fallback == VisionModelFallback.CPU_FALLBACK
    assert config.teed.checkpoint == "7_model.pth"
    assert all(model.precision == VisionModelPrecision.FP16 for _, model in config.model_items())
    assert all(model.device == "cuda:0" for _, model in config.model_items())
    assert all(model.batch_size == 1 for _, model in config.model_items())


def test_load_vision_stack_config_supports_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "vision_stack.yaml"
    config_path.write_text(
        """
vision_stack:
  grounding_dino:
    checkpoint: groundingdino_swint_ogc.pth
    precision: fp32
    device: cpu
    backend: pytorch
    batch_size: 1
    cache_enabled: true
    timeout: 5000
    fallback: skip_stage
  sam2:
    checkpoint: sam2.1_hiera_large.pt
    precision: fp16
    device: cuda:0
    backend: pytorch
    batch_size: 1
    cache_enabled: true
    timeout: 8000
    fallback: cpu_tiled_processing
  florence2:
    checkpoint: Florence-2-large
    precision: fp16
    device: cuda:0
    backend: transformers
    batch_size: 1
    cache_enabled: true
    timeout: 6000
    fallback: skip_stage
  openclip:
    checkpoint: ViT-B-32/laion2b_s34b_b79k
    precision: fp16
    device: cuda:0
    backend: open_clip
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: skip_stage
  insightface:
    checkpoint: buffalo_l
    precision: fp16
    device: cuda:0
    backend: onnxruntime
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: skip_stage
  birefnet:
    checkpoint: BiRefNet-general-epoch_244.pth
    precision: fp16
    device: cuda:0
    backend: pytorch
    batch_size: 1
    cache_enabled: true
    timeout: 5000
    fallback: skip_stage
  bisenet:
    checkpoint: 79999_iter.pth
    precision: fp16
    device: cuda:0
    backend: pytorch
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: skip_stage
  paddleocr:
    checkpoint: PP-OCRv5_server_det+PP-OCRv5_server_rec
    precision: fp16
    device: cuda:0
    backend: paddle
    batch_size: 1
    cache_enabled: true
    timeout: 4000
    fallback: cpu_fallback
  depth_anything:
    checkpoint: depth_anything_v2_vitb.pth
    precision: fp16
    device: cuda:0
    backend: pytorch
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: skip_stage
  teed:
    checkpoint: 7_model.pth
    precision: fp16
    device: cuda:0
    backend: pytorch
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: skip_stage
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(VISION_STACK_CONFIG_ENV, str(config_path))

    config = load_vision_stack_config()

    assert config.grounding_dino.device == "cpu"
    assert config.grounding_dino.precision == VisionModelPrecision.FP32


def test_root_vision_stack_yaml_validates_when_present() -> None:
    assert VISION_STACK_CONFIG_PATH.is_file()
    raw_config = yaml.safe_load(VISION_STACK_CONFIG_PATH.read_text(encoding="utf-8"))

    config = load_vision_stack_config(VISION_STACK_CONFIG_PATH)
    assert config.model_dump() == VisionStackConfig.model_validate(
        raw_config["vision_stack"]
    ).model_dump()


def test_load_vision_stack_config_requires_yaml_file(tmp_path: Path) -> None:
    with pytest.raises(VisionStackConfigError, match="not found"):
        load_vision_stack_config(tmp_path / "missing.yaml")


def test_vision_stack_config_rejects_non_sequential_batch_size() -> None:
    raw_root_config = yaml.safe_load(VISION_STACK_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_config = dict(raw_root_config["vision_stack"])
    grounding_dino = dict(raw_config["grounding_dino"])
    grounding_dino["batch_size"] = 2
    raw_config["grounding_dino"] = grounding_dino

    with pytest.raises(ValueError, match="batch_size=1"):
        VisionStackConfig.model_validate(raw_config)


def test_model_registry_registers_stack_and_tracks_lifecycle() -> None:
    registry = ModelRegistry()
    registered = registry.register_stack(load_vision_stack_config())

    assert len(registered) == 10
    assert registry.get("grounding_dino").lifecycle_state == VisionModelLifecycleState.REGISTERED
    assert registry.get("grounding_dino").metadata == {}
    assert registry.get("grounding_dino").runtime_state == {}
    registry.transition("grounding_dino", VisionModelLifecycleState.CPU_CACHED)
    registry.transition("grounding_dino", VisionModelLifecycleState.GPU_ACTIVE)
    updated = registry.transition("grounding_dino", VisionModelLifecycleState.CPU_CACHED)

    assert updated.lifecycle_state == VisionModelLifecycleState.CPU_CACHED


def test_model_registry_rejects_invalid_lifecycle_transition() -> None:
    registry = ModelRegistry()
    registry.register_stack(load_vision_stack_config())

    with pytest.raises(VisionStackRegistryError, match="Invalid lifecycle transition"):
        registry.transition("sam2", VisionModelLifecycleState.GPU_ACTIVE)

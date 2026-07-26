"""Typed contracts for AI Vision Stack V2.1 configuration and lifecycle."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VisionModelPrecision(str, Enum):
    """Supported inference precision modes for V2.1 vision-stack models."""

    FP16 = "fp16"
    FP32 = "fp32"


class VisionModelBackend(str, Enum):
    """Supported inference backends declared by the V2.1 architecture."""

    PYTORCH = "pytorch"
    ONNXRUNTIME = "onnxruntime"
    PADDLE = "paddle"
    TRANSFORMERS = "transformers"
    TENSORRT = "tensorrt"
    OPEN_CLIP = "open_clip"


class VisionModelFallback(str, Enum):
    """Per-stage fallback policies from the V2.1 configuration contract."""

    SKIP_STAGE = "skip_stage"
    CPU_FALLBACK = "cpu_fallback"
    CPU_TILED_PROCESSING = "cpu_tiled_processing"
    RETRY_ONCE = "retry_once"


class VisionModelLifecycleState(str, Enum):
    """Lifecycle states used by the V2.1 model registry."""

    REGISTERED = "registered"
    CPU_CACHED = "cpu_cached"
    GPU_ACTIVE = "gpu_active"
    EVICTED = "evicted"


class VisionModelConfig(BaseModel):
    """Validated per-model configuration loaded from YAML at worker boot."""

    model_config = ConfigDict(frozen=True)

    checkpoint: str
    precision: VisionModelPrecision
    device: str
    backend: VisionModelBackend
    batch_size: int
    cache_enabled: bool
    timeout: int
    fallback: VisionModelFallback

    @field_validator("checkpoint", "device")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("vision model text fields must not be empty")
        return value.strip()

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_one(cls, value: int) -> int:
        if value != 1:
            raise ValueError("vision stack V2.1 requires batch_size=1")
        return value

    @field_validator("timeout")
    @classmethod
    def timeout_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("vision model timeout must be positive milliseconds")
        return value


class VisionStackConfig(BaseModel):
    """Root V2.1 vision-stack configuration."""

    model_config = ConfigDict(frozen=True)

    grounding_dino: VisionModelConfig
    sam2: VisionModelConfig
    florence2: VisionModelConfig
    openclip: VisionModelConfig
    insightface: VisionModelConfig
    birefnet: VisionModelConfig
    bisenet: VisionModelConfig
    paddleocr: VisionModelConfig
    depth_anything: VisionModelConfig
    teed: VisionModelConfig

    def model_items(self) -> tuple[tuple[str, VisionModelConfig], ...]:
        """Return models in the architecture's sequential registration order."""
        return (
            ("grounding_dino", self.grounding_dino),
            ("florence2", self.florence2),
            ("paddleocr", self.paddleocr),
            ("openclip", self.openclip),
            ("insightface", self.insightface),
            ("bisenet", self.bisenet),
            ("birefnet", self.birefnet),
            ("sam2", self.sam2),
            ("depth_anything", self.depth_anything),
            ("teed", self.teed),
        )


class RegisteredVisionModel(BaseModel):
    """Registry entry for one configured V2.1 model."""

    model_config = ConfigDict(frozen=True)

    name: str
    config: VisionModelConfig
    lifecycle_state: VisionModelLifecycleState = VisionModelLifecycleState.REGISTERED
    metadata: dict[str, Any] = Field(default_factory=dict)
    runtime_state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("registered model name must not be empty")
        return value.strip()

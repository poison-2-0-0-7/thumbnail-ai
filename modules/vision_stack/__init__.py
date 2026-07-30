"""AI Vision Stack V2.1 bootstrap package."""

from .config import (
    VISION_STACK_CONFIG_ENV,
    VISION_STACK_CONFIG_PATH,
    VISION_STACK_MODEL_ORDER,
    VISION_STACK_STAGE_LATENCY_KEYS,
    VISION_STACK_VERSION,
    load_vision_stack_config,
)
from .exceptions import (
    VisionStackCheckpointError,
    VisionStackConfigError,
    VisionStackError,
    VisionStackLoaderError,
    VisionStackRegistryError,
    VisionStackResourceError,
    VisionStackRuntimeError,
)
from .grounding_dino import GroundingDINOWrapper
from .grounding_dino_exceptions import (
    GroundingDINOError,
    GroundingDINOInferenceError,
    GroundingDINOLoadError,
    GroundingDINOOutOfMemoryError,
    GroundingDINOParseError,
)
from .loader import DEFAULT_CHECKPOINT_ROOT, ModelLoader, build_registry, load_config, load_runtime_metadata
from .models import (
    GroundingDINODetection,
    ModelLoadingMetadata,
    PixelBoundingBox,
    RegisteredVisionModel,
    RuntimeBootstrapMetadata,
    VisionModelBackend,
    VisionModelConfig,
    VisionModelFallback,
    VisionModelLifecycleState,
    VisionModelPrecision,
    VisionStackConfig,
)
from .registry import ModelRegistry
from .resources import GPUResourceManager
from .runtime import RuntimeManager, VisionStackRuntime

from .openclip import OpenCLIPWrapper
from .openclip_exceptions import (
    OpenCLIPError,
    OpenCLIPInferenceError,
    OpenCLIPLoadError,
    OpenCLIPOutOfMemoryError,
    OpenCLIPParseError,
)

__all__ = [
    "DEFAULT_CHECKPOINT_ROOT",
    "GPUResourceManager",
    "GroundingDINODetection",
    "GroundingDINOError",
    "GroundingDINOInferenceError",
    "GroundingDINOLoadError",
    "GroundingDINOOutOfMemoryError",
    "GroundingDINOParseError",
    "GroundingDINOWrapper",
    "ModelLoader",
    "ModelLoadingMetadata",
    "ModelRegistry",
    "OpenCLIPError",
    "OpenCLIPInferenceError",
    "OpenCLIPLoadError",
    "OpenCLIPOutOfMemoryError",
    "OpenCLIPParseError",
    "OpenCLIPWrapper",
    "PixelBoundingBox",
    "RegisteredVisionModel",
    "RuntimeBootstrapMetadata",
    "RuntimeManager",
    "VISION_STACK_CONFIG_ENV",
    "VISION_STACK_CONFIG_PATH",
    "VISION_STACK_MODEL_ORDER",
    "VISION_STACK_STAGE_LATENCY_KEYS",
    "VISION_STACK_VERSION",
    "VisionModelBackend",
    "VisionModelConfig",
    "VisionModelFallback",
    "VisionModelLifecycleState",
    "VisionModelPrecision",
    "VisionStackConfig",
    "VisionStackCheckpointError",
    "VisionStackConfigError",
    "VisionStackError",
    "VisionStackLoaderError",
    "VisionStackRegistryError",
    "VisionStackResourceError",
    "VisionStackRuntimeError",
    "VisionStackRuntime",
    "build_registry",
    "load_config",
    "load_runtime_metadata",
    "load_vision_stack_config",
]


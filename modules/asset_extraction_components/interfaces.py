"""
interfaces.py
=============

Abstract Base Classes (ABCs) for Module 8 processors, bridge, writer, and manifest builder.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

from modules.models import (
    AssetExtractionManifest,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    TextRegion,
)


class IPersonProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, faces: FaceAnalysis
    ) -> list[dict[str, Any]]:
        """One dict per detected face: crops/masks as np.ndarray, embeddings
        as list[float], landmarks/pose as coordinate lists, all keyed by
        the same field names used in PersonAsset."""


class ISceneProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Keys: background, foreground, depth_map, segmentation_map,
        sky_mask, ground_mask. Missing keys mean that sub-asset could not
        be produced (model skipped/failed) — not an exception."""


class IObjectProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, objects: list[DetectedObject]
    ) -> list[dict[str, Any]]:
        """One dict per input DetectedObject: crop, mask (np.ndarray),
        plus parent_index/child_indices computed from bbox containment."""


class ITypographyProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, text_regions: list[TextRegion]
    ) -> list[dict[str, Any]]:
        """One dict per input TextRegion: crop, font/alignment/color
        estimates. Pure OpenCV — no model dependency."""


class IVisualPropertiesProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, colors: ColorProfile) -> dict[str, Any]:
        """Palette/gradient/lighting/blur/focus, seeded by Module 4's
        already-computed ColorProfile so brightness/contrast/saturation
        are never recomputed."""


class ICompositionAssetProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, composition: CompositionAnalysis
    ) -> dict[str, Any]:
        """Renders visual overlays (eye-flow map, negative-space mask)
        from Module 4's already-computed scores — never recomputes them."""


class IEffectsProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> dict[str, Any]:
        """Heuristic-only glow/outline/shadow/motion-blur/particle flags."""


class IModelBridge(ABC):
    @abstractmethod
    def run(self, model_name: str, operation: Callable[[Any], Any]) -> Any:
        """Reserve the shared GPU slot for model_name via vision_stack's
        GPUResourceManager, execute operation(model), release. Raises
        ModelUnavailableError if the checkpoint is missing/invalid and
        falls back per the model's configured VisionModelFallback policy."""


class IAssetExtractionWriter(ABC):
    @abstractmethod
    def write_image(self, array: np.ndarray, destination_path: Path) -> bool:
        ...

    @abstractmethod
    def write_json_sidecar(self, data: dict, destination_path: Path) -> bool:
        ...

    @abstractmethod
    def purge_directory(self, target_dir: Path) -> bool:
        ...


class IAssetManifestBuilder(ABC):
    @abstractmethod
    def build(self, **family_results) -> AssetExtractionManifest:
        ...

    @abstractmethod
    def serialize_to_disk(self, manifest: AssetExtractionManifest, path: Path) -> None:
        ...

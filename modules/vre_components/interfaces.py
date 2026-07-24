"""Abstract contracts for Visual Reference Engine components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models import VisualReferenceManifest


class IFaceProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
        """Detect a face, extract a crop, and build a binary mask."""


class ISegmentationProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decompose an image into foreground, background, object crop, and object mask."""


class ITopologyProcessor(ABC):
    @abstractmethod
    def generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """Compute a monocular depth-style conditioning map."""

    @abstractmethod
    def generate_canny_map(self, image: np.ndarray) -> np.ndarray:
        """Compute a Canny structural edge map."""


class IAssetWriter(ABC):
    @abstractmethod
    def write_image(self, array: np.ndarray, destination_path: Path) -> bool:
        """Atomically persist an image numpy array to disk."""

    @abstractmethod
    def purge_directory(self, target_dir: Path) -> bool:
        """Delete a VRE sharded directory and all generated contents."""


class IManifestBuilder(ABC):
    @abstractmethod
    def build(
        self,
        video_id: str,
        source_path: str,
        asset_paths: dict[str, str],
        metadata: dict[str, Any],
    ) -> VisualReferenceManifest:
        """Construct and validate the VisualReferenceManifest schema."""

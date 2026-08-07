"""Abstract base interfaces for Scene Decomposer components."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from ..schemas import Instance


class Detector(ABC):
    """Text-prompted instance detection+segmentation interface."""

    @abstractmethod
    def detect(self, image: np.ndarray, class_prompts: list[str]) -> list[Instance]:
        """Detect and segment instances matching class prompts in image.

        Args:
            image: HxWx3 uint8 RGB image array.
            class_prompts: List of text prompts (e.g. ["person", "logo", "product"]).

        Returns:
            List of detected Instance objects with binary masks and bounding boxes.
        """
        ...


class Matter(ABC):
    """Matting refinement interface for extracting soft alpha mattes."""

    @abstractmethod
    def refine(self, image: np.ndarray, instance: Instance) -> np.ndarray:
        """Refine instance hard binary mask into a soft alpha matte.

        Args:
            image: HxWx3 uint8 RGB image array.
            instance: Instance containing initial hard mask.

        Returns:
            HxW float32 alpha matte array with values in range [0.0, 1.0].
        """
        ...


class DepthEstimator(ABC):
    """Monocular depth estimation interface."""

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate monocular depth map for given image.

        Args:
            image: HxWx3 uint8 RGB image array.

        Returns:
            HxW float32 depth map array (normalized [0.0, 1.0], lower values = foreground).
        """
        ...

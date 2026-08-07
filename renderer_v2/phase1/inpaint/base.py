"""Abstract base interface for BackgroundInpainter."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BackgroundInpainter(ABC):
    """Background inpainting interface."""

    @abstractmethod
    def inpaint(self, image: np.ndarray, inverse_mask: np.ndarray, prompt: str) -> np.ndarray:
        """Synthesize background in masked regions.

        Args:
            image: HxWx3 RGB uint8 source image.
            inverse_mask: HxW float32 or uint8 binary mask where 1/True indicates region to inpaint.
            prompt: Text prompt guiding background synthesis.

        Returns:
            HxWx3 RGB uint8 full-frame image with background inpainted.
        """
        ...

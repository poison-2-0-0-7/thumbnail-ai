"""Depth-style and Canny topology map generation for Module 6.5 VRE."""

from __future__ import annotations

import cv2
import numpy as np

from config import VRE_CANNY_HIGH_THRESHOLD, VRE_CANNY_LOW_THRESHOLD
from vre_components.interfaces import ITopologyProcessor
from vre_exceptions import TopologyExtractionError


class TopologyProcessor(ITopologyProcessor):
    """Generate ControlNet-ready structural maps at source resolution."""

    def __init__(
        self,
        canny_low_threshold: int = VRE_CANNY_LOW_THRESHOLD,
        canny_high_threshold: int = VRE_CANNY_HIGH_THRESHOLD,
    ) -> None:
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold

    def generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        return self._apply_monocular_depth(image)

    def generate_canny_map(self, image: np.ndarray) -> np.ndarray:
        return self._apply_canny_edge_detection(image)

    def _apply_monocular_depth(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            depth = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            return depth.astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise TopologyExtractionError(f"Depth-map extraction failed: {exc}") from exc

    def _apply_canny_edge_detection(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(
                gray,
                threshold1=self.canny_low_threshold,
                threshold2=self.canny_high_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            raise TopologyExtractionError(f"Canny-map extraction failed: {exc}") from exc

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TopologyExtractionError("Topology extraction requires an HxWx3 image array")
        if image.size == 0:
            raise TopologyExtractionError("Topology extraction requires a non-empty image")

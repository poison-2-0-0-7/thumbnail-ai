"""Foreground/background decomposition for Module 6.5 VRE."""

from __future__ import annotations

import cv2
import numpy as np

from models import VisualBoundingBox
from vre_components.interfaces import ISegmentationProcessor
from vre_exceptions import SegmentationInferenceError


class SegmentationProcessor(ISegmentationProcessor):
    """Produce deterministic foreground, background, object crop, and mask assets."""

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._validate_image(image)
        alpha_matte = self._extract_matte(image)
        bbox = self._locate_salient_object(image, alpha_matte)

        foreground = np.zeros_like(image)
        foreground[alpha_matte > 0] = image[alpha_matte > 0]
        background = image.copy()
        background[alpha_matte > 0] = 0
        object_crop = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width].copy()
        object_mask = alpha_matte
        return foreground, background, object_crop, object_mask

    def _extract_matte(self, image: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            inverse = cv2.bitwise_not(otsu)
            mask = self._choose_central_matte(otsu, inverse)
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if int(np.count_nonzero(mask)) == 0:
                mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return mask
        except Exception as exc:  # noqa: BLE001 - CV backend failures are explicit
            raise SegmentationInferenceError(f"Segmentation matte extraction failed: {exc}") from exc

    def _locate_salient_object(
        self, image: np.ndarray, alpha_matte: np.ndarray
    ) -> VisualBoundingBox:
        if alpha_matte.shape != image.shape[:2]:
            raise SegmentationInferenceError("Segmentation matte shape does not match image")
        points = cv2.findNonZero(alpha_matte)
        height, width = image.shape[:2]
        if points is None:
            return VisualBoundingBox(x=0, y=0, width=width, height=height)
        x, y, w, h = cv2.boundingRect(points)
        return VisualBoundingBox(
            x=max(0, int(x)),
            y=max(0, int(y)),
            width=max(1, min(int(w), width - int(x))),
            height=max(1, min(int(h), height - int(y))),
        )

    @staticmethod
    def _choose_central_matte(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        height, width = first.shape[:2]
        y1, y2 = height // 4, height - height // 4
        x1, x2 = width // 4, width - width // 4
        first_score = int(np.count_nonzero(first[y1:y2, x1:x2]))
        second_score = int(np.count_nonzero(second[y1:y2, x1:x2]))
        return first if first_score >= second_score else second

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise SegmentationInferenceError("Segmentation requires an HxWx3 image array")
        if image.size == 0:
            raise SegmentationInferenceError("Segmentation requires a non-empty image")

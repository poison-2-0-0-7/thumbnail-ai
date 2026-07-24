"""Creator-face extraction and mask generation for Module 6.5 VRE."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from config import VRE_FACE_DETECTION_CONFIDENCE
from models import VisualBoundingBox
from vre_components.interfaces import IFaceProcessor
from vre_exceptions import FaceDetectionFailedWarning


class FaceProcessor(IFaceProcessor):
    """Detect the largest face, return its crop and a binary full-frame mask."""

    def __init__(
        self,
        confidence_threshold: float = VRE_FACE_DETECTION_CONFIDENCE,
        cascade_path: Path | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        haarcascades = getattr(getattr(cv2, "data", object()), "haarcascades", "")
        default_path = Path(haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade_path = Path(cascade_path) if cascade_path else default_path

    def process(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
        bbox = self._detect_face_bounding_box(image)
        if bbox is None:
            warnings.warn("No creator face detected", FaceDetectionFailedWarning, stacklevel=2)
            logger.warning("VRE face detection produced no usable creator face")
            return None, None, {"face_detected": False, "confidence": None}

        crop = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width].copy()
        mask = self._generate_alpha_mask(image, bbox)
        logger.debug(
            "VRE face bbox x={x}, y={y}, width={w}, height={h}",
            x=bbox.x,
            y=bbox.y,
            w=bbox.width,
            h=bbox.height,
        )
        return crop, mask, {
            "face_detected": True,
            "confidence": self.confidence_threshold,
            "bbox": bbox.model_dump(),
        }

    def _detect_face_bounding_box(self, image: np.ndarray) -> Optional[VisualBoundingBox]:
        if not _is_color_image(image):
            return None
        cascade_factory = getattr(cv2, "CascadeClassifier", None)
        if cascade_factory is None:
            logger.warning("OpenCV CascadeClassifier is unavailable in this runtime")
            return None
        cascade = cascade_factory(str(self.cascade_path))
        if cascade.empty():
            logger.warning("VRE face cascade unavailable at {path}", path=self.cascade_path)
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(detections) == 0:
            return None

        height, width = image.shape[:2]
        x, y, w, h = max(detections, key=lambda item: int(item[2]) * int(item[3]))
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        w = max(1, min(int(w), width - x))
        h = max(1, min(int(h), height - y))
        return VisualBoundingBox(x=x, y=y, width=w, height=h)

    def _generate_alpha_mask(self, image: np.ndarray, bbox: VisualBoundingBox) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = (bbox.x + bbox.width // 2, bbox.y + bbox.height // 2)
        axes = (max(1, bbox.width // 2), max(1, bbox.height // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
        return mask


def _is_color_image(image: np.ndarray) -> bool:
    return isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3

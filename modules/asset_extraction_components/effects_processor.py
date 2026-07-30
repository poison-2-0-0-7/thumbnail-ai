"""
effects_processor.py
====================

Derives visual effects flags (glow, outline, drop shadow, motion blur, particles)
using classical-CV heuristics. Pure OpenCV — zero ML model dependency.
"""

from typing import Any

import cv2
import numpy as np

from modules.config import ASSET_EFFECTS_MIN_CONFIDENCE_TO_FLAG
from modules.asset_extraction_components.interfaces import IEffectsProcessor


class EffectsProcessor(IEffectsProcessor):
    """Detects classical visual effects heuristically from raw image."""

    def process(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {
                "glow_detected": False,
                "outline_detected": False,
                "drop_shadow_detected": False,
                "motion_blur_detected": False,
                "particles_detected": False,
                "confidence": 0.0,
                "notes": ["Empty or invalid image"],
            }

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        glow_detected, glow_conf = self._detect_glow(image, gray)
        outline_detected, outline_conf = self._detect_outline(gray)
        drop_shadow_detected, shadow_conf = self._detect_drop_shadow(gray)
        motion_blur_detected, blur_conf = self._detect_motion_blur(gray)
        particles_detected, particle_conf = self._detect_particles(gray)

        overall_conf = float(
            np.mean([glow_conf, outline_conf, shadow_conf, blur_conf, particle_conf])
        )

        notes: list[str] = []
        if glow_detected:
            notes.append("Luminance halo glow detected near high-contrast edges")
        if outline_detected:
            notes.append("High-contrast stroke/outline ring detected")
        if drop_shadow_detected:
            notes.append("Directional offset dark shadow blob detected")
        if motion_blur_detected:
            notes.append("Directional motion blur gradient detected")
        if particles_detected:
            notes.append("Isolated particle/sparkle high-contrast blobs detected")

        return {
            "glow_detected": bool(glow_detected),
            "outline_detected": bool(outline_detected),
            "drop_shadow_detected": bool(drop_shadow_detected),
            "motion_blur_detected": bool(motion_blur_detected),
            "particles_detected": bool(particles_detected),
            "confidence": round(overall_conf, 4),
            "notes": notes,
        }

    @staticmethod
    def _detect_glow(image: np.ndarray, gray: np.ndarray) -> tuple[bool, float]:
        """Detect neon/halo glow around edges."""
        if image.ndim != 3:
            return False, 0.1

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        high_val_high_sat = (val > 220) & (sat > 150)
        ratio = float(np.count_nonzero(high_val_high_sat)) / float(gray.size)

        detected = ratio > 0.015
        conf = float(min(1.0, ratio * 20.0))
        return detected, conf

    @staticmethod
    def _detect_outline(gray: np.ndarray) -> tuple[bool, float]:
        """Detect stroke outline rings around edges."""
        edges = cv2.Canny(gray, 80, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel)
        ring_diff = cv2.subtract(dilated, edges)

        ratio = float(np.count_nonzero(ring_diff)) / float(gray.size)
        detected = ratio > 0.05
        conf = float(min(1.0, ratio * 10.0))
        return detected, conf

    @staticmethod
    def _detect_drop_shadow(gray: np.ndarray) -> tuple[bool, float]:
        """Detect offset dark shadow blobs near bright features."""
        dark_mask = (gray < 40).astype(np.uint8) * 255
        bright_mask = (gray > 200).astype(np.uint8) * 255

        # Check for dark region adjacent to bright region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated_bright = cv2.dilate(bright_mask, kernel)
        shadow_overlap = cv2.bitwise_and(dark_mask, dilated_bright)

        ratio = float(np.count_nonzero(shadow_overlap)) / float(gray.size)
        detected = ratio > 0.03
        conf = float(min(1.0, ratio * 15.0))
        return detected, conf

    @staticmethod
    def _detect_motion_blur(gray: np.ndarray) -> tuple[bool, float]:
        """Detect directional motion blur via Sobel ratio anisotropy."""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        var_x = float(np.var(sobelx))
        var_y = float(np.var(sobely))

        if var_x + var_y < 10.0:
            return False, 0.0

        ratio = max(var_x, var_y) / (min(var_x, var_y) + 1e-5)
        detected = ratio > 3.5
        conf = float(min(1.0, (ratio - 1.0) / 5.0))
        return detected, conf

    @staticmethod
    def _detect_particles(gray: np.ndarray) -> tuple[bool, float]:
        """Detect isolated small high-contrast sparkle/particle blobs."""
        h, w = gray.shape[:2]
        bright = (gray > 230).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)

        particle_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 2 <= area <= 50:  # Small particle size range
                particle_count += 1

        detected = particle_count >= 8
        conf = float(min(1.0, particle_count / 20.0))
        return detected, conf

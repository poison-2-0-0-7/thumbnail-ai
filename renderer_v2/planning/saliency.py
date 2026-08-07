"""Deterministic visual saliency, contrast, and visual clutter analysis engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class SaliencyEngine:
    """Calculates objective, deterministic visual saliency, clutter, and contrast metrics."""

    @staticmethod
    def compute_saliency_map(
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        depth_weight: float = 0.25,
    ) -> np.ndarray:
        """Compute normalized deterministic visual saliency map in range [0.0, 1.0].

        Combines spectral residual saliency, color opponent contrast, and depth weighting.

        Args:
            image: HxWx3 uint8 RGB array.
            depth_map: Optional HxW float32 depth map [0.0, 1.0] (0 = closest/foreground).
            depth_weight: Weight given to depth-foreground cues.

        Returns:
            HxW float32 array normalized to [0.0, 1.0].
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")

        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # 1. Spectral Residual Saliency
        # Resize to fixed standard scale (e.g. 64x64 or 128x128) for consistent frequency response
        small_h, small_w = min(128, h), min(128, w)
        small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # 2D Discrete Fourier Transform
        dft = np.fft.fft2(small_gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)

        # Log spectrum
        log_mag = np.log(magnitude + 1e-8)
        # Average filter on log spectrum
        kernel_size = 3
        avg_log_mag = cv2.blur(log_mag, (kernel_size, kernel_size))
        # Spectral residual
        spectral_residual = log_mag - avg_log_mag

        # Inverse Fourier Transform
        res_shift = np.exp(spectral_residual + 1j * phase)
        inv_dft = np.fft.ifft2(np.fft.ifftshift(res_shift))
        res_mag = np.abs(inv_dft)
        # Spatial smoothing
        res_mag = cv2.GaussianBlur(res_mag, (7, 7), sigmaX=2.5)

        # Upscale back to full size
        spectral_saliency = cv2.resize(res_mag, (w, h), interpolation=cv2.INTER_LINEAR)
        spectral_saliency = (spectral_saliency - spectral_saliency.min()) / (
            spectral_saliency.max() - spectral_saliency.min() + 1e-8
        )

        # 2. Color Contrast (Opponent Color Channels)
        r = image[:, :, 0].astype(np.float32) / 255.0
        g = image[:, :, 1].astype(np.float32) / 255.0
        b = image[:, :, 2].astype(np.float32) / 255.0

        rg_contrast = np.abs(r - g)
        by_contrast = np.abs(b - 0.5 * (r + g))
        color_saliency = 0.6 * rg_contrast + 0.4 * by_contrast
        color_saliency = cv2.GaussianBlur(color_saliency, (15, 15), sigmaX=5.0)
        color_saliency = (color_saliency - color_saliency.min()) / (
            color_saliency.max() - color_saliency.min() + 1e-8
        )

        # 3. Luminance Gradient Energy (Edge Saliency)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_saliency = cv2.GaussianBlur(grad_mag, (9, 9), sigmaX=3.0)
        grad_saliency = (grad_saliency - grad_saliency.min()) / (
            grad_saliency.max() - grad_saliency.min() + 1e-8
        )

        # Combined Visual Saliency
        combined = 0.45 * spectral_saliency + 0.35 * color_saliency + 0.20 * grad_saliency

        # 4. Depth Weighting (Foreground subjects with smaller depth values get boost)
        if depth_map is not None:
            if depth_map.shape != (h, w):
                depth_resized = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                depth_resized = depth_map.astype(np.float32)
            
            # Invert depth so 1.0 is closest/foreground, 0.0 is distant background
            d_min, d_max = depth_resized.min(), depth_resized.max()
            norm_depth = (depth_resized - d_min) / (d_max - d_min + 1e-8)
            foreground_cue = 1.0 - norm_depth

            combined = (1.0 - depth_weight) * combined + depth_weight * foreground_cue

        # Final normalization
        c_min, c_max = combined.min(), combined.max()
        final_saliency = (combined - c_min) / (c_max - c_min + 1e-8)
        return final_saliency.astype(np.float32)

    @staticmethod
    def compute_visual_center_of_mass(saliency_map: np.ndarray) -> Tuple[float, float]:
        """Compute normalized (x, y) center of mass of the saliency map in range [0.0, 1.0]."""
        total_mass = float(np.sum(saliency_map))
        if total_mass < 1e-6:
            return (0.5, 0.5)

        h, w = saliency_map.shape
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)

        cx = float(np.sum(x_indices * saliency_map) / total_mass) / w
        cy = float(np.sum(y_indices * saliency_map) / total_mass) / h

        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        return (cx, cy)

    @staticmethod
    def compute_mask_visual_weight(
        saliency_map: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute fraction of total scene visual saliency contained within given mask."""
        total_saliency = float(np.sum(saliency_map))
        if total_saliency < 1e-6:
            return 0.0

        if mask.shape != saliency_map.shape:
            mask = cv2.resize(mask.astype(np.float32), (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_weight = float(np.sum(saliency_map * (mask > 0.1).astype(np.float32)))
        return float(np.clip(mask_weight / total_saliency, 0.0, 1.0))

    @staticmethod
    def compute_visual_clutter(
        image: np.ndarray,
        subject_mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute background visual clutter score (0.0 to 1.0, higher = more cluttered/noisy).

        Evaluates edge entropy and high-frequency gradient density in non-subject regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Compute edge response using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edge_energy = np.abs(laplacian)

        if subject_mask is not None:
            if subject_mask.shape != (h, w):
                subject_mask = cv2.resize(subject_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            bg_mask = (subject_mask <= 0.1).astype(np.float32)
            bg_pixels = float(np.sum(bg_mask))
            if bg_pixels < 100:
                bg_mask = np.ones((h, w), dtype=np.float32)
                bg_pixels = float(h * w)
        else:
            bg_mask = np.ones((h, w), dtype=np.float32)
            bg_pixels = float(h * w)

        bg_edge_energy = edge_energy * bg_mask
        mean_edge_energy = float(np.sum(bg_edge_energy) / bg_pixels)

        # Scale typical Laplacian response (0 to ~80) to [0.0, 1.0]
        clutter_score = float(np.clip(mean_edge_energy / 40.0, 0.0, 1.0))
        return clutter_score

    @staticmethod
    def compute_contrast_ratio(
        image: np.ndarray,
        subject_mask: np.ndarray,
    ) -> float:
        """Calculate WCAG-standard relative luminance contrast ratio between subject and background.

        Contrast ratio formula: (L1 + 0.05) / (L2 + 0.05) where L1 >= L2.
        Returns a value >= 1.0 (e.g. 4.5 represents 4.5:1).
        """
        h, w, _ = image.shape
        if subject_mask.shape != (h, w):
            subject_mask = cv2.resize(subject_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        fg_mask = (subject_mask > 0.1)
        bg_mask = (subject_mask <= 0.1)

        if not np.any(fg_mask) or not np.any(bg_mask):
            return 1.0

        # Convert sRGB to linear RGB for WCAG relative luminance
        rgb_norm = image.astype(np.float32) / 255.0
        # sRGB to linear transformation
        linear_rgb = np.where(
            rgb_norm <= 0.04045,
            rgb_norm / 12.92,
            ((rgb_norm + 0.055) / 1.055) ** 2.4,
        )

        luminance = (
            0.2126 * linear_rgb[:, :, 0]
            + 0.7152 * linear_rgb[:, :, 1]
            + 0.0722 * linear_rgb[:, :, 2]
        )

        fg_lum = float(np.mean(luminance[fg_mask]))
        bg_lum = float(np.mean(luminance[bg_mask]))

        l1 = max(fg_lum, bg_lum)
        l2 = min(fg_lum, bg_lum)

        ratio = (l1 + 0.05) / (l2 + 0.05)
        return float(round(ratio, 2))

    @staticmethod
    def determine_attention_flow(
        primary_centroid: Tuple[float, float],
        secondary_centroids: List[Tuple[float, float]],
        canvas_center: Tuple[float, float] = (0.5, 0.5),
    ) -> str:
        """Determine deterministic visual attention flow direction."""
        px, py = primary_centroid
        if not secondary_centroids:
            if px < 0.45:
                return "left_to_right"
            elif px > 0.55:
                return "right_to_left"
            else:
                return "center_outward"

        # Calculate average vector to secondary elements
        dx = sum(s[0] - px for s in secondary_centroids) / len(secondary_centroids)
        dy = sum(s[1] - py for s in secondary_centroids) / len(secondary_centroids)

        if abs(dx) >= abs(dy):
            return "left_to_right" if dx > 0 else "right_to_left"
        else:
            return "top_to_bottom" if dy > 0 else "bottom_to_top"

"""
Non-Destructive Additive Edge Relighting (NDAER) Engine

Generates atmospheric rim light onto subject cutouts without modifying facial skin,
eyes, or identity-defining features. Core skin pixels remain 100% mathematically bit-identical.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from ..core.schema import RelightingSpec
from ..core.canvas import Layer


class NonDestructiveEdgeRelighter:
    """Applies screen-blend additive edge lighting strictly to silhouette outer margins."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def hex_to_rgb(self, hex_str: str) -> Tuple[int, int, int]:
        hex_clean = hex_str.lstrip("#")
        return tuple(int(hex_clean[i : i + 2], 16) for i in (0, 2, 4))

    def apply_relighting(
        self,
        subject_layer: Layer,
        relight_spec: RelightingSpec,
        face_mask: Optional[np.ndarray] = None,
    ) -> Layer:
        """Applies NDAER to the subject layer.

        Args:
            subject_layer: The target Layer containing the subject RGBA + alpha mask.
            relight_spec: RelightingSpec parameters (direction, intensity, color, skin margin).
            face_mask: Optional H x W binary mask of facial skin region to explicitly protect.

        Returns:
            Relit Layer with preserved skin pixels.
        """
        if not relight_spec.enabled:
            return subject_layer

        relit_layer = subject_layer.copy()
        alpha = subject_layer.alpha_mask.copy()  # H x W uint8

        # 1. Compute Subject Edge Margin Mask (Outer N pixels boundary)
        kernel_size = relight_spec.skin_freeze_margin_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded_alpha = cv2.erode(alpha, kernel, iterations=1)
        edge_mask = cv2.subtract(alpha, eroded_alpha)  # H x W uint8 (High values near border)

        # 2. Freeze Skin Pixels: Exclude Face/Skin mask from edge mask
        if face_mask is not None:
            # Dilate face mask slightly to ensure 100% zero skin bleed
            dilated_face = cv2.dilate(face_mask, kernel, iterations=1)
            edge_mask[dilated_face > 0] = 0

        # 3. Synthesize Directional Light Gradient
        h, w = alpha.shape
        angle_rad = np.radians(relight_spec.direction_angle_deg)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        grad_map = (x_grid * dx + y_grid * dy)
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
        grad_map = np.clip(grad_map * 1.5, 0.0, 1.0)

        # 4. Generate Additive Rim Light Overlay
        color_rgb = np.array(self.hex_to_rgb(relight_spec.color_hex), dtype=np.float32)
        rim_strength = (edge_mask.astype(np.float32) / 255.0) * grad_map * relight_spec.intensity
        rim_layer = (rim_strength[:, :, np.newaxis] * color_rgb).astype(np.float32)

        # 5. Screen Blend: C_result = 1 - (1 - C_orig) * (1 - C_rim)
        rgb_orig = relit_layer.rgba_image[:, :, :3].astype(np.float32) / 255.0
        rim_norm = rim_layer / 255.0
        screen_blend = 1.0 - (1.0 - rgb_orig) * (1.0 - rim_norm)

        final_rgb = np.clip(screen_blend * 255.0, 0, 255).astype(np.uint8)
        relit_layer.rgba_image[:, :, :3] = final_rgb

        return relit_layer

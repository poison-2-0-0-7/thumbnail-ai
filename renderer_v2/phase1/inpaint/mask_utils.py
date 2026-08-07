"""Utility functions for locked region mask building, dilation, and inversion."""

from __future__ import annotations

from typing import List
import cv2
import numpy as np

from ..schemas import Instance, SceneGraph


def build_locked_region_mask(scene_graph: SceneGraph) -> np.ndarray:
    """Build union of all locked instance alpha mattes.

    Args:
        scene_graph: Input SceneGraph containing instances.

    Returns:
        HxW float32 array in range [0.0, 1.0] representing locked foreground pixels.
    """
    h, w = scene_graph.height, scene_graph.width
    combined = np.zeros((h, w), dtype=np.float32)

    for inst in scene_graph.get_locked_instances():
        if inst.alpha_matte is not None:
            combined = np.maximum(combined, inst.alpha_matte)
        elif inst.mask is not None:
            combined = np.maximum(combined, inst.mask.astype(np.float32))

    return np.clip(combined, 0.0, 1.0)


def build_inpaint_inverse_mask(
    locked_region_mask: np.ndarray,
    dilation_px: int = 12,
) -> np.ndarray:
    """Compute inverted inpainting mask (where 1/True = background to inpaint).

    The locked foreground mask is thresholded and dilated slightly (~12px) to prevent
    inpainting artifacts bleeding right up to the subject edge.

    Args:
        locked_region_mask: HxW float32 mask of locked foreground regions [0.0, 1.0].
        dilation_px: Radius in pixels for binary dilation of foreground.

    Returns:
        HxW uint8 binary mask where 255/1 indicates the background region to inpaint.
    """
    binary_fg = (locked_region_mask > 0.1).astype(np.uint8)

    if dilation_px > 0:
        kernel_size = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_fg = cv2.dilate(binary_fg, kernel, iterations=1)
    else:
        dilated_fg = binary_fg

    # Inverse mask: 1 where background, 0 where locked subject
    inverse_bg = (1 - dilated_fg) * 255
    return inverse_bg.astype(np.uint8)

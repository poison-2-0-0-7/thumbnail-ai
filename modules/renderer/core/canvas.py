"""
Rendering Engine Multi-Layer RGBA Canvas & Layer Data Structures

Provides high-performance multi-layer canvas abstractions for layer isolation,
alpha matting, z-ordering, and layer compositions.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .schema import LayerType, LayerAction


class Layer:
    """Represents an isolated 2D graphical layer with continuous 8-bit alpha matte."""

    def __init__(
        self,
        layer_id: str,
        layer_type: LayerType,
        rgba_image: np.ndarray,  # H x W x 4 Uint8
        alpha_mask: np.ndarray,  # H x W Uint8 (0-255)
        z_index: int,
        bounding_box: Tuple[int, int, int, int],  # (x_min, y_min, x_max, y_max)
        action: LayerAction = LayerAction.PRESERVE,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
            raise ValueError(f"rgba_image must be HxWx4 uint8 array, got shape {rgba_image.shape}")
        if alpha_mask.ndim != 2:
            raise ValueError(f"alpha_mask must be HxW uint8 array, got shape {alpha_mask.shape}")

        self.layer_id = layer_id
        self.layer_type = layer_type
        self.rgba_image = rgba_image.astype(np.uint8)
        self.alpha_mask = alpha_mask.astype(np.uint8)
        self.z_index = z_index
        self.bounding_box = bounding_box
        self.action = action
        self.metadata = metadata or {}

    @property
    def height(self) -> int:
        return self.rgba_image.shape[0]

    @property
    def width(self) -> int:
        return self.rgba_image.shape[1]

    def get_rgb(self) -> np.ndarray:
        return self.rgba_image[:, :, :3]

    def copy(repr_self) -> "Layer":
        return Layer(
            layer_id=repr_self.layer_id,
            layer_type=repr_self.layer_type,
            rgba_image=repr_self.rgba_image.copy(),
            alpha_mask=repr_self.alpha_mask.copy(),
            z_index=repr_self.z_index,
            bounding_box=repr_self.bounding_box,
            action=repr_self.action,
            metadata=dict(repr_self.metadata),
        )


class Canvas:
    """Multi-layer RGBA graphics canvas managing layer stack and deterministic compositing."""

    def __init__(self, width: int = 1280, height: int = 720, original_image: Optional[np.ndarray] = None):
        self.width = width
        self.height = height
        self.original_image = original_image  # H x W x 3 RGB uint8
        self.layers: List[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        if layer.width != self.width or layer.height != self.height:
            raise ValueError(
                f"Layer dimensions ({layer.width}x{layer.height}) do not match canvas ({self.width}x{self.height})"
            )
        self.layers.append(layer)
        self.sort_layers()

    def get_layer_by_id(self, layer_id: str) -> Optional[Layer]:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None

    def get_layers_by_type(self, layer_type: LayerType) -> List[Layer]:
        return [layer for layer in self.layers if layer.layer_type == layer_type]

    def sort_layers(self) -> None:
        """Sorts layers ascending by z_index."""
        self.layers.sort(key=lambda l: l.z_index)

    def composite_rgba(self) -> np.ndarray:
        """Renders all layers in z-index order into a single composite 1280x720 RGB image."""
        composite = np.zeros((self.height, self.width, 3), dtype=np.float32)

        self.sort_layers()
        for layer in self.layers:
            if layer.action == LayerAction.REMOVE:
                continue

            layer_rgb = layer.rgba_image[:, :, :3].astype(np.float32)
            alpha = (layer.alpha_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            # Alpha blend equation: C_out = C_src * alpha + C_dst * (1 - alpha)
            composite = layer_rgb * alpha + composite * (1.0 - alpha)

        return np.clip(composite, 0, 255).astype(np.uint8)

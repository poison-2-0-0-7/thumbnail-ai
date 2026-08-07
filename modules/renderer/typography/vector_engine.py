"""
Vector Typography Render Engine

Programmatically renders crisp vector typography, drop shadows, multi-color gradient fills,
and pill container backgrounds onto high-DPI RGBA canvas layers.
"""

from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ..core.schema import TypographySpec, LayerType, LayerAction
from ..core.canvas import Layer


class VectorTypographyEngine:
    """High-DPI vector rendering engine for YouTube thumbnail text overlays."""

    def __init__(self, canvas_width: int = 1280, canvas_height: int = 720):
        self.width = canvas_width
        self.height = canvas_height

    def hex_to_rgba(self, hex_str: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        hex_clean = hex_str.lstrip("#")
        if len(hex_clean) == 6:
            r, g, b = (int(hex_clean[i : i + 2], 16) for i in (0, 2, 4))
            return (r, g, b, alpha)
        elif len(hex_clean) == 8:
            return tuple(int(hex_clean[i : i + 2], 16) for i in (0, 2, 4, 6))
        return (255, 255, 255, alpha)

    def render_typography_layer(
        self,
        spec: TypographySpec,
        target_bbox: Tuple[int, int, int, int],
        layer_id: str = "layer_text_primary",
        z_index: int = 10,
    ) -> Layer:
        """Renders vector typography onto an RGBA canvas layer.

        Args:
            spec: TypographySpec containing text_content, font properties, pill settings, drop shadow.
            target_bbox: (x_min, y_min, x_max, y_max) placement coordinates on canvas.
            layer_id: Unique string identifier for the layer.
            z_index: Ordering priority index.

        Returns:
            Layer object containing RGBA image and 8-bit alpha mask.
        """
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Load font with fallback to default PIL font
        try:
            font = ImageFont.truetype(spec.font_family, spec.font_size)
        except OSError:
            font = ImageFont.load_default()

        text = spec.text_content
        x_min, y_min, x_max, y_max = target_bbox

        # Compute text bounding box size
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top

        # Align text inside target bounding box
        text_x = x_min + (x_max - x_min - text_w) // 2
        text_y = y_min + (y_max - y_min - text_h) // 2

        # 1. Render Pill Container Background (if enabled)
        if spec.pill_container_enabled:
            padding = 20
            pill_box = (
                text_x - padding,
                text_y - padding,
                text_x + text_w + padding,
                text_y + text_h + padding,
            )
            pill_color = self.hex_to_rgba(spec.pill_fill_color)
            draw.rounded_rectangle(
                pill_box,
                radius=spec.pill_corner_radius,
                fill=pill_color,
            )

        # 2. Render Drop Shadow
        if spec.drop_shadow.enabled:
            shadow_img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_img)
            shadow_x = text_x + spec.drop_shadow.offset_x
            shadow_y = text_y + spec.drop_shadow.offset_y
            shadow_color = self.hex_to_rgba(
                spec.drop_shadow.color_hex,
                alpha=int(spec.drop_shadow.opacity * 255),
            )

            shadow_draw.text(
                (shadow_x, shadow_y),
                text,
                font=font,
                fill=shadow_color,
                stroke_width=spec.stroke_width,
                stroke_fill=shadow_color,
            )
            if spec.drop_shadow.blur_radius > 0:
                shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(spec.drop_shadow.blur_radius / 2))
            img = Image.alpha_composite(img, shadow_img)
            draw = ImageDraw.Draw(img)

        # 3. Render Main Text & Stroke Contour
        fill_color = self.hex_to_rgba(spec.fill_colors[0])
        stroke_color = self.hex_to_rgba(spec.stroke_color)

        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=fill_color,
            stroke_width=spec.stroke_width,
            stroke_fill=stroke_color,
        )

        rgba_arr = np.array(img, dtype=np.uint8)
        alpha_mask = rgba_arr[:, :, 3].copy()

        return Layer(
            layer_id=layer_id,
            layer_type=LayerType.TYPOGRAPHY,
            rgba_image=rgba_arr,
            alpha_mask=alpha_mask,
            z_index=z_index,
            bounding_box=target_bbox,
            action=LayerAction.RENDER_VECTOR_TEXT,
            metadata={"text_content": text, "font": spec.font_family},
        )

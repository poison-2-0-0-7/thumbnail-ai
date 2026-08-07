"""
Rendering Engine V2.1 Top-Level Orchestrator

Executes the Layer-Isolated Non-Destructive Editing (LINDE) pipeline:
Original Thumbnail -> Vision & Matting -> EditPlan -> NDAER Relighting -> Vector Typography -> Quality Gate -> Final Thumbnail
"""

from typing import Optional, Tuple
import numpy as np

from .core.schema import EditPlan, QualityReport
from .core.config import RendererConfig
from .core.canvas import Canvas
from .vision.segmentor import CoarseSegmentor
from .vision.matting import AlphaMattingEngine
from .generative.relighter import NonDestructiveEdgeRelighter
from .typography.saliency_solver import SaliencySolver
from .typography.vector_engine import VectorTypographyEngine
from .quality.gatekeeper import QualityGatekeeper


class RenderingEngineV2:
    """Main Rendering Engine V2.1 Entry Point."""

    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()
        self.segmentor = CoarseSegmentor(device=self.config.device, fp16=self.config.fp16_enabled)
        self.matting_engine = AlphaMattingEngine(device=self.config.device)
        self.relighter = NonDestructiveEdgeRelighter(device=self.config.device)
        self.saliency_solver = SaliencySolver()
        self.vector_typography = VectorTypographyEngine()
        self.gatekeeper = QualityGatekeeper(config=self.config)

    def render(
        self,
        original_image_rgb: np.ndarray,  # H x W x 3 uint8
        edit_plan: EditPlan,
    ) -> Tuple[np.ndarray, QualityReport]:
        """Executes full non-destructive thumbnail rendering pipeline.

        Args:
            original_image_rgb: 1280x720 RGB NumPy array of original thumbnail.
            edit_plan: Validated EditPlan containing layer directives & typography specs.

        Returns:
            Tuple of (Rendered 1280x720 RGB array, QualityReport)
        """
        h, w, _ = original_image_rgb.shape
        canvas = Canvas(width=w, height=h, original_image=original_image_rgb)

        # 1. Vision & Layer Extraction Stage
        raw_layers = self.segmentor.extract_layers(original_image_rgb)
        for layer in raw_layers:
            # Refine binary masks into continuous 8-bit alpha mattes
            layer.alpha_mask = self.matting_engine.refine_alpha(original_image_rgb, layer.alpha_mask)
            canvas.add_layer(layer)

        # 2. Non-Destructive Additive Edge Relighting (NDAER) Stage
        for layer_spec in edit_plan.layers:
            layer = canvas.get_layer_by_id(layer_spec.layer_id)
            if layer and layer_spec.relighting and layer_spec.relighting.enabled:
                relit_layer = self.relighter.apply_relighting(layer, layer_spec.relighting)
                # Replace layer in canvas with relit version
                canvas.layers = [l if l.layer_id != layer.layer_id else relit_layer for l in canvas.layers]

        # 3. Vector Typography & Anti-Collision Placement Stage
        for layer_spec in edit_plan.layers:
            if layer_spec.typography_spec:
                spec = layer_spec.typography_spec
                # Calculate optimal bounding box using Saliency Solver
                target_dims = (600, 200)
                protected_masks = [l.alpha_mask for l in canvas.layers if l.z_index > 0]
                optimal_bbox = self.saliency_solver.find_optimal_text_bbox(
                    original_image_rgb,
                    protected_masks,
                    target_dims,
                )

                typo_layer = self.vector_typography.render_typography_layer(
                    spec=spec,
                    target_bbox=optimal_bbox,
                    layer_id=layer_spec.layer_id,
                    z_index=layer_spec.z_index,
                )
                canvas.add_layer(typo_layer)

        # 4. Final RGBA Compositing
        final_rgb = canvas.composite_rgba()

        # 5. Quality Gatekeeper Evaluation
        quality_report = self.gatekeeper.evaluate(final_canvas=canvas)

        return final_rgb, quality_report

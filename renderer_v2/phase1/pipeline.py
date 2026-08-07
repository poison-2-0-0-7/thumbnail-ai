"""Phase1Pipeline orchestrator connecting Scene Decomposer, Locked Region Mask Builder, Inpainter, and Recompositor."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import cv2
import numpy as np
from loguru import logger
from PIL import Image
import torch

from .config import Phase1Config, default_config
from .schemas import Instance, PipelineResult, SceneGraph
from .model_registry import ModelRegistry
from .scene_decomposer.base import Detector, Matter, DepthEstimator
from .scene_decomposer.decomposer import SceneDecomposer
from .inpaint.base import BackgroundInpainter
from .inpaint.mask_utils import build_locked_region_mask, build_inpaint_inverse_mask
from .compositor.recompositor import Recompositor


class Phase1Pipeline:
    """Orchestrates end-to-end Phase 1 thumbnail reconstruction pipeline."""

    def __init__(
        self,
        detector: Detector,
        matter: Matter,
        depth: DepthEstimator,
        inpainter: BackgroundInpainter,
        registry: Optional[ModelRegistry] = None,
        config: Phase1Config = default_config,
    ) -> None:
        self.detector = detector
        self.matter = matter
        self.depth = depth
        self.inpainter = inpainter
        self.registry = registry or ModelRegistry(config)
        self.config = config
        self.decomposer = SceneDecomposer(
            detector=detector,
            matter=matter,
            depth_estimator=depth,
            registry=self.registry,
            config=config,
        )
        self.recompositor = Recompositor(config=config)

    def run(
        self,
        image_input: Union[str, Path, np.ndarray],
        class_prompts: Optional[List[str]] = None,
        inpaint_prompt: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """Execute Phase 1 pipeline end-to-end.

        Args:
            image_input: File path or RGB numpy array for input thumbnail.
            class_prompts: List of text prompts for instance detection.
            inpaint_prompt: Text prompt for background generation.
            output_dir: Target directory for debug artifacts (defaults to config.debug_dir).

        Returns:
            PipelineResult containing final image, scene graph, intermediates, and debug dictionary.
        """
        start_time = time.perf_counter()
        stage_latencies: Dict[str, float] = {}
        
        # Determine debug output directory
        out_dir = Path(output_dir) if output_dir else self.config.debug_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Configure file logging for debug output 12_pipeline.log
        log_file_path = out_dir / "12_pipeline.log"
        logger_id = logger.add(log_file_path, mode="w")

        try:
            logger.info("=== Starting Renderer V2 Phase 1 Pipeline Run ===")
            
            # Load input image
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    raise FileNotFoundError(f"Input image file does not exist: {img_path}")
                pil_img = Image.open(img_path).convert("RGB")
                image = np.array(pil_img)
            else:
                image = image_input

            prompts = class_prompts or self.config.default_class_prompts
            inp_prompt = inpaint_prompt or self.config.default_inpaint_prompt

            # Step 1: Scene Decomposition
            t0 = time.perf_counter()
            scene_graph = self.decomposer.decompose(image, prompts)
            stage_latencies["scene_decomposition"] = time.perf_counter() - t0

            # Step 2: Locked Region Mask Building & Inversion
            t0 = time.perf_counter()
            locked_mask = build_locked_region_mask(scene_graph)
            inverse_bg_mask = build_inpaint_inverse_mask(
                locked_mask, dilation_px=self.config.mask_dilation_px
            )
            stage_latencies["mask_building"] = time.perf_counter() - t0

            # Step 3: Background Inpainting
            t0 = time.perf_counter()
            inpainted_bg = self.inpainter.inpaint(image, inverse_bg_mask, inp_prompt)
            stage_latencies["background_inpainting"] = time.perf_counter() - t0

            if self.registry:
                self.registry.unload_all()

            # Step 4: Recomposition
            t0 = time.perf_counter()
            final_output = self.recompositor.recomposite(
                scene_graph=scene_graph,
                inpainted_background=inpainted_bg,
                feather_px=self.config.mask_feather_px,
            )
            stage_latencies["recomposition"] = time.perf_counter() - t0

            total_runtime = time.perf_counter() - start_time
            peak_vram_gb = self.registry.get_peak_vram_gb() if self.registry else 0.0

            # Step 5: Save mandatory debug artifacts
            debug_artifacts = self._generate_debug_artifacts(
                out_dir=out_dir,
                source_image=image,
                scene_graph=scene_graph,
                locked_mask=locked_mask,
                inverse_bg_mask=inverse_bg_mask,
                inpainted_bg=inpainted_bg,
                final_output=final_output,
                total_runtime=total_runtime,
                peak_vram_gb=peak_vram_gb,
                stage_latencies=stage_latencies,
            )

            logger.info("=== Phase 1 Pipeline Completed Successfully in {t:.2f}s ===", t=total_runtime)

            result = PipelineResult(
                output_image=final_output,
                scene_graph=scene_graph,
                inpainted_background=inpainted_bg,
                locked_region_mask=locked_mask,
                debug_artifacts=debug_artifacts,
            )
            return result

        finally:
            logger.remove(logger_id)

    def _generate_debug_artifacts(
        self,
        out_dir: Path,
        source_image: np.ndarray,
        scene_graph: SceneGraph,
        locked_mask: np.ndarray,
        inverse_bg_mask: np.ndarray,
        inpainted_bg: np.ndarray,
        final_output: np.ndarray,
        total_runtime: float,
        peak_vram_gb: float,
        stage_latencies: Dict[str, float],
    ) -> Dict[str, Any]:
        """Save standard debug files (01-13) and generate self-contained HTML report."""
        artifacts: Dict[str, Any] = {}

        # 01_original.png
        Image.fromarray(source_image).save(out_dir / "01_original.png")
        artifacts["original"] = source_image

        # 02_detection_overlay.png
        overlay = source_image.copy()
        for inst in scene_graph.instances:
            xmin, ymin, xmax, ymax = inst.bbox
            color = (0, 255, 0) if inst.locked else (255, 0, 0)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                overlay,
                f"{inst.instance_id} ({inst.cls})",
                (xmin, max(15, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        Image.fromarray(overlay).save(out_dir / "02_detection_overlay.png")
        artifacts["detection_overlay"] = overlay

        # 03_masks/
        masks_dir = out_dir / "03_masks"
        masks_dir.mkdir(exist_ok=True)
        for inst in scene_graph.instances:
            mask_img = (inst.mask * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(masks_dir / f"{inst.instance_id}.png")

        # 04_alpha_mattes/
        mattes_dir = out_dir / "04_alpha_mattes"
        mattes_dir.mkdir(exist_ok=True)
        for inst in scene_graph.instances:
            matte_img = (inst.alpha_matte * 255).astype(np.uint8)
            Image.fromarray(matte_img).save(mattes_dir / f"{inst.instance_id}.png")

        # 05_depth.png
        depth_norm = (scene_graph.depth_map * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        Image.fromarray(depth_colormap_rgb).save(out_dir / "05_depth.png")
        artifacts["depth"] = depth_colormap_rgb

        # 06_locked_regions.png
        locked_img = (locked_mask * 255).astype(np.uint8)
        Image.fromarray(locked_img).save(out_dir / "06_locked_regions.png")
        artifacts["locked_regions"] = locked_img

        # 07_background_mask.png
        Image.fromarray(inverse_bg_mask).save(out_dir / "07_background_mask.png")
        artifacts["background_mask"] = inverse_bg_mask

        # 08_inpaint.png
        Image.fromarray(inpainted_bg).save(out_dir / "08_inpaint.png")
        artifacts["inpaint"] = inpainted_bg

        # 09_recomposite.png
        Image.fromarray(final_output).save(out_dir / "09_recomposite.png")
        artifacts["recomposite"] = final_output

        # 10_scene_graph.json
        sg_data = {
            "width": scene_graph.width,
            "height": scene_graph.height,
            "instance_count": len(scene_graph.instances),
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "cls": inst.cls,
                    "bbox": inst.bbox,
                    "depth_layer": inst.depth_layer,
                    "locked": inst.locked,
                }
                for inst in scene_graph.instances
            ],
        }
        with open(out_dir / "10_scene_graph.json", "w") as f:
            json.dump(sg_data, f, indent=2)

        # 11_metrics.json
        metrics_data = {
            "total_runtime_seconds": round(total_runtime, 4),
            "peak_vram_gb": round(peak_vram_gb, 4),
            "stage_latencies_seconds": {k: round(v, 4) for k, v in stage_latencies.items()},
            "resolution": f"{scene_graph.width}x{scene_graph.height}",
            "device": self.config.device,
        }
        with open(out_dir / "11_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # 13_report.html
        self._write_html_report(out_dir / "13_report.html", sg_data, metrics_data)

        return artifacts

    def _write_html_report(self, filepath: Path, sg_data: dict, metrics_data: dict) -> None:
        """Write self-contained HTML inspection report."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Thumbnail AI - Phase 1 Pipeline Inspection Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        h1, h2 {{ color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .card h3 {{ margin: 0 0 8px 0; color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; }}
        .card p {{ margin: 0; font-size: 1.5rem; font-weight: bold; color: #f1f5f9; }}
        .stage-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .stage-card {{ background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }}
        .stage-card img {{ width: 100%; height: auto; display: block; background: #000; }}
        .stage-card .title {{ padding: 12px; font-weight: bold; background: #334155; color: #e2e8f0; }}
        pre {{ background: #020617; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; color: #a5f3fc; }}
    </style>
</head>
<body>
    <h1>Thumbnail AI — Phase 1 Inspection Report</h1>
    
    <h2>Performance & Execution Metrics</h2>
    <div class="metrics-grid">
        <div class="card"><h3>Total Runtime</h3><p>{metrics_data['total_runtime_seconds']} s</p></div>
        <div class="card"><h3>Peak VRAM</h3><p>{metrics_data['peak_vram_gb']} GB</p></div>
        <div class="card"><h3>Resolution</h3><p>{metrics_data['resolution']}</p></div>
        <div class="card"><h3>Instances</h3><p>{sg_data['instance_count']}</p></div>
    </div>

    <h2>Pipeline Stage Outputs</h2>
    <div class="stage-grid">
        <div class="stage-card"><div class="title">01. Original Image</div><img src="01_original.png" alt="Original"></div>
        <div class="stage-card"><div class="title">02. Detection Overlay</div><img src="02_detection_overlay.png" alt="Detection"></div>
        <div class="stage-card"><div class="title">05. Depth Map</div><img src="05_depth.png" alt="Depth"></div>
        <div class="stage-card"><div class="title">06. Locked Regions Mask</div><img src="06_locked_regions.png" alt="Locked"></div>
        <div class="stage-card"><div class="title">07. Inpaint Background Mask</div><img src="07_background_mask.png" alt="Mask"></div>
        <div class="stage-card"><div class="title">08. Inpainting Output</div><img src="08_inpaint.png" alt="Inpaint"></div>
        <div class="stage-card"><div class="title">09. Final Recomposite</div><img src="09_recomposite.png" alt="Recomposite"></div>
    </div>

    <h2>Scene Graph JSON</h2>
    <pre>{json.dumps(sg_data, indent=2)}</pre>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

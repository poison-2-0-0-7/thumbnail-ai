"""CLI entry point for running Phase 1 pipeline on an image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
import numpy as np

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry
from renderer_v2.phase1.scene_decomposer.groundingdino_sam2_detector import GroundingDINOSAM2Detector
from renderer_v2.phase1.scene_decomposer.birefnet_matter import BiRefNetMatter
from renderer_v2.phase1.scene_decomposer.depth_anything import DepthAnythingEstimator
from renderer_v2.phase1.inpaint.sdxl_brushnet import SDXLBrushNetInpainter
from renderer_v2.phase1.pipeline import Phase1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Renderer V2 Phase 1 Scene Decomposition & Background Inpainting on an image."
    )
    parser.add_argument("image_path", type=str, help="Path to input thumbnail image (PNG/JPG)")
    parser.add_argument(
        "--class-prompts",
        nargs="+",
        default=["person", "logo", "product"],
        help="Space-separated class prompts for detection (default: person logo product)",
    )
    parser.add_argument(
        "--inpaint-prompt",
        type=str,
        default="modern vibrant YouTube thumbnail background, clean studio lighting, high quality, 8k resolution",
        help="Text prompt for background synthesis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug",
        help="Output directory for debug artifacts and final image (default: debug)",
    )
    parser.add_argument("--vram-limit", type=float, default=8.0, help="Max VRAM limit in GB (default: 8.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.image_path)
    if not input_path.exists():
        logger.error("Input image not found: {path}", path=input_path)
        sys.exit(1)

    config = Phase1Config(
        max_vram_gb=args.vram_limit,
        debug_dir=Path(args.output_dir),
    )
    registry = ModelRegistry(config)

    detector = GroundingDINOSAM2Detector(config=config, registry=registry)
    matter = BiRefNetMatter(config=config, registry=registry)
    depth = DepthAnythingEstimator(config=config, registry=registry)
    inpainter = SDXLBrushNetInpainter(config=config, registry=registry)

    pipeline = Phase1Pipeline(
        detector=detector,
        matter=matter,
        depth=depth,
        inpainter=inpainter,
        registry=registry,
        config=config,
    )

    logger.info("Executing Phase 1 pipeline on {path}...", path=input_path)
    result = pipeline.run(
        image_input=input_path,
        class_prompts=args.class_prompts,
        inpaint_prompt=args.inpaint_prompt,
        output_dir=args.output_dir,
    )

    logger.info("Pipeline execution finished!")
    logger.info("Outputs written to: {out_dir}", out_dir=args.output_dir)
    logger.info("Report available at: {out_dir}/13_report.html", out_dir=args.output_dir)


if __name__ == "__main__":
    main()

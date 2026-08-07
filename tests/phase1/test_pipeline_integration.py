"""Integration tests for Phase1Pipeline end-to-end run and debug artifacts."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry
from renderer_v2.phase1.scene_decomposer.groundingdino_sam2_detector import GroundingDINOSAM2Detector
from renderer_v2.phase1.scene_decomposer.birefnet_matter import BiRefNetMatter
from renderer_v2.phase1.scene_decomposer.depth_anything import DepthAnythingEstimator
from renderer_v2.phase1.inpaint.sdxl_brushnet import SDXLBrushNetInpainter
from renderer_v2.phase1.pipeline import Phase1Pipeline


@pytest.mark.requires_models
def test_pipeline_integration_end_to_end(sample_rgb_image: np.ndarray, test_config: Phase1Config, model_registry: ModelRegistry, tmp_path: Path):
    debug_out_dir = tmp_path / "debug_run"

    detector = GroundingDINOSAM2Detector(config=test_config, registry=model_registry)
    matter = BiRefNetMatter(config=test_config, registry=model_registry)
    depth = DepthAnythingEstimator(config=test_config, registry=model_registry)
    inpainter = SDXLBrushNetInpainter(config=test_config, registry=model_registry)

    pipeline = Phase1Pipeline(
        detector=detector,
        matter=matter,
        depth=depth,
        inpainter=inpainter,
        registry=model_registry,
        config=test_config,
    )

    result = pipeline.run(
        image_input=sample_rgb_image,
        class_prompts=["person", "logo"],
        inpaint_prompt="vibrant neon studio background",
        output_dir=debug_out_dir,
    )

    assert result.output_image.shape == sample_rgb_image.shape
    assert result.scene_graph is not None

    # Assert all mandatory 13 debug outputs exist
    expected_files = [
        "01_original.png",
        "02_detection_overlay.png",
        "03_masks",
        "04_alpha_mattes",
        "05_depth.png",
        "06_locked_regions.png",
        "07_background_mask.png",
        "08_inpaint.png",
        "09_recomposite.png",
        "10_scene_graph.json",
        "11_metrics.json",
        "12_pipeline.log",
        "13_report.html",
    ]

    for fname in expected_files:
        assert (debug_out_dir / fname).exists(), f"Missing required debug artifact: {fname}"

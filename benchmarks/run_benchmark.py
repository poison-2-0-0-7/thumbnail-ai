"""Benchmark execution framework measuring REAL production model VRAM, latency, and inference times."""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from loguru import logger
import torch

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry
from renderer_v2.phase1.scene_decomposer.groundingdino_sam2_detector import GroundingDINOSAM2Detector
from renderer_v2.phase1.scene_decomposer.birefnet_matter import BiRefNetMatter
from renderer_v2.phase1.scene_decomposer.depth_anything import DepthAnythingEstimator
from renderer_v2.phase1.inpaint.sdxl_brushnet import SDXLBrushNetInpainter
from renderer_v2.phase1.inpaint.mask_utils import build_inpaint_inverse_mask


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """Calculate PSNR between two images strictly outside the inpaint mask."""
    outside_mask = (mask == 0)
    if not outside_mask.any():
        return 0.0
    
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    mse = np.mean(diff[outside_mask] ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def run_benchmark(image_path: Optional[Path] = None) -> Dict[str, Any]:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = Phase1Config()
    registry = ModelRegistry(config)

    # Use input image or create test thumbnail
    if image_path and image_path.exists():
        fixture_img = np.array(Image.open(image_path).convert("RGB"))
    else:
        # Create standard 1280x720 thumbnail
        fixture_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        y, x = np.ogrid[:720, :1280]
        fixture_img[:, :, 0] = (x / 1280 * 255).astype(np.uint8)
        fixture_img[:, :, 1] = (y / 720 * 255).astype(np.uint8)
        fixture_img[180:540, 480:800] = [240, 180, 140]

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "device": config.device,
        "vram_ceiling_gb": config.max_vram_gb,
        "models": {},
    }

    logger.info("Starting Benchmark Run with Production Models on {w}x{h} image...", w=fixture_img.shape[1], h=fixture_img.shape[0])

    # 1. Real GroundingDINO + SAM2.1
    registry.reset_vram_stats()
    detector = GroundingDINOSAM2Detector(config=config, registry=registry)
    t0 = time.perf_counter()
    instances = detector.detect(fixture_img, ["person", "logo"])
    latency_det = time.perf_counter() - t0
    vram_det = registry.get_peak_vram_gb()
    registry.unload_all()

    metrics["models"]["groundingdino_sam2.1"] = {
        "peak_vram_gb": round(vram_det, 3),
        "latency_seconds": round(latency_det, 3),
        "instances_detected": len(instances),
    }

    # 2. Real BiRefNet-lite
    registry.reset_vram_stats()
    matter = BiRefNetMatter(config=config, registry=registry)
    t0 = time.perf_counter()
    if instances:
        _ = matter.refine(fixture_img, instances[0])
    latency_mat = time.perf_counter() - t0
    vram_mat = registry.get_peak_vram_gb()
    registry.unload_all()

    metrics["models"]["birefnet_lite"] = {
        "peak_vram_gb": round(vram_mat, 3),
        "latency_seconds": round(latency_mat, 3),
    }

    # 3. Real Depth-Anything V2 Small
    registry.reset_vram_stats()
    depth_est = DepthAnythingEstimator(config=config, registry=registry)
    t0 = time.perf_counter()
    depth_map = depth_est.estimate(fixture_img)
    latency_dep = time.perf_counter() - t0
    vram_dep = registry.get_peak_vram_gb()
    registry.unload_all()

    metrics["models"]["depth_anything_v2"] = {
        "peak_vram_gb": round(vram_dep, 3),
        "latency_seconds": round(latency_dep, 3),
        "depth_min": round(float(depth_map.min()), 3),
        "depth_max": round(float(depth_map.max()), 3),
    }

    # 4. Real SDXL Inpaint
    registry.reset_vram_stats()
    inpainter = SDXLBrushNetInpainter(config=config, registry=registry)
    dummy_mask = np.zeros(fixture_img.shape[:2], dtype=np.uint8)
    dummy_mask[200:400, 500:700] = 255
    t0 = time.perf_counter()
    inp_out = inpainter.inpaint(fixture_img, dummy_mask, config.default_inpaint_prompt)
    latency_inp = time.perf_counter() - t0
    vram_inp = registry.get_peak_vram_gb()
    psnr_inp = calculate_psnr(fixture_img, inp_out, dummy_mask)
    registry.unload_all()

    metrics["models"]["sdxl_inpaint"] = {
        "peak_vram_gb": round(vram_inp, 3),
        "latency_seconds": round(latency_inp, 3),
        "masked_preservation_psnr_db": round(psnr_inp, 2),
    }

    # Save benchmark JSON
    json_path = results_dir / f"real_benchmark_{int(time.time())}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save CSV
    csv_path = results_dir / "real_benchmark_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Peak_VRAM_GB", "Latency_Seconds"])
        for k, v in metrics["models"].items():
            writer.writerow([k, v.get("peak_vram_gb", 0), v.get("latency_seconds", 0)])

    logger.info("Benchmark complete! Metrics written to {p}", p=json_path)
    return metrics


if __name__ == "__main__":
    sample_p = Path(__file__).parent.parent / "tests" / "phase1" / "fixtures" / "sample_talking_head.png"
    run_benchmark(sample_p)

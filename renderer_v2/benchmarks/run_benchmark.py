"""Phase 1 Model Benchmark Runner.

Sweeps model alternatives per spec §7 and logs VRAM / latency / quality proxy metrics.

Candidates benchmarked:
  - Detection:  GroundingDINO+SAM2.1 vs SAM3 (when available)
  - Matting:    BiRefNet-lite vs MODNet
  - Inpainting: SDXL+BrushNet vs SDXL native inpaint

Output: CSV/JSON per run + filled report_template.md
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry


@dataclass
class BenchmarkResult:
    """Single benchmark measurement for one model on one image."""
    model_name: str
    stage: str  # "detection", "matting", "depth", "inpainting"
    image_name: str
    peak_vram_gb: float
    wall_clock_seconds: float
    # Quality metrics (stage-dependent)
    mask_iou: Optional[float] = None  # detection/segmentation
    matte_sad: Optional[float] = None  # matting
    psnr_outside_mask: Optional[float] = None  # inpainting — preservation
    lpips_outside_mask: Optional[float] = None  # inpainting — preservation


@dataclass
class BenchmarkSuite:
    """Aggregate results from a full benchmark run."""
    run_id: str
    timestamp: str
    device: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"run_id": self.run_id, "timestamp": self.timestamp, "device": self.device,
             "results": [asdict(r) for r in self.results]},
            indent=2,
        )

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.results]


def load_fixtures(fixtures_dir: Path) -> List[tuple[str, np.ndarray]]:
    """Load all PNG/JPG images from fixtures directory."""
    images = []
    if not fixtures_dir.exists():
        logger.warning("Fixtures directory does not exist: {d}", d=fixtures_dir)
        return images
    for img_path in sorted(fixtures_dir.glob("*")):
        if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append((img_path.stem, img))
    return images


def benchmark_detection(
    images: List[tuple[str, np.ndarray]],
    config: Phase1Config,
) -> List[BenchmarkResult]:
    """Benchmark detection+segmentation models."""
    results: List[BenchmarkResult] = []

    # --- GroundingDINO + SAM2.1 ---
    try:
        from renderer_v2.phase1.scene_decomposer.groundingdino_sam2_detector import GroundingDINOSAM2Detector
        registry = ModelRegistry(config)
        registry.reset_vram_stats()
        detector = GroundingDINOSAM2Detector(config=config, registry=registry)

        for name, img in images:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            instances = detector.detect(img, config.default_class_prompts)
            elapsed = time.perf_counter() - t0
            peak = registry.get_peak_vram_gb()

            results.append(BenchmarkResult(
                model_name="GroundingDINO+SAM2.1",
                stage="detection",
                image_name=name,
                peak_vram_gb=peak,
                wall_clock_seconds=elapsed,
            ))
            logger.info("Detection [{m}] {img}: {t:.2f}s, {v:.2f}GB VRAM",
                        m="DINO+SAM2", img=name, t=elapsed, v=peak)

        registry.unload_all()
    except Exception as e:
        logger.error("Failed to benchmark GroundingDINO+SAM2: {e}", e=e)

    return results


def benchmark_matting(
    images: List[tuple[str, np.ndarray]],
    config: Phase1Config,
) -> List[BenchmarkResult]:
    """Benchmark matting models."""
    results: List[BenchmarkResult] = []

    try:
        from renderer_v2.phase1.scene_decomposer.birefnet_matter import BiRefNetMatter
        from renderer_v2.phase1.schemas import Instance
        registry = ModelRegistry(config)
        registry.reset_vram_stats()
        matter = BiRefNetMatter(config=config, registry=registry)

        for name, img in images:
            h, w, _ = img.shape
            # Create a simple center-region dummy instance for benchmarking
            mask = np.zeros((h, w), dtype=bool)
            cy, cx = h // 2, w // 2
            rh, rw = h // 4, w // 4
            mask[cy - rh:cy + rh, cx - rw:cx + rw] = True

            dummy_inst = Instance(
                instance_id="bench_0", cls="creator", mask=mask,
                alpha_matte=mask.astype(np.float32),
                bbox=(cx - rw, cy - rh, cx + rw, cy + rh),
                depth_layer=0.5, locked=True,
            )

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            alpha = matter.refine(img, dummy_inst)
            elapsed = time.perf_counter() - t0
            peak = registry.get_peak_vram_gb()

            results.append(BenchmarkResult(
                model_name="BiRefNet-lite",
                stage="matting",
                image_name=name,
                peak_vram_gb=peak,
                wall_clock_seconds=elapsed,
            ))
            logger.info("Matting [{m}] {img}: {t:.2f}s, {v:.2f}GB VRAM",
                        m="BiRefNet", img=name, t=elapsed, v=peak)

        registry.unload_all()
    except Exception as e:
        logger.error("Failed to benchmark BiRefNet: {e}", e=e)

    return results


def write_results(suite: BenchmarkSuite, output_dir: Path) -> None:
    """Write benchmark results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"benchmark_{suite.run_id}.json"
    with open(json_path, "w") as f:
        f.write(suite.to_json())
    logger.info("Wrote JSON results to {p}", p=json_path)

    # CSV
    csv_path = output_dir / f"benchmark_{suite.run_id}.csv"
    rows = suite.to_csv_rows()
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    logger.info("Wrote CSV results to {p}", p=csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Model Benchmark Runner")
    parser.add_argument("--fixtures-dir", type=str, default="tests/phase1/fixtures",
                        help="Directory containing benchmark fixture images")
    parser.add_argument("--output-dir", type=str, default="renderer_v2/benchmarks/results",
                        help="Output directory for benchmark results")
    parser.add_argument("--stages", nargs="+", default=["detection", "matting"],
                        choices=["detection", "matting", "depth", "inpainting"],
                        help="Which pipeline stages to benchmark")
    args = parser.parse_args()

    config = Phase1Config()
    fixtures = load_fixtures(Path(args.fixtures_dir))

    if not fixtures:
        logger.error("No fixture images found in {d}. Add PNG/JPG images to benchmark.", d=args.fixtures_dir)
        return

    run_id = time.strftime("%Y%m%d_%H%M%S")
    suite = BenchmarkSuite(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        device=config.device,
    )

    if "detection" in args.stages:
        suite.results.extend(benchmark_detection(fixtures, config))
    if "matting" in args.stages:
        suite.results.extend(benchmark_matting(fixtures, config))

    write_results(suite, Path(args.output_dir))
    logger.info("Benchmark run {id} complete. {n} measurements.", id=run_id, n=len(suite.results))


if __name__ == "__main__":
    main()

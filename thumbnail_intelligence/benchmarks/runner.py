"""
runner.py
=========

BenchmarkRunner Implementation for Phase 6.1 Benchmark Framework.
Executes benchmark sessions across datasets, measures quality, latency, VRAM, and builds side-by-side visual comparison artifacts.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from thumbnail_intelligence.benchmarks.dataset_loader import DatasetLoader
from thumbnail_intelligence.benchmarks.failure_analyzer import FailureAnalyzer
from thumbnail_intelligence.benchmarks.models import (
    BenchmarkResult,
    BenchmarkSession,
    BenchmarkSummaryMetrics,
    DatasetItem,
    FailureCategory,
)
from thumbnail_intelligence.optimization.engine import IterativeOptimizationEngine
from thumbnail_intelligence.optimization.models import StoppingPolicy
from renderer_v2.runtime.manager import ModelRuntimeManager

logger = logging.getLogger(__name__)


class BenchmarkRunnerError(RuntimeError):
    """Exception raised for benchmark runner errors."""
    pass


class BenchmarkRunner:
    """Executes closed-loop benchmarks across dataset samples."""

    def __init__(
        self,
        optimization_engine: Optional[IterativeOptimizationEngine] = None,
        runtime_manager: Optional[ModelRuntimeManager] = None,
    ) -> None:
        self.opt_engine = optimization_engine or IterativeOptimizationEngine()
        self.runtime_manager = runtime_manager or ModelRuntimeManager()

    def run_benchmark(
        self,
        items: List[DatasetItem],
        dataset_name: str = "benchmark_dataset",
        stopping_policy: Optional[StoppingPolicy] = None,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkSession:
        """Run benchmark evaluation across a list of DatasetItem samples.

        Args:
            items: List of DatasetItem samples to evaluate.
            dataset_name: Name of benchmark dataset.
            stopping_policy: Optional StoppingPolicy thresholds.
            output_directory: Directory to store benchmark session artifacts.
            context_overrides: Additional runtime execution metadata overrides.

        Returns:
            BenchmarkSession containing results for all dataset items and summary metrics.
        """
        if not items:
            raise BenchmarkRunnerError("Cannot run benchmark with empty items list.")

        session_id = f"bench_sess_{uuid.uuid4().hex[:8]}"
        out_dir = Path(output_directory) if output_directory else Path(tempfile.mkdtemp(prefix=f"bench_{session_id}_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== Starting BenchmarkRunner Session '{session_id}' ({len(items)} items, dataset='{dataset_name}') ===")

        results: List[BenchmarkResult] = []
        failure_dist: Dict[str, int] = {f.value: 0 for f in FailureCategory}
        cost_dist: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

        sess_t0 = time.time()
        policy = stopping_policy or StoppingPolicy(max_iterations=2)

        for idx, item in enumerate(items, start=1):
            logger.info(f"--- Benchmarking Item {idx}/{len(items)}: '{item.item_id}' ({item.title}) ---")
            item_out_dir = out_dir / item.item_id
            item_out_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            try:
                # 1. Run Iterative Optimization Engine
                if item.brief:
                    opt_sess = self.opt_engine.optimize_brief(
                        brief=item.brief,
                        policy=policy,
                        output_directory=item_out_dir,
                        context_overrides=context_overrides,
                    )
                else:
                    from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
                    brief = DesignBrief()
                    opt_sess = self.opt_engine.optimize_brief(
                        brief=brief,
                        policy=policy,
                        output_directory=item_out_dir,
                        context_overrides=context_overrides,
                    )

                runtime_s = time.time() - t0
                report = opt_sess.report
                best_it = opt_sess.best_iteration

                initial_score = report.initial_score
                final_score = report.final_score
                gain_pts = report.total_gain_pts

                win_eval = best_it.evaluation_set.get_result(best_it.candidate_id)
                ctr_score = win_eval.metrics["estimated_ctr_score"].score if win_eval else 80.0

                # VRAM tracking
                vram_status = self.runtime_manager.get_memory_status()
                peak_vram = vram_status.get("peak_vram_gb", 0.0)
                gpu_mb = vram_status.get("allocated_vram_mb", 0.0)

                # Generate side-by-side visual comparison artifact
                orig_img = item.original_thumbnail_path
                gen_img = opt_sess.history.iterations[0].image_path
                imp_img = report.best_image_path
                comp_path = str(item_out_dir / "visual_comparison.png")

                self._build_visual_comparison(
                    orig_path=orig_img,
                    gen_path=gen_img,
                    imp_path=imp_img,
                    initial_score=initial_score,
                    final_score=final_score,
                    ctr_score=ctr_score,
                    output_path=comp_path,
                )

                # Categorize failure if score is substandard
                fail_cat, fail_reason = FailureAnalyzer.categorize_failure(eval_result=best_it.evaluation_result)
                is_success = fail_cat == FailureCategory.NONE

                res = BenchmarkResult(
                    item_id=item.item_id,
                    success=is_success,
                    failure_category=fail_cat,
                    failure_reason=fail_reason if not is_success else None,
                    initial_score=initial_score,
                    final_score=final_score,
                    score_gain_pts=gain_pts,
                    ctr_prediction=ctr_score,
                    iterations_required=report.total_iterations,
                    runtime_s=round(runtime_s, 2),
                    peak_vram_gb=round(peak_vram, 2),
                    gpu_memory_used_mb=round(gpu_mb, 1),
                    estimated_render_cost=report.estimated_render_cost,
                    original_image_path=orig_img,
                    generated_image_path=gen_img,
                    improved_image_path=imp_img,
                    visual_comparison_path=comp_path,
                    optimization_session=opt_sess,
                )

                results.append(res)
                failure_dist[fail_cat.value] += 1
                cost_dist[report.estimated_render_cost] = cost_dist.get(report.estimated_render_cost, 0) + 1

            except Exception as e:
                runtime_s = time.time() - t0
                fail_cat, fail_reason = FailureAnalyzer.categorize_failure(error_message=str(e))
                logger.error(f"Item '{item.item_id}' benchmark failed: {str(e)}")

                res = BenchmarkResult(
                    item_id=item.item_id,
                    success=False,
                    failure_category=fail_cat,
                    failure_reason=fail_reason,
                    initial_score=0.0,
                    final_score=0.0,
                    score_gain_pts=0.0,
                    ctr_prediction=0.0,
                    iterations_required=1,
                    runtime_s=round(runtime_s, 2),
                    peak_vram_gb=0.0,
                    gpu_memory_used_mb=0.0,
                    estimated_render_cost="HIGH",
                    original_image_path=item.original_thumbnail_path,
                    generated_image_path=None,
                    improved_image_path=None,
                    visual_comparison_path=None,
                    optimization_session=None,
                )
                results.append(res)
                failure_dist[fail_cat.value] += 1

        sess_total_runtime = time.time() - sess_t0
        total_cnt = len(results)
        succ_cnt = sum(1 for r in results if r.success)
        fail_cnt = total_cnt - succ_cnt
        succ_rate = (succ_cnt / float(total_cnt)) * 100.0 if total_cnt > 0 else 0.0
        fail_rate = (fail_cnt / float(total_cnt)) * 100.0 if total_cnt > 0 else 0.0

        succ_results = [r for r in results if r.success]
        avg_init = float(np.mean([r.initial_score for r in succ_results])) if succ_results else 0.0
        avg_final = float(np.mean([r.final_score for r in succ_results])) if succ_results else 0.0
        avg_gain = float(np.mean([r.score_gain_pts for r in succ_results])) if succ_results else 0.0
        avg_ctr = float(np.mean([r.ctr_prediction for r in succ_results])) if succ_results else 0.0
        avg_iters = float(np.mean([r.iterations_required for r in succ_results])) if succ_results else 1.0
        avg_runtime = float(np.mean([r.runtime_s for r in results])) if results else 0.0
        peak_vram_all = max([r.peak_vram_gb for r in results], default=0.0)
        avg_gpu_mb = float(np.mean([r.gpu_memory_used_mb for r in results])) if results else 0.0
        opt_eff = avg_gain / (avg_runtime + 1e-8)

        summary = BenchmarkSummaryMetrics(
            total_samples=total_cnt,
            successful_samples=succ_cnt,
            failed_samples=fail_cnt,
            success_rate_pct=round(succ_rate, 1),
            failure_rate_pct=round(fail_rate, 1),
            avg_initial_score=round(avg_init, 2),
            avg_final_score=round(avg_final, 2),
            avg_score_improvement_pts=round(avg_gain, 2),
            avg_ctr_prediction=round(avg_ctr, 2),
            avg_iterations_required=round(avg_iters, 2),
            avg_runtime_s=round(avg_runtime, 2),
            total_runtime_s=round(sess_total_runtime, 2),
            avg_gpu_memory_mb=round(avg_gpu_mb, 1),
            peak_vram_gb=round(peak_vram_all, 2),
            optimization_efficiency=round(opt_eff, 2),
            failure_distribution=failure_dist,
            render_cost_distribution=cost_dist,
        )

        session = BenchmarkSession(
            session_id=session_id,
            schema_version="1.0.0",
            dataset_name=dataset_name,
            results=results,
            summary=summary,
        )

        logger.info(f"=== Completed BenchmarkRunner Session '{session_id}' (Success: {succ_rate:.1f}%, Avg Score: {avg_final:.1f}) ===")
        return session

    def _build_visual_comparison(
        self,
        orig_path: Optional[str],
        gen_path: str,
        imp_path: str,
        initial_score: float,
        final_score: float,
        ctr_score: float,
        output_path: str,
    ) -> None:
        """Create a 3-column side-by-side visual comparison raster (Original | Initial | Final Improved)."""
        target_w, target_h = 640, 360

        def load_rescale(p: Optional[str], default_text: str) -> np.ndarray:
            if p and os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    return cv2.resize(img, (target_w, target_h))

            blank = np.full((target_h, target_w, 3), 40, dtype=np.uint8)
            cv2.putText(blank, default_text, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            return blank

        img_orig = load_rescale(orig_path, "Original Thumbnail")
        img_gen = load_rescale(gen_path, "Generated Thumbnail")
        img_imp = load_rescale(imp_path, "Improved Thumbnail")

        # Draw labels
        cv2.putText(img_orig, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img_gen, f"GENERATED (Score: {initial_score:.1f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img_imp, f"FINAL IMPROVED (Score: {final_score:.1f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Concatenate horizontally
        concat_img = np.hstack([img_orig, img_gen, img_imp])

        # Add top banner
        banner = np.full((60, concat_img.shape[1], 3), 15, dtype=np.uint8)
        banner_text = f"Thumbnail AI Benchmark Comparison | Initial Score: {initial_score:.1f} -> Final Score: {final_score:.1f} (Gain: +{final_score - initial_score:.1f} pts) | CTR Score: {ctr_score:.1f}"
        cv2.putText(banner, banner_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        final_composite = np.vstack([banner, concat_img])
        cv2.imwrite(output_path, final_composite)

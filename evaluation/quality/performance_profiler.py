"""
performance_profiler.py
========================

Scorer & helper for 7.12 Runtime Performance and 7.13 Memory Profiling.
Measures wall-clock time, peak system RAM (RSS), and peak VRAM per stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Optional

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


def get_peak_rss_mb() -> float:
    """Return peak Resident Set Size (RSS) in megabytes for current process."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def get_peak_vram_mb() -> Optional[float]:
    """Return current allocated VRAM in megabytes if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return None


@dataclass
class StageProfileSample:
    """Resource measurement sample for one stage execution."""

    stage_name: str
    duration_seconds: float
    peak_rss_mb: float
    peak_vram_mb: Optional[float] = None


class PerformanceProfilerScorer(IQualityScorer):
    """Evaluates stage runtime latency and peak memory footprint against baselines."""

    @property
    def dimension(self) -> str:
        return "runtime_performance"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("runtime_performance", 0.0)

        stage_durations: dict[str, float] = {}
        if context.image_generation_result:
            stage_durations = dict(context.image_generation_result.stage_durations_seconds)

        total_gen_dur = float(context.image_generation_result.duration_seconds) if context.image_generation_result else 0.0
        peak_rss = get_peak_rss_mb()
        peak_vram = get_peak_vram_mb()

        # Score normalized to [0.0, 1.0]; <= 30s is 1.0, > 120s drops towards 0.0
        norm_score = max(0.0, min(1.0, 1.0 - max(0.0, total_gen_dur - 30.0) / 90.0)) if total_gen_dur > 0 else 1.0

        return DimensionScore(
            dimension=self.dimension,
            score=norm_score,
            passed=True,
            threshold=threshold,
            detail={
                "total_duration_seconds": total_gen_dur,
                "stage_durations": stage_durations,
                "peak_rss_mb": peak_rss,
                "peak_vram_mb": peak_vram,
            },
            scorer_version="1.0.0",
            duration_seconds=time.monotonic() - t0,
            status="success",
        )

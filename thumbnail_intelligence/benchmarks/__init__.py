"""
Benchmark Framework Package (Phase 6.1 — Benchmark & Evaluation Framework).
Objectively measures Thumbnail AI quality, accuracy, performance, resource usage, and stability across video datasets.

Produces:
- BenchmarkReport (HTML, Markdown, JSON, CSV)
- Leaderboard (Ranked pipeline performance)
- Performance & Resource Usage Tracking (Latency, Peak VRAM, GPU memory)
- FailureAnalysis (7 standardized failure categories)
"""

from thumbnail_intelligence.benchmarks.dataset_loader import (
    DatasetLoader,
    DatasetLoaderError,
)
from thumbnail_intelligence.benchmarks.failure_analyzer import FailureAnalyzer
from thumbnail_intelligence.benchmarks.framework import (
    BenchmarkFramework,
    BenchmarkFrameworkError,
)
from thumbnail_intelligence.benchmarks.leaderboard import LeaderboardBuilder
from thumbnail_intelligence.benchmarks.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSession,
    BenchmarkSummaryMetrics,
    DatasetItem,
    FailureCategory,
    Leaderboard,
    LeaderboardEntry,
)
from thumbnail_intelligence.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkRunnerError,
)

__all__ = [
    "BenchmarkFramework",
    "BenchmarkFrameworkError",
    "BenchmarkRunner",
    "BenchmarkRunnerError",
    "DatasetLoader",
    "DatasetLoaderError",
    "FailureAnalyzer",
    "LeaderboardBuilder",
    "FailureCategory",
    "DatasetItem",
    "BenchmarkResult",
    "BenchmarkSummaryMetrics",
    "LeaderboardEntry",
    "Leaderboard",
    "BenchmarkReport",
    "BenchmarkSession",
]

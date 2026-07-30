"""
evaluation/benchmarking package.
"""

from .golden_sample_manager import GoldenSampleManager, run_golden_regression_suite
from .historical_store import HistoricalStore
from .interfaces import IHistoricalStore, IRegressionRule
from .regression_detector import (
    DimensionRegressionRule,
    FailureRateSpikeRule,
    OverallScoreDropRule,
    PerformanceRegressionRule,
    RegressionDetector,
    detect_regressions,
)

__all__ = [
    "DimensionRegressionRule",
    "FailureRateSpikeRule",
    "GoldenSampleManager",
    "HistoricalStore",
    "IHistoricalStore",
    "IRegressionRule",
    "OverallScoreDropRule",
    "PerformanceRegressionRule",
    "RegressionDetector",
    "detect_regressions",
    "run_golden_regression_suite",
]

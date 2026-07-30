"""
evaluation/config.py
====================

Configuration settings and constants for the Pipeline Validation & Quality Evaluation Framework (PVQEF).
"""

from pathlib import Path
from typing import Literal

from modules.config import (
    EVAL_GOLDEN_DIR,
    EVAL_HISTORY_PATH,
    EVAL_LOG_PATH,
    EVAL_RUNS_DIR,
    PROJECT_ROOT,
)

#: Default scoring weights across all fourteen evaluation dimensions.
EVAL_QUALITY_WEIGHTS: dict[str, float] = {
    "prompt_adherence": 0.15,
    "face_preservation": 0.10,
    "object_preservation": 0.10,
    "background_quality": 0.05,
    "composition": 0.10,
    "text_readability": 0.10,
    "color_harmony": 0.05,
    "visual_consistency": 0.05,
    "attractiveness": 0.10,
    "determinism": 0.05,
    "inline_qa": 0.10,
    "runtime_performance": 0.05,
    "memory_profile": 0.0,
    "failure_rate": 0.0,
}

#: Thresholds for quality dimensions (minimum score to pass hard-gate if applicable)
EVAL_DIMENSION_THRESHOLDS: dict[str, float] = {
    "prompt_adherence": 0.60,
    "face_preservation": 0.50,
    "object_preservation": 0.50,
    "background_quality": 0.50,
    "composition": 0.50,
    "text_readability": 0.60,
    "color_harmony": 0.50,
    "visual_consistency": 0.50,
    "attractiveness": 0.50,
    "determinism": 0.90,
    "inline_qa": 0.60,
    "runtime_performance": 0.0,
    "memory_profile": 0.0,
    "failure_rate": 0.0,
}

#: Regression detector rolling window size (number of prior runs to average)
EVAL_REGRESSION_WINDOW: int = 5

#: Maximum allowed drop in overall weighted score before flagging a regression
EVAL_REGRESSION_SCORE_DELTA: float = 0.05

#: Maximum allowed increase in failure rate (skipped/total)
EVAL_REGRESSION_FAILURE_DELTA: float = 0.10

#: Maximum allowed latency multiplier (e.g. 1.5x of baseline duration)
EVAL_REGRESSION_LATENCY_MULTIPLIER: float = 1.5

#: Threshold for divergence between Module 7 inline score and independent PVQEF score
EVAL_QA_DIVERGENCE_THRESHOLD: float = 0.20

#: Minimum SSIM score for determinism check
EVAL_DETERMINISM_SSIM_THRESHOLD: float = 0.95

#: Repeat count for generation determinism checks (7.11)
EVAL_DETERMINISM_REPEAT_COUNT: int = 3

#: Maximum concurrent GPU creators
EVAL_MAX_CONCURRENCY: int = 1

#: Maximum concurrent CPU-only tasks
EVAL_CPU_ONLY_CONCURRENCY: int = 4

#: Evaluation profile ("full" vs "lightweight")
EVAL_PROFILE: Literal["full", "lightweight"] = "full"

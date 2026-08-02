"""
observability/config.py
======================

Configuration settings and constants for the Pipeline Observability & Root Cause Engine (PORCE).
"""

from pathlib import Path

from modules.config import (
    LOG_DIR,
    OBS_FACTS_DIR,
    OBS_GENERATION_TRACES_DIR,
    OBS_LOG_PATH,
    OBS_REPORTS_DIR,
    OBS_TRACES_DIR,
    PROJECT_ROOT,
)

__all__ = [
    "PROJECT_ROOT",
    "LOG_DIR",
    "OBS_LOG_PATH",
    "OBS_TRACES_DIR",
    "OBS_REPORTS_DIR",
    "OBS_GENERATION_TRACES_DIR",
    "OBS_FACTS_DIR",
    "OBS_FACTS_VERSION",
    "OBS_RULE_REGISTRY_ENABLED",
    "OBS_MASK_IOU_THRESHOLD",
    "OBS_PERCEPTUAL_HASH_THRESHOLD",
    "OBS_LOG_CORRELATION_WINDOW_HOURS",
]

#: Schema version for extracted facts.
OBS_FACTS_VERSION: str = "1.0.0"

#: Per-rule enablement overrides. Empty dict means all registered rules are enabled.
OBS_RULE_REGISTRY_ENABLED: dict[str, bool] = {}

#: Intersection-over-Union threshold for detecting layer mask overlaps.
OBS_MASK_IOU_THRESHOLD: float = 0.5

#: Perceptual hash distance threshold for background regeneration check.
OBS_PERCEPTUAL_HASH_THRESHOLD: float = 10.0

#: Correlation window in hours (default 7 days / 168 hours).
OBS_LOG_CORRELATION_WINDOW_HOURS: int = 168


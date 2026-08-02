"""
optimization/config.py
======================

Configuration constants for the Thumbnail Quality Optimization Layer.
"""

from pathlib import Path
from modules.config import PROJECT_ROOT

#: Master toggle for the optimization loop (Phase 2)
OPTIMIZATION_LOOP_ENABLED: bool = True

#: Safe win margin required for candidate to beat original (scale 0-1)
OPTIMIZATION_MIN_WIN_MARGIN: float = 0.05

#: Minimum structural similarity (SSIM) threshold before flagging over-editing
OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY: float = 0.40

#: Maximum allowed identity drift before flagging subject identity loss
OPTIMIZATION_MAX_IDENTITY_DRIFT: float = 0.30

#: Maximum number of optimization retry attempts
OPTIMIZATION_MAX_RETRIES: int = 2

#: Enable feedback prior provider into Decision Engine / Strategy Engine
OPTIMIZATION_FEEDBACK_ENABLED: bool = False

#: Deep PVQEF scoring on candidates (default False uses inline QualityAssuranceReport)
OPTIMIZATION_DEEP_SCORE: bool = False

#: Run acceptance gate in report-only mode without withholding thumbnails
OPTIMIZATION_ACCEPTANCE_REPORT_ONLY: bool = False

#: Directory for storing sharded outcome records
OPTIMIZATION_OUTCOMES_DIR: Path = PROJECT_ROOT / "data" / "optimization" / "outcomes"

#: Minimum outcome sample size before feedback priors become active
OPTIMIZATION_FEEDBACK_MIN_SAMPLE_SIZE: int = 5

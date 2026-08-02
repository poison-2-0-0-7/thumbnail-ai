"""
outcome_recorder.py
===================

Persists decision-linked optimization outcome records to sharded atomic storage.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from optimization.config import OPTIMIZATION_OUTCOMES_DIR
from optimization.exceptions import FeedbackError


class OptimizationOutcome(BaseModel):
    """Decision-linked optimization outcome audit record."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    niche: str = "general"
    decisions_applied: list[str] = []
    hook_type_used: Optional[str] = None
    candidate_strategy_name: str = "default"
    beats_original: bool
    delta: float
    per_dimension_delta: dict[str, float] = {}
    recorded_at: str


class OutcomeRecorder:
    """Atomic recorder for OptimizationOutcome records."""

    def __init__(self, storage_dir: Path = OPTIMIZATION_OUTCOMES_DIR) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def record(self, outcome: OptimizationOutcome) -> Path:
        """
        Atomically write outcome record to sharded directory.
        """
        video_dir = self.storage_dir / outcome.video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        target_path = video_dir / "outcome.json"

        try:
            payload = outcome.model_dump_json(indent=2)
            # Atomic write pattern: write to temp file, flush/sync, atomic rename
            fd, tmp_path_str = tempfile.mkstemp(dir=video_dir, prefix="tmp_outcome_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

            # Rename temp file to target_path atomically (on Windows replace if exists)
            os.replace(tmp_path_str, target_path)
            logger.info("Persisted OptimizationOutcome for {vid} to {path}", vid=outcome.video_id, path=target_path)
            return target_path
        except Exception as exc:
            logger.error("Failed to record outcome for {vid}: {exc}", vid=outcome.video_id, exc=exc)
            raise FeedbackError(f"Failed to record outcome for {outcome.video_id}: {exc}") from exc

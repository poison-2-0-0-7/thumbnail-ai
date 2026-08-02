"""
observability/reporting/persistence.py
=======================================

Atomic persistence and loading for canonical RootCauseReport objects in PORCE.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from observability.config import OBS_TRACES_DIR
from observability.exceptions import RootCausePersistenceError
from observability.reporting.interfaces import IRootCausePersistence
from observability.reporting.models import RootCauseReport


class RootCausePersistence(IRootCausePersistence):
    """
    Handles atomic persistence and deterministic loading for RootCauseReport objects.
    Stores canonical report at data/observability/traces/{video_id}/root_cause_report.json.
    """

    def __init__(self, traces_dir: Optional[Path] = None) -> None:
        self.traces_dir = traces_dir or OBS_TRACES_DIR

    def get_target_path(self, video_id: str) -> Path:
        """Return target file path for root cause report."""
        return self.traces_dir / video_id / "root_cause_report.json"

    def save(self, report: RootCauseReport) -> Path:
        """
        Atomically save RootCauseReport to disk using temp-file-then-replace pattern.
        """
        target_path = self.get_target_path(report.video_id)
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        json_str = report.model_dump_json(indent=2)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=target_dir, prefix=".rc_report_tmp_", suffix=".json"
        )
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            Path(tmp_path).replace(target_path)
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RootCausePersistenceError(
                f"Failed to save RootCauseReport for video_id={report.video_id}: {exc}"
            ) from exc

        logger.debug(
            "Saved RootCauseReport for video_id={vid} to {path}",
            vid=report.video_id,
            path=target_path,
        )
        return target_path

    def load_file(self, path: Path) -> RootCauseReport:
        """
        Load and validate RootCauseReport from a specific path.
        """
        if not path.is_file():
            raise RootCausePersistenceError(f"RootCauseReport file not found at {path}")
        try:
            content = path.read_text(encoding="utf-8")
            return RootCauseReport.model_validate_json(content)
        except Exception as exc:
            raise RootCausePersistenceError(
                f"Failed to parse RootCauseReport at {path}: {exc}"
            ) from exc

    def load(self, video_id: str) -> Optional[RootCauseReport]:
        """
        Load RootCauseReport for video_id if present on disk.
        """
        target_path = self.get_target_path(video_id)
        if not target_path.is_file():
            return None
        try:
            return self.load_file(target_path)
        except Exception as exc:
            logger.warning(
                "Could not load RootCauseReport at {path}: {exc}",
                path=target_path,
                exc=exc,
            )
            return None


class RootCauseLoader:
    """
    Helper class for loading RootCauseReport objects by video_id or path.
    """

    def __init__(self, persistence: Optional[RootCausePersistence] = None) -> None:
        self.persistence = persistence or RootCausePersistence()

    def load_by_video_id(self, video_id: str) -> Optional[RootCauseReport]:
        """Load RootCauseReport for video_id."""
        return self.persistence.load(video_id)

    def load_by_path(self, path: Path | str) -> RootCauseReport:
        """Load RootCauseReport from specific path."""
        return self.persistence.load_file(Path(path))

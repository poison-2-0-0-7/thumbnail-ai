"""
observability/trace/log_correlator.py
=====================================

Scans module log files, correlates log entries by video_id, and orders them chronologically.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules import config
from observability.interfaces import ILogCorrelator
from observability.models import LogLineRef

# Regex for matching Loguru formatted log lines:
# e.g. "2026-07-09 12:03:29 | INFO     | module1.csv_reader | Message text"
# or "2026-07-09 12:03:29.123 | INFO | module7 | Message text"
LOGURU_LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s*\|\s*(?P<level>[A-Z]+)"
    r"\s*\|\s*(?P<module>[^\s\|]+)"
    r"\s*\|\s*(?P<message>.*)$"
)


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Attempt to parse common log timestamp strings into UTC datetime."""
    ts_str = ts_str.replace("T", " ").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class LogCorrelator(ILogCorrelator):
    """
    Correlates log entries across all module log files for a specific video_id.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_files: Optional[dict[str, Path]] = None,
    ) -> None:
        self.log_dir = log_dir or config.LOG_DIR
        if log_files is not None:
            self.log_files = log_files
        else:
            self.log_files = {
                "module1": config.MODULE1_LOG_PATH,
                "module2": config.MODULE2_LOG_PATH,
                "module3": config.MODULE3_LOG_PATH,
                "module4": config.MODULE4_LOG_PATH,
                "module5": config.MODULE5_LOG_PATH,
                "module5.5": getattr(config, "MODULE55_LOG_PATH", config.LOG_DIR / "module5_5.log"),
                "module6": config.MODULE6_LOG_PATH,
                "module6.5": getattr(config, "MODULE65_LOG_PATH", config.LOG_DIR / "module6_5.log"),
                "module8": getattr(config, "MODULE8_LOG_PATH", config.LOG_DIR / "module8.log"),
                "module5_spec": config.MODULE5_LOG_PATH,
                "module9": getattr(config, "MODULE9_LOG_PATH", config.LOG_DIR / "module9.log"),
                "module10": getattr(config, "MODULE10_LOG_PATH", config.LOG_DIR / "module10.log"),
                "module10.5": getattr(config, "MODULE10_5_LOG_PATH", config.LOG_DIR / "module10_5.log"),
                "module7": config.MODULE7_LOG_PATH,
                "evaluation": getattr(config, "EVAL_LOG_PATH", config.LOG_DIR / "evaluation.log"),
                "comfyui": getattr(config, "COMFYUI_PROCESS_LOG_PATH", config.LOG_DIR / "comfyui_process.log"),
            }

    def correlate(self, video_id: str) -> list[LogLineRef]:
        """
        Scan all configured log files for line matches containing video_id, parse them,
        and return them sorted chronologically.
        """
        matched_lines: list[tuple[datetime, int, LogLineRef]] = []
        global_order_counter = 0

        for module_tag, log_path in self.log_files.items():
            if not log_path.exists() or not log_path.is_file():
                continue

            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_idx, line in enumerate(f, start=1):
                        if video_id not in line:
                            continue

                        raw_line = line.rstrip("\r\n")
                        match = LOGURU_LINE_PATTERN.match(raw_line)
                        if match:
                            ts_str = match.group("timestamp")
                            level = match.group("level")
                            mod_name = match.group("module")
                            msg = match.group("message")
                        else:
                            ts_str = None
                            level = "INFO"
                            mod_name = module_tag
                            msg = raw_line

                        dt = _parse_timestamp(ts_str) if ts_str else None
                        sort_key_dt = dt or datetime.min.replace(tzinfo=timezone.utc)

                        log_ref = LogLineRef(
                            file_path=str(log_path.resolve()),
                            line_number=line_idx,
                            timestamp=ts_str,
                            level=level,
                            module=mod_name,
                            message=msg,
                            raw_line=raw_line,
                        )

                        global_order_counter += 1
                        matched_lines.append((sort_key_dt, global_order_counter, log_ref))
            except Exception:
                continue

        # Sort chronologically by timestamp, maintaining file line insertion order for ties
        matched_lines.sort(key=lambda x: (x[0], x[1]))
        return [item[2] for item in matched_lines]

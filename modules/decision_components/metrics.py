"""
metrics.py
==========

Structured metrics recorder for Module 9 runs. Appends JSONL rows to module9_metrics.jsonl.
Mirrors image_generator.py MetricsCollector pattern.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from modules.config import MODULE9_METRICS_PATH


class MetricsCollector:
    """Collects and appends per-video decision engine execution metrics."""

    def __init__(self, metrics_path: Path = MODULE9_METRICS_PATH) -> None:
        self.metrics_path = Path(metrics_path)

    def record_run(
        self,
        video_id: str,
        duration_seconds: float,
        candidate_count: int,
        resolved_count: int,
        llm_adjudications_count: int,
        conflicts_resolved_count: int,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one structured metrics record to metrics JSONL file."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "video_id": video_id,
            "duration_seconds": round(duration_seconds, 4),
            "candidate_count": candidate_count,
            "resolved_count": resolved_count,
            "llm_adjudications_count": llm_adjudications_count,
            "conflicts_resolved_count": conflicts_resolved_count,
            "status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record.update(extra)

        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

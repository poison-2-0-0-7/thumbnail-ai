"""LearningFeedbackStore: Append-only JSONL store for candidate selection feedback and score history."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class LearningFeedbackRecord(BaseModel):
    """Append-only learning feedback record for candidate evaluation history."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    winning_candidate_index: int
    algorithmic_winner_index: int
    winning_strategy: str
    was_overridden: bool = False
    score_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LearningFeedbackStore:
    """Append-only store using JSONL persistence with atomic writes for selection feedback."""

    def __init__(self, store_path: Path | None = None) -> None:
        from config import PROJECT_ROOT
        self.store_path = store_path or (PROJECT_ROOT / "data" / "learning_feedback.jsonl")

    def record_feedback(
        self,
        video_id: str,
        winning_candidate_index: int,
        algorithmic_winner_index: int,
        winning_strategy: str,
        was_overridden: bool,
        score_breakdown: dict[str, dict[str, float]] | None = None,
    ) -> LearningFeedbackRecord:
        """
        Record selection feedback entry into JSONL store.

        Returns:
            Recorded LearningFeedbackRecord.
        """
        if score_breakdown:
            score_breakdown = {str(k): v for k, v in score_breakdown.items()}

        record = LearningFeedbackRecord(
            video_id=video_id,
            winning_candidate_index=winning_candidate_index,
            algorithmic_winner_index=algorithmic_winner_index,
            winning_strategy=winning_strategy,
            was_overridden=was_overridden,
            score_breakdown=score_breakdown or {},
        )

        self._append_atomic(record)
        logger.info(
            "Recorded candidate learning feedback for video_id={vid}: winner={winner} (strat='{strat}', overridden={overridden})",
            vid=video_id,
            winner=winning_candidate_index,
            strat=winning_strategy,
            overridden=was_overridden,
        )
        return record

    def _append_atomic(self, record: LearningFeedbackRecord) -> None:
        """Append record atomically to JSONL file."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json() + "\n"

        # Read existing contents if file exists
        existing_content = ""
        if self.store_path.is_file():
            try:
                existing_content = self.store_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Error reading existing learning feedback file: {exc}", exc=exc)

        # Write to temporary file then replace atomically
        tmp_path = self.store_path.parent / f".learning_feedback.tmp.{time.time_ns()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(existing_content + line)
            tmp_path.replace(self.store_path)
        except Exception as exc:
            logger.error("Failed atomic append to learning feedback store: {exc}", exc=exc)
            if tmp_path.exists():
                tmp_path.unlink()

    def read_records(self) -> list[LearningFeedbackRecord]:
        """Read all historical feedback records from JSONL file."""
        if not self.store_path.is_file():
            return []

        records = []
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(LearningFeedbackRecord.model_validate_json(line))
        except Exception as exc:
            logger.error("Failed reading learning feedback records: {exc}", exc=exc)
        return records

"""HumanReviewWorkspace: Human Review Mode, manual override handling, and timeout fallback."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from typing import Any, Sequence
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from models import CandidateScore, CandidateStrategy


class ManualSelectionRecord(BaseModel):
    """Record of a manual selection override decision by a human reviewer."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    selected_candidate_index: int
    algorithmic_winner_index: int
    override_reason: str = ""
    reviewer_id: str = "human_operator"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HumanReviewWorkspace:
    """Workspace manager for human review mode, manual candidate selection, and review persistence."""

    def __init__(
        self,
        enabled: bool = False,
        timeout_seconds: float = 300.0,
        workspace_dir: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        from config import PROJECT_ROOT
        self.workspace_dir = workspace_dir or (PROJECT_ROOT / "data" / "human_review")

    def process_review(
        self,
        video_id: str,
        algorithmic_winner: tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]],
        all_candidates: Sequence[tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]]],
        candidate_scores: Sequence[CandidateScore],
        manual_override_index: int | None = None,
        override_reason: str = "",
    ) -> tuple[tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]], ManualSelectionRecord | None]:
        """
        Process candidate selection with optional human review override.

        If disabled or timed out or no override provided, returns algorithmic winner.
        If manual_override_index is valid and provided, returns the human selected candidate.
        """
        alg_winner_idx = algorithmic_winner[0]

        if not self.enabled:
            logger.debug("Human review mode disabled; returning algorithmic winner idx={idx}", idx=alg_winner_idx)
            return algorithmic_winner, None

        target_dir = self.workspace_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing manual selection record file
        selection_file = target_dir / "manual_selection.json"
        if manual_override_index is None and selection_file.is_file():
            try:
                data = json.loads(selection_file.read_text(encoding="utf-8"))
                manual_override_index = data.get("selected_candidate_index")
                override_reason = data.get("override_reason", "Loaded from persisted review workspace")
            except Exception as exc:
                logger.warning("Failed to load existing manual selection record: {exc}", exc=exc)

        # If manual override specified, validate and select candidate
        if manual_override_index is not None:
            candidate_map = {c[0]: c for c in all_candidates}
            if manual_override_index in candidate_map:
                selected_cand = candidate_map[manual_override_index]
                rec = ManualSelectionRecord(
                    video_id=video_id,
                    selected_candidate_index=manual_override_index,
                    algorithmic_winner_index=alg_winner_idx,
                    override_reason=override_reason or "Manual human selection override",
                    reviewer_id="human_operator",
                )
                self.persist_record(target_dir, rec)
                logger.info(
                    "Human review override applied for video_id={vid}: selected candidate {sel} over algorithmic winner {alg}",
                    vid=video_id,
                    sel=manual_override_index,
                    alg=alg_winner_idx,
                )
                return selected_cand, rec
            else:
                logger.warning(
                    "Requested manual override candidate index {idx} not found; falling back to algorithmic winner",
                    idx=manual_override_index,
                )

        # Timeout fallback to algorithmic winner
        rec = ManualSelectionRecord(
            video_id=video_id,
            selected_candidate_index=alg_winner_idx,
            algorithmic_winner_index=alg_winner_idx,
            override_reason="Default algorithmic selection (no human override)",
            reviewer_id="algorithmic_fallback",
        )
        self.persist_record(target_dir, rec)
        return algorithmic_winner, rec

    def persist_record(self, workspace_dir: Path, record: ManualSelectionRecord) -> Path:
        """Persist manual selection record atomically."""
        workspace_dir.mkdir(parents=True, exist_ok=True)
        target_path = workspace_dir / "manual_selection.json"
        tmp_path = workspace_dir / f"manual_selection.tmp.{time.time_ns()}"

        content = record.model_dump_json(indent=2)
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(target_path)
        return target_path

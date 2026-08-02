"""
observability/facts/persistence.py
===================================

Atomic persistence and loading for FactCollection objects in PORCE.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from observability.config import OBS_FACTS_DIR
from observability.exceptions import FactPersistenceError
from observability.facts.interfaces import IFactPersistence
from observability.facts.models import FactCollection
from observability.facts.serializer import FactSerializer


class FactPersistence(IFactPersistence):
    """
    Handles atomic disk persistence and loading for FactCollection objects.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or OBS_FACTS_DIR
        self.serializer = FactSerializer()

    def get_target_path(self, video_id: str) -> Path:
        """Return target filepath for facts for a given video_id."""
        return self.output_dir / video_id / "facts.json"

    def save(self, collection: FactCollection) -> Path:
        """
        Atomically write FactCollection to disk using temp-file-then-replace pattern.
        """
        target_path = self.get_target_path(collection.video_id)
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        json_str = self.serializer.serialize(collection)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=target_dir, prefix=".facts_tmp_", suffix=".json"
        )
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            Path(tmp_path).replace(target_path)
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise FactPersistenceError(
                f"Failed to save FactCollection for video_id={collection.video_id}: {exc}"
            ) from exc

        logger.debug(
            "Saved FactCollection for video_id={vid} to {path}",
            vid=collection.video_id,
            path=target_path,
        )
        return target_path

    def load_file(self, path: Path) -> FactCollection:
        """
        Load and validate FactCollection from a specific path.
        """
        if not path.is_file():
            raise FactPersistenceError(f"Facts file not found at path: {path}")
        try:
            content = path.read_text(encoding="utf-8")
            return self.serializer.deserialize(content)
        except Exception as exc:
            raise FactPersistenceError(f"Failed to load facts from {path}: {exc}") from exc

    def load(self, video_id: str) -> Optional[FactCollection]:
        """
        Load FactCollection for video_id if present on disk.
        """
        target_path = self.get_target_path(video_id)
        if not target_path.is_file():
            return None
        try:
            return self.load_file(target_path)
        except Exception as exc:
            logger.warning(
                "Could not load FactCollection at {path}: {exc}",
                path=target_path,
                exc=exc,
            )
            return None


class FactLoader:
    """
    Helper class for loading facts by video_id or path.
    """

    def __init__(self, persistence: Optional[FactPersistence] = None) -> None:
        self.persistence = persistence or FactPersistence()

    def load_by_video_id(self, video_id: str) -> Optional[FactCollection]:
        """Load FactCollection for video_id."""
        return self.persistence.load(video_id)

    def load_by_path(self, path: Path | str) -> FactCollection:
        """Load FactCollection from specific path."""
        return self.persistence.load_file(Path(path))

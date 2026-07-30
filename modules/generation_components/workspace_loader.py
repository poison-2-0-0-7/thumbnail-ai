"""
workspace_loader.py
===================

Loads persisted CompositionWorkspace JSON files for Module 7 Phase 3.
"""

from __future__ import annotations

from pathlib import Path

from config import MODULE7_COMPOSITION_WORKSPACE_ROOT, MODULE7_LOG_PATH
from generation_components.interfaces import ICompositionWorkspaceLoader
from module7_exceptions import WorkspaceNotFoundError
from models import CompositionWorkspace
from loguru import logger

def _configure_logger() -> None:
    """Ensure Loguru sink is configured for Module 7."""
    try:
        logger.add(
            MODULE7_LOG_PATH,
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            enqueue=True,
        )
    except ValueError:
        pass

_configure_logger()


class CompositionWorkspaceLoader(ICompositionWorkspaceLoader):
    """Loads a persisted CompositionWorkspace from disk by video_id."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir else MODULE7_COMPOSITION_WORKSPACE_ROOT

    def load(self, video_id: str) -> CompositionWorkspace:
        """
        Load a CompositionWorkspace from disk.

        Args:
            video_id: Video identifier.

        Returns:
            CompositionWorkspace model instance.

        Raises:
            WorkspaceNotFoundError: If the workspace file does not exist or is invalid.
        """
        if not video_id or not video_id.strip():
            raise WorkspaceNotFoundError("video_id must not be empty.")

        video_id = video_id.strip()
        candidates = [
            self._root_dir / video_id / "composition.json",
            self._root_dir / f"{video_id}.json",
            self._root_dir / video_id / "workspace.json",
        ]

        target_file: Path | None = None
        for candidate in candidates:
            if candidate.is_file():
                target_file = candidate
                break

        if target_file is None:
            logger.warning("CompositionWorkspace for video_id={video_id} not found in {root_dir}", video_id=video_id, root_dir=self._root_dir)
            raise WorkspaceNotFoundError(f"Composition workspace not found for video_id '{video_id}' under {self._root_dir}")

        try:
            content = target_file.read_text(encoding="utf-8")
            workspace = CompositionWorkspace.model_validate_json(content)
            if workspace.video_id != video_id:
                raise WorkspaceNotFoundError(
                    f"Workspace video_id mismatch: expected '{video_id}', got '{workspace.video_id}'"
                )
            logger.info("Successfully loaded CompositionWorkspace for video_id={video_id} from {path}", video_id=video_id, path=target_file)
            return workspace
        except WorkspaceNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to parse CompositionWorkspace for video_id={video_id}: {exc}", video_id=video_id, exc=exc)
            raise WorkspaceNotFoundError(f"Failed to parse CompositionWorkspace for '{video_id}': {exc}") from exc

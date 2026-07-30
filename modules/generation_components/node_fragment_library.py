"""
node_fragment_library.py
========================

Discovers and loads declarative ComfyUI workflow graph fragments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import MODULE7_FRAGMENT_LIBRARY_DIR, MODULE7_LOG_PATH
from generation_components.interfaces import INodeFragmentLibrary
from module7_exceptions import WorkflowBuildError
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


class NodeFragmentLibrary(INodeFragmentLibrary):
    """Discovers and loads declarative graph fragments from the filesystem."""

    def __init__(self, fragment_dir: Path | None = None) -> None:
        self._fragment_dir = Path(fragment_dir) if fragment_dir else MODULE7_FRAGMENT_LIBRARY_DIR

    def discover(self) -> list[Path]:
        """
        Discover available fragment JSON files in deterministic sorted order.

        Returns:
            Sorted list of file Paths.
        """
        if not self._fragment_dir.exists():
            return []
        return sorted(self._fragment_dir.glob("*.json"))

    def load(self, fragment_id: str) -> dict[str, Any]:
        """
        Load a fragment definition by fragment_id.

        Args:
            fragment_id: Fragment identifier (e.g. 'controlnet_depth').

        Returns:
            Fragment definition dict.

        Raises:
            WorkflowBuildError: If fragment file is missing or malformed.
        """
        if not fragment_id or not fragment_id.strip():
            raise WorkflowBuildError("fragment_id must not be empty.")

        clean_id = fragment_id.strip()
        fragment_file = self._fragment_dir / f"{clean_id}.json"

        if not fragment_file.is_file():
            raise WorkflowBuildError(f"Fragment file not found for '{clean_id}' at '{fragment_file}'.")

        try:
            content = fragment_file.read_text(encoding="utf-8")
            data = json.loads(content)
            self._validate_fragment_schema(clean_id, data)
            logger.info("Loaded fragment '{fragment_id}' from {path}", fragment_id=clean_id, path=fragment_file)
            return data
        except WorkflowBuildError:
            raise
        except Exception as exc:
            logger.error("Failed to parse fragment '{fragment_id}': {exc}", fragment_id=clean_id, exc=exc)
            raise WorkflowBuildError(f"Failed to parse fragment '{clean_id}': {exc}") from exc

    def _validate_fragment_schema(self, fragment_id: str, data: dict[str, Any]) -> None:
        """Validate required top-level fragment keys."""
        if not isinstance(data, dict):
            raise WorkflowBuildError(f"Fragment '{fragment_id}' content must be a JSON object.")

        attach = data.get("_attach")
        if not isinstance(attach, dict):
            raise WorkflowBuildError(f"Fragment '{fragment_id}' missing '_attach' section.")

        point = attach.get("point")
        if not point or not isinstance(point, str):
            raise WorkflowBuildError(f"Fragment '{fragment_id}' missing valid '_attach.point'.")

        out_node = attach.get("output_node")
        if out_node is None:
            raise WorkflowBuildError(f"Fragment '{fragment_id}' missing '_attach.output_node'.")

        graph = data.get("graph")
        if not isinstance(graph, dict):
            raise WorkflowBuildError(f"Fragment '{fragment_id}' missing valid 'graph' object.")

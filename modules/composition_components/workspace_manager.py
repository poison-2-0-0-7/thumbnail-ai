"""
workspace_manager.py
====================

Handles sharded directory creation, atomic persistence, workspace deserialization,
and hash-based cache resume for Module 10 Asset Composer.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Optional

from composition_components.interfaces import IWorkspaceManager
from composition_exceptions import (
    WorkspacePersistenceError,
    WorkspaceValidationError,
)
from config import (
    COMPOSITION_MANIFEST_FILENAME,
    COMPOSITION_WORKSPACE_ROOT,
)
from models import CompositionWorkspace, WorkspaceMetadata, WorkspaceStatistics


class WorkspaceManager(IWorkspaceManager):
    """Manager for filesystem persistence, loading, purging, and resuming workspaces."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self._root_dir = Path(root_dir) if root_dir else COMPOSITION_WORKSPACE_ROOT

    def target_dir(self, video_id: str) -> Path:
        """Get the workspace directory path for a video_id."""
        return self._root_dir / video_id

    def persist(self, workspace: CompositionWorkspace) -> Path:
        """
        Atomically write CompositionWorkspace and sharded files to disk.

        Args:
            workspace: Validated CompositionWorkspace instance.

        Returns:
            Path to the persisted workspace directory.
        """
        target = self.target_dir(workspace.video_id)
        self._root_dir.mkdir(parents=True, exist_ok=True)

        # Use temp directory in same filesystem root for atomic move
        temp_dir = Path(tempfile.mkdtemp(dir=self._root_dir, prefix=f".tmp_{workspace.video_id}_"))

        try:
            # 1. Create subdirectories
            subdirs = [
                "background",
                "foreground",
                "people",
                "objects",
                "text",
                "effects",
                "masks",
                "depth",
                "lighting",
                "layers",
            ]
            for sub in subdirs:
                (temp_dir / sub).mkdir(parents=True, exist_ok=True)

            # 2. Write per-layer JSON snippets
            for layer in workspace.layers:
                layer_file = temp_dir / "layers" / f"{layer.layer_id}.json"
                layer_file.write_text(layer.model_dump_json(indent=2), encoding="utf-8")

            # 3. Write composition.json
            comp_file = temp_dir / "composition.json"
            comp_file.write_text(workspace.model_dump_json(indent=2), encoding="utf-8")

            # 4. Write metadata.json
            meta_data = {
                "metadata": workspace.metadata.model_dump(),
                "statistics": workspace.statistics.model_dump(),
            }
            meta_file = temp_dir / "metadata.json"
            meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

            # 5. Write workspace_manifest.json
            manifest_data = {
                "video_id": workspace.video_id,
                "created_at": workspace.metadata.created_at,
                "vre_source_hash": workspace.metadata.vre_source_hash,
                "redesign_spec_hash": workspace.metadata.redesign_spec_hash,
                "prompt_package_hash": workspace.metadata.prompt_package_hash,
                "engine_version": workspace.metadata.engine_version,
                "validated": True,
                "layers_written": True,
            }
            manifest_file = temp_dir / COMPOSITION_MANIFEST_FILENAME
            manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

            # 6. Atomic replacement of target directory
            if target.exists():
                shutil.rmtree(target)
            temp_dir.rename(target)
            return target

        except Exception as exc:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise WorkspacePersistenceError(
                f"Failed to persist workspace for '{workspace.video_id}': {exc}"
            ) from exc

    def load(self, video_id: str) -> CompositionWorkspace:
        """Load CompositionWorkspace from disk."""
        target = self.target_dir(video_id)
        comp_file = target / "composition.json"

        if not comp_file.is_file():
            raise WorkspaceValidationError(
                f"Workspace file not found for video_id '{video_id}' at '{comp_file}'."
            )

        try:
            content = comp_file.read_text(encoding="utf-8")
            return CompositionWorkspace.model_validate_json(content)
        except Exception as exc:
            raise WorkspaceValidationError(
                f"Failed to parse workspace JSON for '{video_id}': {exc}"
            ) from exc

    def resume(
        self, video_id: str, expected_hashes: dict[str, str]
    ) -> Optional[CompositionWorkspace]:
        """
        Check if a valid cached workspace exists matching expected hashes.

        Args:
            video_id: Video identifier.
            expected_hashes: Dict containing 'vre_source_hash', 'redesign_spec_hash', 'prompt_package_hash'.

        Returns:
            CompositionWorkspace if valid cache hit, else None.
        """
        target = self.target_dir(video_id)
        manifest_file = target / COMPOSITION_MANIFEST_FILENAME

        if not manifest_file.is_file():
            return None

        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not manifest.get("validated", False):
                return None

            for key in ("vre_source_hash", "redesign_spec_hash", "prompt_package_hash"):
                expected = expected_hashes.get(key)
                if expected and manifest.get(key, "").lower() != expected.lower():
                    return None

            return self.load(video_id)
        except Exception:
            return None

    def purge(self, video_id: str) -> bool:
        """Purge workspace directory for video_id."""
        target = self.target_dir(video_id)
        if target.exists():
            shutil.rmtree(target)
            return True
        return False

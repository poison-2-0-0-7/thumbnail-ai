"""
asset_writer.py
===============

Atomic persistence for Module 8 image assets and JSON sidecar files.
Composed over vre_components.asset_writer.AssetWriter.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from modules.asset_extraction_components.interfaces import IAssetExtractionWriter
from modules.asset_extraction_exceptions import AssetWriteError
from vre_components.asset_writer import AssetWriter as VREAssetWriter
from vre_exceptions import AssetWriteError as VREAssetWriteError


class AssetExtractionWriter(IAssetExtractionWriter):
    """Atomic image and JSON sidecar writer for Module 8 asset extraction."""

    def __init__(self) -> None:
        self._vre_writer = VREAssetWriter()

    def write_image(self, array: np.ndarray, destination_path: Path) -> bool:
        """Write image array atomically via VRE writer composition."""
        dest = Path(destination_path)
        try:
            return self._vre_writer.write_image(array, dest)
        except VREAssetWriteError as exc:
            raise AssetWriteError(f"Could not write asset image to {dest}: {exc}") from exc
        except Exception as exc:
            raise AssetWriteError(f"Unexpected error writing asset image to {dest}: {exc}") from exc

    def write_json_sidecar(self, data: Any, destination_path: Path) -> bool:
        """Write JSON sidecar dictionary or list atomically."""
        dest = Path(destination_path)
        self._ensure_directory_exists(dest.parent)
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=dest.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
            ) as handle:
                tmp_name = handle.name
                json.dump(data, handle, indent=2, ensure_ascii=False)

            Path(tmp_name).replace(dest)
            logger.debug("Wrote JSON sidecar -> {path}", path=dest)
            return True
        except OSError as exc:
            raise AssetWriteError(f"Could not write JSON sidecar to {dest}: {exc}") from exc
        finally:
            if tmp_name is not None:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass

    def purge_directory(self, target_dir: Path) -> bool:
        """Purge asset directory atomically."""
        try:
            return self._vre_writer.purge_directory(target_dir)
        except VREAssetWriteError as exc:
            raise AssetWriteError(f"Could not purge asset directory {target_dir}: {exc}") from exc

    @staticmethod
    def _ensure_directory_exists(path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AssetWriteError(f"Could not create asset directory {path}: {exc}") from exc

"""Atomic asset persistence for Module 6.5 VRE."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from vre_exceptions import AssetWriteError
from vre_components.interfaces import IAssetWriter


class AssetWriter(IAssetWriter):
    """Persist generated VRE image assets with temp-file replacement."""

    def write_image(self, array: np.ndarray, destination_path: Path) -> bool:
        self._atomic_write(array, Path(destination_path))
        logger.debug("Wrote VRE asset -> {path}", path=destination_path)
        return True

    def purge_directory(self, target_dir: Path) -> bool:
        target = Path(target_dir)
        try:
            if target.exists():
                shutil.rmtree(target)
            return True
        except OSError as exc:
            raise AssetWriteError(f"Could not purge VRE asset directory {target}: {exc}") from exc

    @staticmethod
    def _ensure_directory_exists(path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AssetWriteError(f"Could not create VRE asset directory {path}: {exc}") from exc

    def _atomic_write(self, array: np.ndarray, path: Path) -> None:
        if not isinstance(array, np.ndarray) or array.size == 0:
            raise AssetWriteError(f"Cannot write empty or non-array VRE asset to {path}")

        self._ensure_directory_exists(path.parent)
        suffix = path.suffix or ".png"
        tmp_name: str | None = None
        try:
            contiguous = np.ascontiguousarray(array)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=suffix, delete=False
            ) as handle:
                tmp_name = handle.name
            ok = cv2.imwrite(tmp_name, contiguous)
            if not ok:
                raise AssetWriteError(f"OpenCV could not encode VRE asset {path}")
            Path(tmp_name).replace(path)
        except AssetWriteError:
            raise
        except OSError as exc:
            raise AssetWriteError(f"Could not write VRE asset to {path}: {exc}") from exc
        finally:
            if tmp_name is not None:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass

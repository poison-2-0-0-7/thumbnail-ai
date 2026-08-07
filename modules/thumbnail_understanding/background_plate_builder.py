"""
background_plate_builder.py
===========================

Clean Background Plate Builder (Phase 14).

Constructs and manages clean background plate representations for redesign editing,
ensuring subjects are cleanly separated from the background without subject bleed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from models import AssetExtractionManifest
from thumbnail_understanding.schemas import LayerCategory, SceneLayer


class BackgroundPlateBuilder:
    """Builds clean background plate references for layer-aware generation."""

    @classmethod
    def resolve_background_plate(
        cls,
        source_thumbnail_path: str,
        asset_manifest: Optional[AssetExtractionManifest] = None,
    ) -> tuple[str, bool]:
        """
        Return (background_plate_path, is_reconstructed).
        """
        if asset_manifest and asset_manifest.scene and asset_manifest.scene.background_only:
            bg_path = asset_manifest.scene.background_only.file_path
            if Path(bg_path).is_file():
                return bg_path, True

        # Fallback to source thumbnail path as base plate if no isolated background asset exists yet
        return source_thumbnail_path, False

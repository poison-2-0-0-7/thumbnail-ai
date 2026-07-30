"""
asset_registry.py
=================

Indexes VRE assets from VisualReferenceManifest, resolves assets by role,
and verifies filesystem integrity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from composition_components.interfaces import IAssetRegistry
from models import AssetMetadata, VisualReferenceManifest


class AssetRegistry(IAssetRegistry):
    """Read-only registry for indexing and verifying Visual Reference Engine (VRE) assets."""

    def __init__(self, manifest: Optional[VisualReferenceManifest] = None) -> None:
        self._assets: dict[str, AssetMetadata] = {}
        if manifest is not None:
            self.index(manifest)

    def index(self, manifest: VisualReferenceManifest) -> dict[str, AssetMetadata]:
        """Index all non-None assets from a VisualReferenceManifest."""
        self._assets.clear()
        for key, asset in manifest.assets.items():
            if asset is not None:
                self._assets[key] = asset
        return dict(self._assets)

    def resolve(self, role: str) -> Optional[AssetMetadata]:
        """Resolve AssetMetadata by asset key/role."""
        return self._assets.get(role)

    def verify_integrity(self) -> list[str]:
        """
        Verify that indexed asset files exist and their SHA-256 checksums match metadata.

        Returns:
            List of invalid asset_ids (empty if all assets are verified).
        """
        invalid_assets: list[str] = []
        for asset_id, asset in self._assets.items():
            path = Path(asset.file_path)
            if not path.is_file():
                invalid_assets.append(asset_id)
                continue

            try:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if digest.lower() != asset.checksum.lower():
                    invalid_assets.append(asset_id)
            except Exception:
                invalid_assets.append(asset_id)

        return invalid_assets

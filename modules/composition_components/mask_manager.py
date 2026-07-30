"""
mask_manager.py
===============

Binds VRE mask references (face_mask, object_mask) and handles mask feathering
configuration.
"""

from __future__ import annotations

from typing import Optional

from composition_components.interfaces import IAssetRegistry, IMaskManager
from config import COMPOSITION_TEXT_FEATHER_PX
from models import LayerRole, MaskReference


class MaskManager(IMaskManager):
    """Manager for resolving and configuring layer mask references from VRE registry."""

    def bind(self, registry: IAssetRegistry, role: LayerRole) -> Optional[MaskReference]:
        """
        Bind VRE mask path and checksum for a layer role.

        Args:
            registry: IAssetRegistry instance.
            role: LayerRole enum.

        Returns:
            MaskReference if a matching mask asset is found, else None.
        """
        mask_asset = None
        feather_px = 0

        if role == LayerRole.PERSON:
            mask_asset = registry.resolve("face_mask")
            feather_px = COMPOSITION_TEXT_FEATHER_PX
        elif role == LayerRole.OBJECT:
            mask_asset = registry.resolve("object_mask")

        if mask_asset is None:
            return None

        return MaskReference(
            mask_path=mask_asset.file_path,
            mask_checksum=mask_asset.checksum,
            feather_px=feather_px,
            source="vre",
        )

    def feather(self, mask_ref: MaskReference, feather_px: int) -> MaskReference:
        """Return a copy of MaskReference with updated feathering."""
        return mask_ref.model_copy(update={"feather_px": max(0, feather_px)})

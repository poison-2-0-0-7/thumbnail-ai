"""
test_mask_manager.py
====================

Unit tests for MaskManager component in Module 10 Asset Composer.
"""

from __future__ import annotations

from composition_components.asset_registry import AssetRegistry
from composition_components.mask_manager import MaskManager
from models import AssetMetadata, LayerRole, MaskReference, VisualReferenceManifest


def test_mask_manager_bind_and_feather():
    valid_hash = "a" * 64
    manifest = VisualReferenceManifest(
        video_id="test_vid",
        source_image_path="source.jpg",
        source_hash=valid_hash,
        created_at="2026-07-30T00:00:00Z",
        assets={
            "face_mask": AssetMetadata(
                asset_type="face_mask",
                file_path="data/visual_references/test_vid/face_mask.png",
                checksum=valid_hash,
                resolution=(200, 200),
            ),
            "object_mask": AssetMetadata(
                asset_type="object_mask",
                file_path="data/visual_references/test_vid/object_mask.png",
                checksum=valid_hash,
                resolution=(150, 150),
            ),
        },
    )

    registry = AssetRegistry(manifest)
    manager = MaskManager()

    # Person mask
    person_mask = manager.bind(registry, LayerRole.PERSON)
    assert person_mask is not None
    assert person_mask.mask_path.endswith("face_mask.png")
    assert person_mask.feather_px > 0

    # Object mask
    obj_mask = manager.bind(registry, LayerRole.OBJECT)
    assert obj_mask is not None
    assert obj_mask.mask_path.endswith("object_mask.png")
    assert obj_mask.feather_px == 0

    # Background mask (None)
    bg_mask = manager.bind(registry, LayerRole.BACKGROUND)
    assert bg_mask is None

    # Feather modification
    feathered = manager.feather(person_mask, feather_px=12)
    assert feathered.feather_px == 12

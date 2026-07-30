"""
test_composition_validator.py
==============================

Unit tests for CompositionValidator component in Module 10 Asset Composer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from composition_components.composition_validator import CompositionValidator
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    LayerDecision,
    LayerGroup,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    MaskReference,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


@pytest.fixture
def temp_mask_file(tmp_path: Path):
    mask_file = tmp_path / "face_mask.png"
    mask_bytes = b"fake_mask_data"
    mask_file.write_bytes(mask_bytes)
    digest = hashlib.sha256(mask_bytes).hexdigest()
    return str(mask_file), digest


def test_composition_validator_success(temp_mask_file):
    mask_path, mask_checksum = temp_mask_file
    valid_hash = "a" * 64

    layer = CompositionLayer(
        layer_id="l1",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            mask=MaskReference(mask_path=mask_path, mask_checksum=mask_checksum),
            transform=LayerTransform(),
            z_index=10,
        ),
    )

    workspace = CompositionWorkspace(
        video_id="test_vid",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[layer],
        groups=[LayerGroup(group_id="g1", role=LayerRole.PERSON, layer_ids=["l1"])],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=1, kept=1),
        metadata=WorkspaceMetadata(
            video_id="test_vid",
            created_at="2026-07-30T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        ),
    )

    validator = CompositionValidator()
    errors = validator.validate(workspace)
    assert errors == []


def test_composition_validator_failures(tmp_path: Path):
    valid_hash = "b" * 64

    # Layer with non-existent source file and corrupted mask checksum
    bad_layer = CompositionLayer(
        layer_id="l1",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            source_path=str(tmp_path / "non_existent.png"),
            mask=MaskReference(
                mask_path=str(tmp_path / "non_existent_mask.png"),
                mask_checksum=valid_hash,
            ),
            transform=LayerTransform(),
            z_index=10,
        ),
    )

    workspace = CompositionWorkspace(
        video_id="test_vid",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[bad_layer],
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=99),  # mismatch!
        metadata=WorkspaceMetadata(
            video_id="test_vid",
            created_at="2026-07-30T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        ),
    )

    validator = CompositionValidator()
    errors = validator.validate(workspace)
    assert len(errors) >= 3
    assert any("non-existent source_path" in err for err in errors)
    assert any("non-existent mask_path" in err for err in errors)
    assert any("does not match layer count" in err for err in errors)

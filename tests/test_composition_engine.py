"""
test_composition_engine.py
===========================

Unit and integration tests for Module 10 AssetComposer orchestrator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from composition_engine import AssetComposer
from composition_exceptions import AssetRegistryError, CompositionInputInvalidError
from models import (
    AssetMetadata,
    BoundingBox,
    ColorDirection,
    CompositionWorkspace,
    GenerationBundle,
    GenerationParameters,
    LayoutDirection,
    ModelSettings,
    PromptPackage,
    QualityParameters,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    VisualReferenceManifest,
)
from visual_reference_engine import VisualReferenceEngine


class FakeVREEngine(VisualReferenceEngine):
    """Fake VRE engine returning pre-crafted VisualReferenceManifest fixtures."""

    def __init__(self, manifest: VisualReferenceManifest) -> None:
        self._manifest = manifest

    def prepare_assets(
        self, video_id: str, thumbnail_path: str, options: dict | None = None
    ) -> VisualReferenceManifest:
        return self._manifest


@pytest.fixture
def temp_workspace_files(tmp_path: Path):
    face_file = tmp_path / "creator_face.png"
    face_bytes = b"fake_face_image"
    face_file.write_bytes(face_bytes)
    face_hash = hashlib.sha256(face_bytes).hexdigest()

    mask_file = tmp_path / "face_mask.png"
    mask_bytes = b"fake_mask_image"
    mask_file.write_bytes(mask_bytes)
    mask_hash = hashlib.sha256(mask_bytes).hexdigest()

    bg_file = tmp_path / "background.png"
    bg_bytes = b"fake_bg_image"
    bg_file.write_bytes(bg_bytes)
    bg_hash = hashlib.sha256(bg_bytes).hexdigest()

    manifest = VisualReferenceManifest(
        video_id="I-bnBd5lCew",
        source_image_path=str(face_file),
        source_hash="a" * 64,
        created_at="2026-07-30T00:00:00Z",
        assets={
            "creator_face": AssetMetadata(
                asset_type="creator_face",
                file_path=str(face_file),
                checksum=face_hash,
                resolution=(200, 200),
            ),
            "face_mask": AssetMetadata(
                asset_type="face_mask",
                file_path=str(mask_file),
                checksum=mask_hash,
                resolution=(200, 200),
            ),
            "background": AssetMetadata(
                asset_type="background",
                file_path=str(bg_file),
                checksum=bg_hash,
                resolution=(1280, 720),
            ),
        },
    )
    return manifest


def test_asset_composer_end_to_end_prepare_generation_workspace(
    tmp_path: Path, temp_workspace_files
):
    video_id = "I-bnBd5lCew"
    fake_vre = FakeVREEngine(temp_workspace_files)

    composer = AssetComposer(storage_root=tmp_path, vre_engine=fake_vre)
    bundle = composer.prepare_generation_workspace(video_id, options={"use_cache": False})

    assert isinstance(bundle, GenerationBundle)
    assert bundle.video_id == video_id
    assert bundle.canvas.width == 1280
    assert bundle.canvas.height == 720
    assert "background" in bundle.reference_image_paths
    assert "person" in bundle.reference_image_paths
    assert "person" in bundle.mask_paths

    # Verify persisted workspace on disk
    ws_dir = tmp_path / video_id
    assert ws_dir.is_dir()
    assert (ws_dir / "composition.json").is_file()
    assert (ws_dir / "workspace_manifest.json").is_file()

    # Verify load_workspace
    loaded_ws = composer.load_workspace(video_id)
    assert loaded_ws.video_id == video_id
    assert len(loaded_ws.layers) >= 2


def test_asset_composer_determinism(tmp_path: Path, temp_workspace_files):
    video_id = "I-bnBd5lCew"
    fake_vre = FakeVREEngine(temp_workspace_files)

    composer = AssetComposer(storage_root=tmp_path, vre_engine=fake_vre)
    ws1 = composer.compose_workspace(video_id, options={"use_cache": False})
    ws2 = composer.compose_workspace(video_id, options={"use_cache": False})

    # Compare serialized layer order and layer details
    dump1 = ws1.model_dump_json(
        exclude={"metadata": {"created_at"}, "duration_seconds": True}, indent=2
    )
    dump2 = ws2.model_dump_json(
        exclude={"metadata": {"created_at"}, "duration_seconds": True}, indent=2
    )

    assert dump1 == dump2



def test_asset_composer_missing_upstream_spec(tmp_path: Path):
    composer = AssetComposer(storage_root=tmp_path)
    with pytest.raises(CompositionInputInvalidError):
        composer.compose_workspace("non_existent_video_id")


def test_asset_composer_clean_workspace(tmp_path: Path, temp_workspace_files):
    video_id = "I-bnBd5lCew"
    fake_vre = FakeVREEngine(temp_workspace_files)

    composer = AssetComposer(storage_root=tmp_path, vre_engine=fake_vre)
    composer.prepare_generation_workspace(video_id, options={"use_cache": False})

    ws_dir = tmp_path / video_id
    assert ws_dir.is_dir()

    purged = composer.clean_workspace(video_id)
    assert purged is True
    assert not ws_dir.exists()

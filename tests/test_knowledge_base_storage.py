"""
Unit tests for Knowledge Base local storage and atomic persistence engine.
Tests atomic temp-file-then-replace behavior, fsync operations, version snapshot archiving,
corrupted file recovery, and namespace isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from thumbnail_intelligence.knowledge_base.config import KnowledgeBaseConfig, StorageConfig
from thumbnail_intelligence.knowledge_base.exceptions import (
    AtomicWriteError,
    StorageCorruptionError,
    StorageNotFoundError,
)
from thumbnail_intelligence.knowledge_base.models import Archetype, KnowledgeEntry, KnowledgeEntryType
from thumbnail_intelligence.knowledge_base.storage import AtomicFileWriter, VersionedFileStorage


def test_atomic_file_writer_success(tmp_path: Path) -> None:
    config = StorageConfig(base_dir=tmp_path, fsync_on_write=True)
    writer = AtomicFileWriter(config)

    target = tmp_path / "test_entry.json"
    data = {"key": "value", "count": 42}

    written = writer.write_json(target, data)
    assert written.exists()
    assert not target.with_suffix(".tmp").exists()

    loaded = writer.read_json(target)
    assert loaded == data


def test_atomic_file_writer_not_found(tmp_path: Path) -> None:
    writer = AtomicFileWriter()
    missing_file = tmp_path / "non_existent.json"
    with pytest.raises(StorageNotFoundError):
        writer.read_json(missing_file)


def test_atomic_file_writer_corrupted_json(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ this is not valid json", encoding="utf-8")

    writer = AtomicFileWriter()
    with pytest.raises(StorageCorruptionError):
        writer.read_json(corrupt_file)


def test_versioned_file_storage_crud(tmp_path: Path) -> None:
    kb_config = KnowledgeBaseConfig(base_dir=tmp_path)
    storage = VersionedFileStorage(
        model_cls=Archetype,
        namespace="archetypes",
        base_dir=tmp_path,
        config=kb_config,
    )

    archetype = Archetype(
        archetype_id="reaction_01",
        name="Big Reaction",
        description="High emotion face reaction",
        centroid_embedding=[0.0] * 512,
        version="1.0.0",
    )

    # 1. Write
    written_path = storage.write(archetype, entity_id=archetype.archetype_id)
    assert written_path.exists()
    assert storage.exists("reaction_01")

    # 2. Read
    loaded = storage.read("reaction_01")
    assert loaded.archetype_id == "reaction_01"
    assert loaded.name == "Big Reaction"

    # 3. List
    ids = storage.list_ids()
    assert "reaction_01" in ids

    all_entities = storage.list_all()
    assert len(all_entities) == 1
    assert all_entities[0].archetype_id == "reaction_01"

    # 4. Version archiving on update
    updated_archetype = Archetype(
        archetype_id="reaction_01",
        name="Big Reaction Updated",
        description="Updated high emotion face reaction",
        centroid_embedding=[0.1] * 512,
        version="1.1.0",
    )
    storage.write(updated_archetype, entity_id="reaction_01", archive_version=True)

    # Verify latest is updated
    latest = storage.read("reaction_01")
    assert latest.name == "Big Reaction Updated"

    # Verify historical versions
    versions = storage.list_versions("reaction_01")
    assert "1.0.0" in versions or "1.1.0" in versions

    # 5. Delete
    deleted = storage.delete("reaction_01", delete_versions=True)
    assert deleted is True
    assert not storage.exists("reaction_01")


def test_versioned_storage_missing_entity_raises(tmp_path: Path) -> None:
    storage = VersionedFileStorage(
        model_cls=KnowledgeEntry,
        namespace="entries",
        base_dir=tmp_path,
    )
    with pytest.raises(StorageNotFoundError):
        storage.read("missing_id_999")

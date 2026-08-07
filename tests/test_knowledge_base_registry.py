"""
Unit tests for Knowledge Registry.
Tests CRUD lifecycle operations, duplicate error guards, version querying, filter predicates,
and pluggable IndexHook callback triggers for future embedding indexing.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from thumbnail_intelligence.knowledge_base.config import KnowledgeBaseConfig
from thumbnail_intelligence.knowledge_base.exceptions import (
    DuplicateEntryError,
    EntryNotFoundError,
    InvalidQueryError,
)
from thumbnail_intelligence.knowledge_base.models import Archetype
from thumbnail_intelligence.knowledge_base.registry import IndexHook, KnowledgeRegistry
from thumbnail_intelligence.knowledge_base.storage import VersionedFileStorage


class MockIndexHook(IndexHook[Archetype]):
    """Mock vector index hook verifying index lifecycle callbacks."""

    def __init__(self) -> None:
        self.registered_entries: list[Archetype] = []
        self.updated_entries: list[Archetype] = []
        self.removed_ids: list[str] = []

    def on_registered(self, entry: Archetype) -> None:
        self.registered_entries.append(entry)

    def on_updated(self, entry: Archetype) -> None:
        self.updated_entries.append(entry)

    def on_removed(self, entry_id: str) -> None:
        self.removed_ids.append(entry_id)


def test_registry_registration_and_lookup(tmp_path: Path) -> None:
    kb_config = KnowledgeBaseConfig(base_dir=tmp_path)
    storage = VersionedFileStorage(
        model_cls=Archetype,
        namespace="archetypes",
        base_dir=tmp_path,
        config=kb_config,
    )
    registry = KnowledgeRegistry(
        model_cls=Archetype,
        id_field="archetype_id",
        storage=storage,
    )

    hook = MockIndexHook()
    registry.register_index_hook(hook)

    archetype = Archetype(
        archetype_id="curiosity_gap",
        name="Curiosity Gap",
        description="Teasing outcome to stimulate click intent",
        centroid_embedding=[0.0] * 512,
        typical_emotion="curiosity",
    )

    # 1. Register
    reg = registry.register(archetype)
    assert reg.archetype_id == "curiosity_gap"
    assert registry.exists("curiosity_gap")
    assert registry.count() == 1
    assert len(hook.registered_entries) == 1
    assert hook.registered_entries[0].archetype_id == "curiosity_gap"

    # 2. Duplicate registration without allow_overwrite should fail
    with pytest.raises(DuplicateEntryError):
        registry.register(archetype, allow_overwrite=False)

    # 3. Lookup & Get
    found = registry.lookup("curiosity_gap")
    assert found is not None
    assert found.name == "Curiosity Gap"

    got = registry.get("curiosity_gap")
    assert got.name == "Curiosity Gap"

    # 4. Missing lookup & get
    assert registry.lookup("non_existent") is None
    with pytest.raises(EntryNotFoundError):
        registry.get("non_existent")

    # 5. Update
    updated = Archetype(
        archetype_id="curiosity_gap",
        name="Curiosity Gap Advanced",
        description="Refined curiosity gap template",
        centroid_embedding=[0.1] * 512,
        typical_emotion="surprise",
    )
    registry.update(updated)
    assert registry.get("curiosity_gap").name == "Curiosity Gap Advanced"
    assert len(hook.updated_entries) == 1

    # 6. List and filter
    results = registry.list(filter_fn=lambda a: a.typical_emotion == "surprise")
    assert len(results) == 1

    # 7. Remove
    removed = registry.remove("curiosity_gap")
    assert removed is True
    assert not registry.exists("curiosity_gap")
    assert "curiosity_gap" in hook.removed_ids


def test_registry_pagination_and_sorting(tmp_path: Path) -> None:
    registry = KnowledgeRegistry(
        model_cls=Archetype,
        id_field="archetype_id",
    )

    for i in range(5):
        arch = Archetype(
            archetype_id=f"arch_{i:02d}",
            name=f"Archetype {i}",
            description=f"Description {i}",
            centroid_embedding=[0.0] * 512,
            example_count=i * 10,
        )
        registry.register(arch)

    assert registry.count() == 5

    # Test limit and offset
    page1 = registry.list(limit=2, offset=0, sort_by="archetype_id")
    assert len(page1) == 2
    assert page1[0].archetype_id == "arch_00"

    page2 = registry.list(limit=2, offset=2, sort_by="archetype_id")
    assert len(page2) == 2
    assert page2[0].archetype_id == "arch_02"

    # Test reverse sorting by example_count
    sorted_rev = registry.list(sort_by="example_count", reverse=True)
    assert sorted_rev[0].example_count == 40
    assert sorted_rev[-1].example_count == 0

    # Test invalid offset
    with pytest.raises(InvalidQueryError):
        registry.list(offset=-1)

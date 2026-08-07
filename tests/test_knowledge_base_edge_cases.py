"""
Comprehensive unit tests for edge cases, error conditions, and corner scenarios
across all Knowledge Base Foundation subsystems.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from pydantic import ValidationError as PydanticValidationError

from thumbnail_intelligence.knowledge_base.config import (
    KnowledgeBaseConfig,
    StorageConfig,
    VersioningConfig,
)
from thumbnail_intelligence.knowledge_base.exceptions import (
    ConstraintValidationError,
    DuplicateEntryError,
    EntryNotFoundError,
    EvidenceValidationError,
    InvalidQueryError,
    KnowledgeBaseError,
    MigrationError,
    RegistryError,
    SchemaValidationError,
    StorageError,
    StorageNotFoundError,
    UnsupportedVersionError,
    ValidationError,
    VersioningError,
)
from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    BrandConstraint,
    ChannelProfile,
    CreatorProfile,
    DesignPattern,
    DesignReason,
    DesignReasonType,
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
    IdentityConstraint,
    KnowledgeEntry,
    KnowledgeEntryType,
    VisualPattern,
)
from thumbnail_intelligence.knowledge_base.registry import KnowledgeRegistry
from thumbnail_intelligence.knowledge_base.storage import VersionedFileStorage
from thumbnail_intelligence.knowledge_base.validation import (
    ConstraintValidator,
    EvidenceValidator,
    ModelValidator,
)
from thumbnail_intelligence.knowledge_base.versioning import MigrationRegistry, SemVer


def test_base_kb_error_structure() -> None:
    err = KnowledgeBaseError(
        message="Test knowledge error",
        error_code="KB_TEST_CODE",
        context={"entity_id": "ent_123", "subsystem": "storage"},
    )
    d = err.to_dict()
    assert d["error_code"] == "KB_TEST_CODE"
    assert d["message"] == "Test knowledge error"
    assert d["context"]["entity_id"] == "ent_123"
    assert "KB_TEST_CODE" in str(err)


def test_storage_edge_cases_and_backups(tmp_path: Path) -> None:
    cfg = KnowledgeBaseConfig(
        base_dir=tmp_path,
        storage=StorageConfig(base_dir=tmp_path, enable_backups=True),
    )
    storage = VersionedFileStorage(
        model_cls=Archetype,
        namespace="test_archetypes",
        base_dir=tmp_path,
        config=cfg,
    )

    # Write version 1.0.0
    arch_v1 = Archetype(
        archetype_id="test_arch",
        name="Version 1",
        description="Initial version",
        version="1.0.0",
    )
    storage.write(arch_v1, entity_id="test_arch", archive_version=True)

    # Write version 1.1.0 (should archive 1.0.0)
    arch_v2 = Archetype(
        archetype_id="test_arch",
        name="Version 2",
        description="Second version",
        version="1.1.0",
    )
    storage.write(arch_v2, entity_id="test_arch", archive_version=True)

    # Read active version
    active = storage.read("test_arch")
    assert active.name == "Version 2"

    # Read historical archived version
    hist = storage.read_version("test_arch", "1.0.0")
    assert hist.name == "Version 1"

    # Reading non-existent version should raise StorageNotFoundError
    with pytest.raises(StorageNotFoundError):
        storage.read_version("test_arch", "9.9.9")

    # Deleting non-existent entity returns False
    assert storage.delete("non_existent_id") is False

    # Sanitizing empty ID raises StorageError
    with pytest.raises(StorageError):
        storage.read("   ")


def test_storage_clear(tmp_path: Path) -> None:
    storage = VersionedFileStorage(
        model_cls=VisualPattern,
        namespace="test_patterns",
        base_dir=tmp_path,
    )
    vp = VisualPattern(
        pattern_id="pat_1",
        name="Pattern 1",
        description="Description 1",
    )
    storage.write(vp, entity_id="pat_1")
    assert storage.exists("pat_1")

    storage.clear()
    assert not storage.exists("pat_1")
    assert len(storage.list_ids()) == 0


def test_registry_edge_cases(tmp_path: Path) -> None:
    registry = KnowledgeRegistry(
        model_cls=CreatorProfile,
        id_field="creator_id",
    )

    # Registering wrong type raises RegistryError
    with pytest.raises(RegistryError):
        registry.register("invalid_string")  # type: ignore

    # Updating non-existent entity raises EntryNotFoundError
    creator = CreatorProfile(
        creator_id="creator_unregistered",
        display_name="Unregistered",
    )
    with pytest.raises(EntryNotFoundError):
        registry.update(creator)

    # Looking up blank or None returns None
    assert registry.lookup("") is None
    assert registry.lookup(None) is None  # type: ignore

    # Removing blank or non-existent returns False
    assert registry.remove("") is False
    assert registry.remove("missing_id") is False

    # Clearing registry
    registry.register(creator)
    assert registry.count() == 1
    registry.clear()
    assert registry.count() == 0


def test_versioning_and_migration_edge_cases() -> None:
    reg = MigrationRegistry()

    # Registering failing migration hook
    def faulty_hook(data: dict) -> dict:
        raise ValueError("Intentional transform failure")

    reg.register("FaultyModel", "1.0.0", "2.0.0", faulty_hook)
    with pytest.raises(MigrationError):
        reg.migrate("FaultyModel", {"schema_version": "1.0.0"}, "2.0.0")

    # Querying unsupported version
    with pytest.raises(UnsupportedVersionError):
        reg.get_migration_path("FaultyModel", "1.0.0", "5.0.0")


def test_model_invalid_timestamps_and_versions() -> None:
    # Invalid ISO timestamp
    with pytest.raises(PydanticValidationError):
        EvidenceReference(
            source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
            source_id="elem_0",
            created_at="not-a-timestamp",
        )

    # Invalid schema version format
    with pytest.raises(PydanticValidationError):
        EvidenceReference(
            source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
            source_id="elem_0",
            schema_version="invalid-version-format",
        )


def test_evidence_validator_bulk() -> None:
    ref1 = EvidenceReference(
        source_type=EvidenceSourceType.BRAND_RULE,
        source_id="rule_01",
    )
    ref2 = EvidenceReference(
        source_type=EvidenceSourceType.PSYCHOLOGY_DRIVER,
        source_id="driver_01",
    )
    reason1 = DesignReason(
        reason_id="reason_01",
        claim="Preserve color palette",
        reason_type=DesignReasonType.BRAND_CONSISTENCY,
        evidence=[ref1],
    )
    reason2 = DesignReason(
        reason_id="reason_02",
        claim="High curiosity hook",
        reason_type=DesignReasonType.AUDIENCE_PSYCHOLOGY,
        evidence=[ref2],
    )

    EvidenceValidator.validate_design_reasons([reason1, reason2])

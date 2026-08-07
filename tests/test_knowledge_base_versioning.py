"""
Unit tests for Knowledge Base semantic versioning and migration engine.
Tests SemVer parsing, ordering, backward compatibility, multi-step migration paths,
and error handling on invalid schema versions.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.knowledge_base.exceptions import (
    MigrationError,
    UnsupportedVersionError,
    VersioningError,
)
from thumbnail_intelligence.knowledge_base.versioning import MigrationRegistry, SemVer


def test_semver_parsing_and_comparison() -> None:
    v1 = SemVer.parse("1.0.0")
    v1_1 = SemVer.parse("1.1.0")
    v2 = SemVer.parse("2.0.0")

    assert v1.major == 1 and v1.minor == 0 and v1.patch == 0
    assert v1 < v1_1 < v2
    assert v1 == SemVer.parse("1.0.0")
    assert str(v1) == "1.0.0"

    # Compatibility check
    assert v1_1.is_compatible(v1) is True
    assert v1.is_compatible(v1_1) is False
    assert v2.is_compatible(v1) is False


def test_semver_invalid_strings() -> None:
    with pytest.raises(VersioningError):
        SemVer.parse("invalid_version")

    with pytest.raises(VersioningError):
        SemVer.parse(123)  # type: ignore


def test_migration_registry_single_and_multi_step() -> None:
    reg = MigrationRegistry()

    def v1_to_v1_1(data: dict) -> dict:
        d = dict(data)
        d["new_field"] = "default_value"
        return d

    def v1_1_to_v2_0(data: dict) -> dict:
        d = dict(data)
        d["renamed_field"] = d.pop("old_field", "renamed_default")
        return d

    reg.register("TestModel", "1.0.0", "1.1.0", v1_to_v1_1)
    reg.register("TestModel", "1.1.0", "2.0.0", v1_1_to_v2_0)

    assert reg.has_migration("TestModel", "1.0.0", "2.0.0") is True
    assert reg.has_migration("TestModel", "1.0.0", "3.0.0") is False

    initial_payload = {
        "schema_version": "1.0.0",
        "old_field": "legacy_value",
    }

    migrated = reg.migrate("TestModel", initial_payload, target_version="2.0.0")
    assert migrated["schema_version"] == "2.0.0"
    assert migrated["new_field"] == "default_value"
    assert migrated["renamed_field"] == "legacy_value"
    assert "old_field" not in migrated


def test_migration_registry_unsupported_version_raises() -> None:
    reg = MigrationRegistry()
    with pytest.raises(UnsupportedVersionError):
        reg.migrate("UnknownModel", {"schema_version": "1.0.0"}, "2.0.0")

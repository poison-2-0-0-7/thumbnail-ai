"""
storage.py
==========

Production local JSON storage engine with atomic writes, versioned history,
schema validation, and migration hooks for the Thumbnail Intelligence Knowledge Base.

Follows the project's atomic write-then-replace convention (temp file + fsync + os.replace)
to guarantee zero corruption under crash or power-loss conditions without external database dependencies.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

from thumbnail_intelligence.knowledge_base.config import (
    DEFAULT_KB_DATA_DIR,
    KnowledgeBaseConfig,
    StorageConfig,
)
from thumbnail_intelligence.knowledge_base.exceptions import (
    AtomicWriteError,
    DeserializationError,
    StorageCorruptionError,
    StorageError,
    StorageIOError,
    StorageNotFoundError,
)
from thumbnail_intelligence.knowledge_base.serialization import KBSerializer
from thumbnail_intelligence.knowledge_base.versioning import (
    MigrationRegistry,
    global_migration_registry,
)

T = TypeVar("T", bound=BaseModel)


class AtomicFileWriter:
    """
    Handles atomic file writes using temp-file creation, data flushing,
    fsync, and atomic filesystem replacement.
    """

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self.config = config or StorageConfig()

    def write_json(self, target_path: Path, data: Dict[str, Any]) -> Path:
        """
        Atomically write dictionary as JSON to target_path.
        Guarantees that target_path is either completely written or untouched.
        """
        target_path = Path(target_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f"{target_path.name}{self.config.temp_suffix}")

        try:
            with open(temp_path, "w", encoding=self.config.encoding) as f:
                json.dump(
                    data,
                    f,
                    indent=self.config.indent,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                f.flush()
                if self.config.fsync_on_write:
                    os.fsync(f.fileno())

            # Atomic rename (POSIX atomic rename, Windows replace)
            os.replace(temp_path, target_path)

            # Ensure parent directory entry is synced if possible on POSIX
            if self.config.fsync_on_write and hasattr(os, "O_DIRECTORY"):
                try:
                    dir_fd = os.open(str(target_path.parent), os.O_RDONLY | os.O_DIRECTORY)
                    os.fsync(dir_fd)
                    os.close(dir_fd)
                except Exception:
                    pass

            return target_path

        except Exception as e:
            # Clean up orphan temp file on failure
            if temp_path.exists():
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise AtomicWriteError(
                message=f"Failed to atomically write JSON to '{target_path}': {e}",
                context={"target_path": str(target_path), "temp_path": str(temp_path), "error": str(e)},
            ) from e

    def read_json(self, target_path: Path) -> Dict[str, Any]:
        """
        Read and parse JSON from target_path with robust error handling.
        """
        target_path = Path(target_path).resolve()
        if not target_path.exists():
            raise StorageNotFoundError(
                message=f"Target storage file does not exist: {target_path}",
                context={"target_path": str(target_path)},
            )
        try:
            with open(target_path, "r", encoding=self.config.encoding) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise StorageCorruptionError(
                message=f"Corrupted or invalid JSON in '{target_path}': {e}",
                context={"target_path": str(target_path), "line": e.lineno, "col": e.colno},
            ) from e
        except OSError as e:
            raise StorageIOError(
                message=f"I/O error reading '{target_path}': {e}",
                context={"target_path": str(target_path), "error": str(e)},
            ) from e


class VersionedFileStorage(Generic[T]):
    """
    Typed, versioned local storage engine for Knowledge Base entities.
    Provides:
    - Namespace directory isolation under data/intelligence_kb/<namespace>/
    - Atomic JSON read/write operations
    - Historical version archiving on update
    - Dynamic schema validation and on-read migration hooks
    - Extensibility for custom serializers or index synchronization
    """

    def __init__(
        self,
        model_cls: Type[T],
        namespace: str,
        base_dir: Optional[Path] = None,
        config: Optional[KnowledgeBaseConfig] = None,
        migration_registry: Optional[MigrationRegistry] = None,
    ) -> None:
        self.model_cls = model_cls
        self.namespace = namespace.strip()
        self.kb_config = config or KnowledgeBaseConfig()
        self.base_dir = (base_dir or self.kb_config.base_dir) / self.namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.base_dir / "versions"
        if self.kb_config.storage.enable_backups:
            self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.writer = AtomicFileWriter(self.kb_config.storage)
        self.migration_registry = migration_registry or global_migration_registry

    def _sanitize_id(self, entity_id: str) -> str:
        """Sanitize identifier for safe filesystem path usage."""
        safe = "".join(c for c in entity_id if c.isalnum() or c in ("-", "_", ".")).strip()
        if not safe:
            raise StorageError(
                message=f"Invalid entity identifier for storage: '{entity_id}'",
                context={"entity_id": entity_id},
            )
        return safe

    def _get_path(self, entity_id: str) -> Path:
        """Get canonical JSON path for an entity."""
        safe_id = self._sanitize_id(entity_id)
        return self.base_dir / f"{safe_id}.json"

    def _get_version_path(self, entity_id: str, version: str) -> Path:
        """Get historical archived version path for an entity."""
        safe_id = self._sanitize_id(entity_id)
        safe_v = self._sanitize_id(version)
        return self.versions_dir / f"{safe_id}@{safe_v}.json"

    def write(self, entity: T, entity_id: str, archive_version: bool = True) -> Path:
        """
        Persist model instance atomically to disk.
        If existing version exists and archive_version is True, archive it in versions/ folder.
        """
        target_path = self._get_path(entity_id)

        # Archive existing version if it exists and backup enabled
        if target_path.exists() and archive_version and self.kb_config.storage.enable_backups:
            try:
                old_data = self.writer.read_json(target_path)
                old_ver = old_data.get("schema_version", old_data.get("version", "1.0.0"))
                version_path = self._get_version_path(entity_id, old_ver)
                self.writer.write_json(version_path, old_data)
            except Exception as e:
                logger.warning(f"Could not archive previous version of {entity_id}: {e}")

        # Serialize and write
        data = KBSerializer.to_dict(entity)
        written_path = self.writer.write_json(target_path, data)
        return written_path

    def read(self, entity_id: str) -> T:
        """
        Read and deserialize an entity by ID.
        Applies automated schema migration on read if the disk payload version is older.
        """
        target_path = self._get_path(entity_id)
        raw_data = self.writer.read_json(target_path)

        # Check schema version
        target_model_version = getattr(self.model_cls, "model_fields", {}).get("schema_version")
        expected_ver = (
            target_model_version.default
            if target_model_version and target_model_version.default
            else self.kb_config.versioning.default_schema_version
        )

        disk_ver = raw_data.get("schema_version", raw_data.get("version", "1.0.0"))

        if disk_ver != expected_ver and self.kb_config.versioning.auto_migrate_on_read:
            model_name = self.model_cls.__name__
            if self.migration_registry.has_migration(model_name, disk_ver, expected_ver):
                raw_data = self.migration_registry.migrate(model_name, raw_data, expected_ver)
                # Persist updated migrated model back
                self.writer.write_json(target_path, raw_data)

        try:
            return KBSerializer.from_dict(raw_data, self.model_cls)
        except DeserializationError as e:
            raise StorageCorruptionError(
                message=f"Failed to deserialize {self.model_cls.__name__} for id '{entity_id}': {e}",
                context={"entity_id": entity_id, "path": str(target_path), "error": str(e)},
            ) from e

    def read_version(self, entity_id: str, version: str) -> T:
        """
        Read a historical archived snapshot of an entity.
        """
        version_path = self._get_version_path(entity_id, version)
        if not version_path.exists():
            # Check current file if version matches
            current_entity = self.read(entity_id)
            curr_ver = getattr(current_entity, "schema_version", getattr(current_entity, "version", "1.0.0"))
            if curr_ver == version:
                return current_entity
            raise StorageNotFoundError(
                message=f"Version '{version}' for entity '{entity_id}' not found in archive",
                context={"entity_id": entity_id, "version": version, "path": str(version_path)},
            )
        raw_data = self.writer.read_json(version_path)
        return KBSerializer.from_dict(raw_data, self.model_cls)

    def delete(self, entity_id: str, delete_versions: bool = False) -> bool:
        """
        Delete an entity from disk.
        Returns True if deleted, False if entity did not exist.
        """
        target_path = self._get_path(entity_id)
        deleted = False
        if target_path.exists():
            try:
                target_path.unlink()
                deleted = True
            except OSError as e:
                raise StorageIOError(
                    message=f"Failed to delete entity file '{target_path}': {e}",
                    context={"entity_id": entity_id, "error": str(e)},
                ) from e

        if delete_versions and self.versions_dir.exists():
            safe_id = self._sanitize_id(entity_id)
            for v_file in self.versions_dir.glob(f"{safe_id}@*.json"):
                try:
                    v_file.unlink(missing_ok=True)
                except Exception:
                    pass

        return deleted

    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists on disk."""
        return self._get_path(entity_id).exists()

    def list_ids(self) -> List[str]:
        """List all entity identifiers in this storage namespace."""
        ids: List[str] = []
        for file_path in self.base_dir.glob("*.json"):
            if file_path.is_file() and not file_path.name.endswith(self.kb_config.storage.temp_suffix):
                ids.append(file_path.stem)
        return sorted(ids)

    def list_all(self) -> List[T]:
        """Load and return all entities in this namespace."""
        entities: List[T] = []
        for entity_id in self.list_ids():
            try:
                entities.append(self.read(entity_id))
            except Exception as e:
                logger.warning(f"Failed to load entity {entity_id} in {self.namespace}: {e}")
        return entities

    def list_versions(self, entity_id: str) -> List[str]:
        """List all available version tags for an entity."""
        versions: List[str] = []
        safe_id = self._sanitize_id(entity_id)

        # Include current active version
        if self.exists(entity_id):
            current = self.read(entity_id)
            curr_ver = getattr(current, "schema_version", getattr(current, "version", "1.0.0"))
            versions.append(curr_ver)

        if self.versions_dir.exists():
            for v_file in self.versions_dir.glob(f"{safe_id}@*.json"):
                # Extract version from filename: id@version.json
                parts = v_file.stem.split("@")
                if len(parts) == 2 and parts[1] not in versions:
                    versions.append(parts[1])

        return sorted(versions)

    def clear(self) -> None:
        """Clear all active entities and versions in this namespace."""
        for p in self.base_dir.glob("*.json"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        if self.versions_dir.exists():
            for p in self.versions_dir.glob("*.json"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

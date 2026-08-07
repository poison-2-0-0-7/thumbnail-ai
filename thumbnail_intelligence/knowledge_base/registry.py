"""
registry.py
===========

High-performance typed registry implementation for the Thumbnail Intelligence Knowledge Base.
Provides CRUD lifecycle operations, version querying, filter predicates, and pluggable index hooks
so future vector/embedding indexes can attach seamlessly without altering the public API.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Type, TypeVar, runtime_checkable

from loguru import logger
from pydantic import BaseModel

from thumbnail_intelligence.knowledge_base.exceptions import (
    DuplicateEntryError,
    EntryNotFoundError,
    InvalidQueryError,
    RegistryError,
)
from thumbnail_intelligence.knowledge_base.storage import VersionedFileStorage

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class IndexHook(Protocol[T]):
    """
    Pluggable index interface for vector, inverted, or embedding indexes.
    Enables future embedding indexes to hook into registry mutations without API changes.
    """

    def on_registered(self, entry: T) -> None:
        """Invoked immediately after an entry is registered in storage."""
        ...

    def on_updated(self, entry: T) -> None:
        """Invoked immediately after an entry is updated."""
        ...

    def on_removed(self, entry_id: str) -> None:
        """Invoked immediately after an entry is removed from the registry."""
        ...


class KnowledgeRegistry(Generic[T]):
    """
    Typed registry managing in-memory caching, disk storage persistence, and index synchronization
    for a specific Knowledge Base entity family.
    """

    def __init__(
        self,
        model_cls: Type[T],
        id_field: str,
        storage: Optional[VersionedFileStorage[T]] = None,
    ) -> None:
        self.model_cls = model_cls
        self.id_field = id_field.strip()
        self.storage = storage
        self._cache: Dict[str, T] = {}
        self._index_hooks: List[IndexHook[T]] = []
        self._initialized = False

    def register_index_hook(self, hook: IndexHook[T]) -> None:
        """
        Attach a vector or auxiliary index hook to this registry.
        Future embedding indexes attach here without altering registry signatures.
        """
        if hook not in self._index_hooks:
            self._index_hooks.append(hook)

    def _extract_id(self, entry: T) -> str:
        """Extract unique string identifier from an entry instance."""
        val = getattr(entry, self.id_field, None)
        if val is None or not str(val).strip():
            raise RegistryError(
                message=f"Entity does not have valid identifier in field '{self.id_field}'",
                context={"id_field": self.id_field, "entity": str(entry)},
            )
        return str(val).strip()

    def register(self, entry: T, allow_overwrite: bool = False) -> T:
        """
        Register a new entity in the registry and persist to storage.
        Triggers index hooks on registration.
        """
        if not isinstance(entry, self.model_cls):
            raise RegistryError(
                message=f"Expected instance of {self.model_cls.__name__}, got {type(entry).__name__}",
                context={"expected": self.model_cls.__name__, "received": type(entry).__name__},
            )

        entry_id = self._extract_id(entry)

        # Check for duplication
        if not allow_overwrite:
            if entry_id in self._cache or (self.storage and self.storage.exists(entry_id)):
                raise DuplicateEntryError(
                    message=f"Entry with id '{entry_id}' is already registered.",
                    context={"entry_id": entry_id, "model": self.model_cls.__name__},
                )

        # Persist if storage is configured
        if self.storage:
            self.storage.write(entry, entity_id=entry_id, archive_version=allow_overwrite)

        # Update in-memory cache
        self._cache[entry_id] = entry

        # Notify attached index hooks
        for hook in self._index_hooks:
            try:
                hook.on_registered(entry)
            except Exception as e:
                logger.warning(f"IndexHook on_registered failed for {entry_id}: {e}")

        return entry

    def lookup(self, entry_id: str, version: Optional[str] = None) -> Optional[T]:
        """
        Lookup an entity by ID and optional version tag.
        Returns None if not found.
        """
        if not entry_id or not str(entry_id).strip():
            return None
        safe_id = str(entry_id).strip()

        # If specific version requested
        if version and self.storage:
            try:
                return self.storage.read_version(safe_id, version)
            except Exception:
                return None

        # Check in-memory cache first
        if safe_id in self._cache:
            return self._cache[safe_id]

        # Load from disk storage
        if self.storage and self.storage.exists(safe_id):
            try:
                loaded = self.storage.read(safe_id)
                self._cache[safe_id] = loaded
                return loaded
            except Exception as e:
                logger.warning(f"Failed to read entry {safe_id} from storage: {e}")
                return None

        return None

    def get(self, entry_id: str, version: Optional[str] = None) -> T:
        """
        Get an entity by ID, raising EntryNotFoundError if missing.
        """
        res = self.lookup(entry_id, version=version)
        if res is None:
            raise EntryNotFoundError(
                message=f"Entry '{entry_id}' of type {self.model_cls.__name__} not found",
                context={"entry_id": entry_id, "version": version, "model": self.model_cls.__name__},
            )
        return res

    def update(self, entry: T) -> T:
        """
        Update an existing entity, maintaining version archives and notifying index hooks.
        """
        if not isinstance(entry, self.model_cls):
            raise RegistryError(
                message=f"Expected instance of {self.model_cls.__name__}, got {type(entry).__name__}",
                context={"expected": self.model_cls.__name__, "received": type(entry).__name__},
            )

        entry_id = self._extract_id(entry)

        # Check if entity exists
        if not self.exists(entry_id):
            raise EntryNotFoundError(
                message=f"Cannot update non-existent entry '{entry_id}'",
                context={"entry_id": entry_id, "model": self.model_cls.__name__},
            )

        # Persist updated entry and archive prior version
        if self.storage:
            self.storage.write(entry, entity_id=entry_id, archive_version=True)

        self._cache[entry_id] = entry

        # Notify attached index hooks
        for hook in self._index_hooks:
            try:
                hook.on_updated(entry)
            except Exception as e:
                logger.warning(f"IndexHook on_updated failed for {entry_id}: {e}")

        return entry

    def remove(self, entry_id: str, delete_storage: bool = True) -> bool:
        """
        Remove an entity from the registry and storage.
        Notifies index hooks upon removal.
        """
        if not entry_id:
            return False
        safe_id = str(entry_id).strip()

        existed = False
        if safe_id in self._cache:
            del self._cache[safe_id]
            existed = True

        if self.storage and delete_storage:
            deleted_from_disk = self.storage.delete(safe_id, delete_versions=False)
            existed = existed or deleted_from_disk

        if existed:
            for hook in self._index_hooks:
                try:
                    hook.on_removed(safe_id)
                except Exception as e:
                    logger.warning(f"IndexHook on_removed failed for {safe_id}: {e}")

        return existed

    def list(
        self,
        filter_fn: Optional[Callable[[T], bool]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        sort_by: Optional[str] = None,
        reverse: bool = False,
    ) -> List[T]:
        """
        List and filter registered entities with pagination and sorting.
        """
        if offset < 0:
            raise InvalidQueryError(
                message=f"Offset must be non-negative, got {offset}",
                context={"offset": offset},
            )

        # Ensure all disk entities are cached
        if self.storage:
            for disk_id in self.storage.list_ids():
                if disk_id not in self._cache:
                    try:
                        self._cache[disk_id] = self.storage.read(disk_id)
                    except Exception:
                        pass

        results = list(self._cache.values())

        if filter_fn is not None:
            results = [e for e in results if filter_fn(e)]

        if sort_by is not None:
            def sort_key(item: T) -> Any:
                val = getattr(item, sort_by, None)
                if val is None:
                    return (0, 0, "")
                if isinstance(val, (int, float)):
                    return (1, val, "")
                return (2, 0, str(val))

            results = sorted(results, key=sort_key, reverse=reverse)


        if offset > 0:
            results = results[offset:]

        if limit is not None and limit >= 0:
            results = results[:limit]

        return results

    def version(self, entry_id: str) -> List[str]:
        """
        Retrieve all available version identifiers for an entry.
        """
        if not entry_id:
            return []
        safe_id = str(entry_id).strip()

        if self.storage:
            return self.storage.list_versions(safe_id)

        if safe_id in self._cache:
            entry = self._cache[safe_id]
            ver = getattr(entry, "schema_version", getattr(entry, "version", "1.0.0"))
            return [ver]

        return []

    def exists(self, entry_id: str) -> bool:
        """Check if an entry is registered in memory or on disk."""
        if not entry_id:
            return False
        safe_id = str(entry_id).strip()
        if safe_id in self._cache:
            return True
        if self.storage:
            return self.storage.exists(safe_id)
        return False

    def count(self, filter_fn: Optional[Callable[[T], bool]] = None) -> int:
        """Count total matching entries in the registry."""
        return len(self.list(filter_fn=filter_fn))

    def clear(self) -> None:
        """Clear all registry entries from memory and disk."""
        ids = list(self._cache.keys())
        self._cache.clear()
        if self.storage:
            self.storage.clear()
        for eid in ids:
            for hook in self._index_hooks:
                try:
                    hook.on_removed(eid)
                except Exception:
                    pass

"""
exceptions.py
=============

Structured exception hierarchy for the Thumbnail Intelligence Knowledge Base.
Provides domain-specific exceptions with structured context, machine-readable error codes,
and payload inspection for all operations across storage, validation, registry, versioning,
and serialization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class KnowledgeBaseError(Exception):
    """
    Base exception for all Thumbnail Intelligence Knowledge Base errors.
    All subsystem exceptions must inherit from this class.
    """

    default_error_code: str = "KB_ERROR"

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.default_error_code
        self.context: Dict[str, Any] = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception into a structured serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        ctx_str = f" | Context: {self.context}" if self.context else ""
        return f"[{self.error_code}] {self.message}{ctx_str}"


# ---------------------------------------------------------------------------
# Validation Exceptions
# ---------------------------------------------------------------------------


class ValidationError(KnowledgeBaseError):
    """Base exception for model or schema validation failures."""

    default_error_code = "KB_VALIDATION_ERROR"


class SchemaValidationError(ValidationError):
    """Raised when a data payload fails Pydantic schema validation or type checks."""

    default_error_code = "KB_SCHEMA_VALIDATION_ERROR"


class EvidenceValidationError(ValidationError):
    """
    Raised when an intelligence model or claim lacks a resolvable or valid EvidenceReference.
    Enforces the grounding gate requirement (Interpretation, not invention).
    """

    default_error_code = "KB_EVIDENCE_VALIDATION_ERROR"


class ConstraintValidationError(ValidationError):
    """Raised when brand or identity constraints are violated or self-contradictory."""

    default_error_code = "KB_CONSTRAINT_VALIDATION_ERROR"


class IntegrityValidationError(ValidationError):
    """Raised when an entry's internal cross-field invariants fail."""

    default_error_code = "KB_INTEGRITY_VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Storage Exceptions
# ---------------------------------------------------------------------------


class StorageError(KnowledgeBaseError):
    """Base exception for persistence, disk, and filesystem errors."""

    default_error_code = "KB_STORAGE_ERROR"


class AtomicWriteError(StorageError):
    """Raised when an atomic write-then-replace operation fails to complete."""

    default_error_code = "KB_ATOMIC_WRITE_ERROR"


class StorageNotFoundError(StorageError):
    """Raised when an entry file or target directory is not found on disk."""

    default_error_code = "KB_STORAGE_NOT_FOUND"


class StorageCorruptionError(StorageError):
    """Raised when persisted JSON data is corrupt, truncated, or unparseable."""

    default_error_code = "KB_STORAGE_CORRUPTION_ERROR"


class StorageIOError(StorageError):
    """Raised when an underlying OS or I/O operation fails during storage access."""

    default_error_code = "KB_STORAGE_IO_ERROR"


class StoragePermissionError(StorageError):
    """Raised when storage access is denied due to filesystem permissions."""

    default_error_code = "KB_STORAGE_PERMISSION_ERROR"


# ---------------------------------------------------------------------------
# Registry Exceptions
# ---------------------------------------------------------------------------


class RegistryError(KnowledgeBaseError):
    """Base exception for registry operations (registration, lookup, removal)."""

    default_error_code = "KB_REGISTRY_ERROR"


class EntryNotFoundError(RegistryError):
    """Raised when an entry with the specified identifier is not found in the registry."""

    default_error_code = "KB_ENTRY_NOT_FOUND"


class DuplicateEntryError(RegistryError):
    """Raised when attempting to register an entry with an ID that already exists."""

    default_error_code = "KB_DUPLICATE_ENTRY"


class EntryConflictError(RegistryError):
    """Raised when a concurrent update or version collision occurs."""

    default_error_code = "KB_ENTRY_CONFLICT"


class InvalidQueryError(RegistryError):
    """Raised when a query filter or lookup specification is invalid."""

    default_error_code = "KB_INVALID_QUERY"


# ---------------------------------------------------------------------------
# Versioning & Migration Exceptions
# ---------------------------------------------------------------------------


class VersioningError(KnowledgeBaseError):
    """Base exception for version comparison and migration failures."""

    default_error_code = "KB_VERSIONING_ERROR"


class SchemaVersionMismatchError(VersioningError):
    """Raised when a model or record's schema version does not match expected version."""

    default_error_code = "KB_VERSION_MISMATCH"


class MigrationError(VersioningError):
    """Raised when applying a migration transform hook between schema versions fails."""

    default_error_code = "KB_MIGRATION_ERROR"


class UnsupportedVersionError(VersioningError):
    """Raised when an entry's schema version is unsupported or has no migration path."""

    default_error_code = "KB_UNSUPPORTED_VERSION"


# ---------------------------------------------------------------------------
# Serialization Exceptions
# ---------------------------------------------------------------------------


class SerializationError(KnowledgeBaseError):
    """Base exception for JSON encoding or model serialization failures."""

    default_error_code = "KB_SERIALIZATION_ERROR"


class DeserializationError(SerializationError):
    """Raised when converting raw dictionaries or JSON into structured models fails."""

    default_error_code = "KB_DESERIALIZATION_ERROR"


class TypeSerializationError(SerializationError):
    """Raised when encountering an unsupported or non-serializable object type."""

    default_error_code = "KB_TYPE_SERIALIZATION_ERROR"

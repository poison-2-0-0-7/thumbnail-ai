"""
knowledge_base
==============

Knowledge Base Foundation for the Thumbnail Intelligence Engine.
Provides production data contracts, atomic local storage, semantic versioning,
auditable grounding validation, typed registries, and high-level repositories.
"""

from __future__ import annotations

from thumbnail_intelligence.knowledge_base.config import (
    DEFAULT_KB_DATA_DIR,
    DEFAULT_KB_LOG_FILE,
    KnowledgeBaseConfig,
    StorageConfig,
    VersioningConfig,
)
from thumbnail_intelligence.knowledge_base.exceptions import (
    AtomicWriteError,
    ConstraintValidationError,
    DeserializationError,
    DuplicateEntryError,
    EntryConflictError,
    EntryNotFoundError,
    EvidenceValidationError,
    InvalidQueryError,
    KnowledgeBaseError,
    MigrationError,
    RegistryError,
    SchemaValidationError,
    SchemaVersionMismatchError,
    SerializationError,
    StorageCorruptionError,
    StorageError,
    StorageIOError,
    StorageNotFoundError,
    StoragePermissionError,
    TypeSerializationError,
    UnsupportedVersionError,
    ValidationError,
    VersioningError,
)
from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    ArchetypeMatch,
    BaseKBModel,
    BrandConstraint,
    ChannelProfile,
    CompetitorProfile,
    CompetitorStatus,
    CreatorProfile,
    DesignPattern,
    DesignReason,
    DesignReasonType,
    DifferentiationSummary,
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
    IdentityConstraint,
    KnowledgeEntry,
    KnowledgeEntryType,
    PatternScope,
    ThumbnailPattern,
    VisualPattern,
)
from thumbnail_intelligence.knowledge_base.registry import IndexHook, KnowledgeRegistry
from thumbnail_intelligence.knowledge_base.repository import KnowledgeBaseRepository
from thumbnail_intelligence.knowledge_base.serialization import KBSerializer, KnowledgeBaseJSONEncoder
from thumbnail_intelligence.knowledge_base.storage import AtomicFileWriter, VersionedFileStorage
from thumbnail_intelligence.knowledge_base.validation import (
    ConstraintValidator,
    EvidenceValidator,
    ModelValidator,
    SchemaIntegrityValidator,
)
from thumbnail_intelligence.knowledge_base.versioning import (
    MigrationHook,
    MigrationRegistry,
    SemVer,
    global_migration_registry,
)

__all__ = [
    # Config
    "KnowledgeBaseConfig",
    "StorageConfig",
    "VersioningConfig",
    "DEFAULT_KB_DATA_DIR",
    "DEFAULT_KB_LOG_FILE",
    # Exceptions
    "KnowledgeBaseError",
    "ValidationError",
    "SchemaValidationError",
    "EvidenceValidationError",
    "ConstraintValidationError",
    "StorageError",
    "AtomicWriteError",
    "StorageNotFoundError",
    "StorageCorruptionError",
    "StorageIOError",
    "StoragePermissionError",
    "RegistryError",
    "EntryNotFoundError",
    "DuplicateEntryError",
    "EntryConflictError",
    "InvalidQueryError",
    "VersioningError",
    "SchemaVersionMismatchError",
    "MigrationError",
    "UnsupportedVersionError",
    "SerializationError",
    "DeserializationError",
    "TypeSerializationError",
    # Models
    "BaseKBModel",
    "KnowledgeEntry",
    "KnowledgeEntryType",
    "CreatorProfile",
    "ChannelProfile",
    "CompetitorProfile",
    "CompetitorStatus",
    "Archetype",
    "ArchetypeMatch",
    "EvidenceReference",
    "EvidenceSourceType",
    "EvidenceGrade",
    "DesignReason",
    "DesignReasonType",
    "BrandConstraint",
    "IdentityConstraint",
    "VisualPattern",
    "DesignPattern",
    "PatternScope",
    "ThumbnailPattern",
    "DifferentiationSummary",
    # Storage & Serialization
    "AtomicFileWriter",
    "VersionedFileStorage",
    "KBSerializer",
    "KnowledgeBaseJSONEncoder",
    # Validation & Versioning
    "ModelValidator",
    "EvidenceValidator",
    "ConstraintValidator",
    "SchemaIntegrityValidator",
    "SemVer",
    "MigrationRegistry",
    "MigrationHook",
    "global_migration_registry",
    # Registry & Repository
    "KnowledgeRegistry",
    "IndexHook",
    "KnowledgeBaseRepository",
]

"""
config.py
=========

Configuration management and directory paths for the Thumbnail Intelligence Knowledge Base.
Defines storage locations, file naming conventions, atomic write parameters, schema version defaults,
and environment overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _get_project_root() -> Path:
    """Resolve the root of the thumbnail-ai repository."""
    # thumbnail_intelligence/knowledge_base/config.py -> parent x3 is project root
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT: Path = _get_project_root()

# Default Knowledge Base root directory
DEFAULT_KB_DATA_DIR: Path = PROJECT_ROOT / "data" / "intelligence_kb"

# Default Knowledge Base log directory
DEFAULT_KB_LOG_DIR: Path = PROJECT_ROOT / "logs"
DEFAULT_KB_LOG_FILE: Path = DEFAULT_KB_LOG_DIR / "knowledge_base.log"


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for local JSON persistence and atomic writes."""

    base_dir: Path = DEFAULT_KB_DATA_DIR
    temp_suffix: str = ".tmp"
    backup_suffix: str = ".bak"
    fsync_on_write: bool = True
    enable_backups: bool = True
    max_backup_versions: int = 5
    indent: int = 2
    encoding: str = "utf-8"
    permissions_mode: Optional[int] = None


@dataclass(frozen=True)
class VersioningConfig:
    """Configuration for schema versioning, compatibility, and migration."""

    default_schema_version: str = "1.0.0"
    current_engine_version: str = "1.0.0"
    enforce_strict_version_match: bool = False
    auto_migrate_on_read: bool = True


@dataclass
class KnowledgeBaseConfig:
    """
    Central configuration object for all Knowledge Base repositories and storage subsystems.
    Supports environment variable overrides and programmatic dependency injection.
    """

    base_dir: Path = field(default_factory=lambda: DEFAULT_KB_DATA_DIR)
    storage: StorageConfig = field(default_factory=StorageConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    log_file: Path = field(default_factory=lambda: DEFAULT_KB_LOG_FILE)
    embedding_dim: int = 512
    default_embedding_model: str = "OpenCLIP-ViT-B-32"

    def __post_init__(self) -> None:
        # Allow environment variable override of KB base directory
        env_base = os.getenv("THUMBNAIL_AI_KB_DIR")
        if env_base:
            self.base_dir = Path(env_base).resolve()
            self.storage = StorageConfig(base_dir=self.base_dir)

    @property
    def creator_profiles_dir(self) -> Path:
        return self.base_dir / "creator_profiles"

    @property
    def channel_profiles_dir(self) -> Path:
        return self.base_dir / "channel_profiles"

    @property
    def competitor_profiles_dir(self) -> Path:
        return self.base_dir / "competitors"

    @property
    def archetypes_dir(self) -> Path:
        return self.base_dir / "archetypes"

    @property
    def historical_index_dir(self) -> Path:
        return self.base_dir / "historical_index"

    @property
    def design_patterns_dir(self) -> Path:
        return self.base_dir / "design_patterns"

    @property
    def visual_patterns_dir(self) -> Path:
        return self.base_dir / "visual_patterns"

    @property
    def thumbnail_patterns_dir(self) -> Path:
        return self.base_dir / "thumbnail_patterns"

    @property
    def brand_constraints_dir(self) -> Path:
        return self.base_dir / "brand_constraints"

    @property
    def identity_constraints_dir(self) -> Path:
        return self.base_dir / "identity_constraints"

    @property
    def design_briefs_dir(self) -> Path:
        return self.base_dir / "design_briefs"

    @property
    def backups_dir(self) -> Path:
        return self.base_dir / "backups"

    def ensure_directories(self) -> None:
        """Ensure all required Knowledge Base filesystem directories exist."""
        dirs = [
            self.base_dir,
            self.creator_profiles_dir,
            self.channel_profiles_dir,
            self.competitor_profiles_dir,
            self.archetypes_dir,
            self.historical_index_dir,
            self.design_patterns_dir,
            self.visual_patterns_dir,
            self.thumbnail_patterns_dir,
            self.brand_constraints_dir,
            self.identity_constraints_dir,
            self.design_briefs_dir,
            self.backups_dir,
            self.log_file.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "base_dir": str(self.base_dir),
            "storage": {
                "temp_suffix": self.storage.temp_suffix,
                "backup_suffix": self.storage.backup_suffix,
                "fsync_on_write": self.storage.fsync_on_write,
                "enable_backups": self.storage.enable_backups,
                "max_backup_versions": self.storage.max_backup_versions,
                "indent": self.storage.indent,
                "encoding": self.storage.encoding,
            },
            "versioning": {
                "default_schema_version": self.versioning.default_schema_version,
                "current_engine_version": self.versioning.current_engine_version,
                "enforce_strict_version_match": self.versioning.enforce_strict_version_match,
                "auto_migrate_on_read": self.versioning.auto_migrate_on_read,
            },
            "log_file": str(self.log_file),
            "embedding_dim": self.embedding_dim,
            "default_embedding_model": self.default_embedding_model,
        }

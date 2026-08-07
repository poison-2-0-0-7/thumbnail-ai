"""
versioning.py
=============

Semantic versioning, compatibility checks, and schema migration engine for the Knowledge Base.
Provides:
- Semantic Version parsing and mathematical comparison (SemVer)
- Migration hook registration and migration pipeline chaining
- Backward compatibility verification and schema version upgrades
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from thumbnail_intelligence.knowledge_base.exceptions import (
    MigrationError,
    SchemaVersionMismatchError,
    UnsupportedVersionError,
    VersioningError,
)

MigrationHook = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True, order=True)
class SemVer:
    """
    Semantic version representation (major.minor.patch) with strict ordering and compatibility checks.
    """

    major: int
    minor: int
    patch: int

    _SEMVER_REGEX = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")

    @classmethod
    def parse(cls, version_str: str) -> SemVer:
        """Parse a semver string like '1.0.0' or 'v2.1.3' into a SemVer instance."""
        if not isinstance(version_str, str):
            raise VersioningError(
                message=f"Version must be a string, got {type(version_str).__name__}",
                context={"value": str(version_str)},
            )
        match = cls._SEMVER_REGEX.match(version_str.strip())
        if not match:
            raise VersioningError(
                message=f"Invalid semantic version string '{version_str}'. Expected format: 'X.Y.Z'",
                context={"version": version_str},
            )
        major, minor, patch = map(int, match.groups()[:3])
        return cls(major=major, minor=minor, patch=patch)

    def is_compatible(self, target: SemVer) -> bool:
        """
        Check if self is backward-compatible with target.
        Major versions must match (unless major=0). Current minor/patch must be >= target.
        """
        if self.major != target.major:
            return False
        return self >= target

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class MigrationRegistry:
    """
    Registry for schema migration hooks.
    Maintains directional transform graphs to seamlessly upgrade older on-disk JSON payloads
    to modern schema definitions without loss of data.
    """

    def __init__(self) -> None:
        # (model_name, from_version, to_version) -> MigrationHook
        self._hooks: Dict[Tuple[str, str, str], MigrationHook] = {}

    def register(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
        hook: MigrationHook,
    ) -> None:
        """
        Register a migration transform callable from from_version to to_version.
        """
        # Validate versions
        SemVer.parse(from_version)
        SemVer.parse(to_version)
        key = (model_name.strip(), from_version.strip(), to_version.strip())
        self._hooks[key] = hook

    def has_migration(self, model_name: str, from_version: str, to_version: str) -> bool:
        """Check if a direct or indirect migration path exists."""
        try:
            path = self.get_migration_path(model_name, from_version, to_version)
            return len(path) > 0
        except VersioningError:
            return False

    def get_migration_path(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
    ) -> List[MigrationHook]:
        """
        Find shortest migration transformation sequence using BFS.
        """
        from_v = from_version.strip()
        to_v = to_version.strip()

        if from_v == to_v:
            return []

        # Find direct hook
        direct_key = (model_name, from_v, to_v)
        if direct_key in self._hooks:
            return [self._hooks[direct_key]]

        # Build adjacency graph for model
        graph: Dict[str, List[Tuple[str, MigrationHook]]] = {}
        for (m, f, t), hook in self._hooks.items():
            if m == model_name:
                graph.setdefault(f, []).append((t, hook))

        # BFS search
        queue = [(from_v, [])]
        visited = {from_v}

        while queue:
            current_v, path = queue.pop(0)
            if current_v == to_v:
                return path

            for next_v, hook in graph.get(current_v, []):
                if next_v not in visited:
                    visited.add(next_v)
                    queue.append((next_v, path + [hook]))

        raise UnsupportedVersionError(
            message=f"No migration path registered for model '{model_name}' from v{from_v} to v{to_v}",
            context={"model_name": model_name, "from_version": from_v, "to_version": to_v},
        )

    def migrate(
        self,
        model_name: str,
        data: Dict[str, Any],
        target_version: str,
    ) -> Dict[str, Any]:
        """
        Apply migration sequence to upgrade data dictionary to target_version.
        """
        current_version = data.get("schema_version", data.get("version", "1.0.0"))
        if current_version == target_version:
            return data

        hooks = self.get_migration_path(model_name, current_version, target_version)
        current_payload = dict(data)

        for idx, hook in enumerate(hooks):
            try:
                current_payload = hook(current_payload)
            except Exception as e:
                raise MigrationError(
                    message=f"Migration failed during step {idx + 1} for '{model_name}': {e}",
                    context={"model_name": model_name, "step": idx + 1, "error": str(e)},
                ) from e

        # Ensure schema_version is stamped with target_version
        current_payload["schema_version"] = target_version
        if "version" in current_payload:
            current_payload["version"] = target_version
        return current_payload


# Global migration registry instance
global_migration_registry = MigrationRegistry()

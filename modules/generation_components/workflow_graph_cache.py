"""WorkflowGraphCache: process-memory graph materialization cache for a single generation run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger


class WorkflowGraphCache:
    """Run-scoped cache for materialized template dicts and assembled fragments."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def make_key(
        self,
        template_path: str,
        workflow_version: str,
        profile_name: str,
        conditioning_hash: str,
    ) -> tuple[str, str, str, str]:
        """Construct deterministic cache key."""
        return (template_path, workflow_version, profile_name, conditioning_hash)

    def get(self, key: tuple[str, str, str, str]) -> Optional[dict[str, Any]]:
        """Lookup cached base graph materialization."""
        if not self.enabled:
            return None
        hit = self._cache.get(key)
        if hit is not None:
            logger.debug("WorkflowGraphCache hit for key={key}", key=key)
        return hit

    def put(self, key: tuple[str, str, str, str], base_graph: dict[str, Any]) -> None:
        """Store base graph materialization in cache."""
        if not self.enabled:
            return
        self._cache[key] = base_graph
        logger.debug("WorkflowGraphCache store for key={key}", key=key)

    def clear(self) -> None:
        """Clear cache contents."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

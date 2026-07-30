"""
capability_probe.py
====================

Probes live ComfyUI server via /object_info to verify installed node types.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

from config import (
    MODULE7_CAPABILITY_PROBE_CACHE_SECONDS,
    MODULE7_CAPABILITY_PROBE_ENABLED,
    MODULE7_LOG_PATH,
)
from generation_components.interfaces import ICapabilityProbe
from module7_exceptions import UnsupportedNodeTypeWarning
from loguru import logger


def _configure_logger() -> None:
    """Ensure Loguru sink is configured for Module 7."""
    try:
        logger.add(
            MODULE7_LOG_PATH,
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            enqueue=True,
        )
    except ValueError:
        pass

_configure_logger()


class CapabilityProbe(ICapabilityProbe):
    """Probes installed ComfyUI node types and caches results per pipeline run."""

    def __init__(
        self,
        client: Any | None = None,
        enabled: bool = MODULE7_CAPABILITY_PROBE_ENABLED,
        cache_ttl_seconds: float = MODULE7_CAPABILITY_PROBE_CACHE_SECONDS,
    ) -> None:
        self.client = client
        self.enabled = enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_types: frozenset[str] | None = None
        self._last_probe_time: float = 0.0

    def installed_node_types(self) -> frozenset[str]:
        """
        Return set of installed node class_types.

        Returns:
            frozenset of node class_type strings, or empty set if disabled/unavailable.
        """
        if not self.enabled or self.client is None:
            return frozenset()

        now = time.monotonic()
        if self._cached_types is not None and (now - self._last_probe_time) < self.cache_ttl_seconds:
            return self._cached_types

        try:
            info = self.client.object_info()
            if isinstance(info, dict):
                self._cached_types = frozenset(info.keys())
                self._last_probe_time = now
                logger.info("CapabilityProbe cached {count} installed ComfyUI node types", count=len(self._cached_types))
                return self._cached_types
        except Exception as exc:
            logger.warning("CapabilityProbe failed to fetch /object_info: {exc}", exc=exc)

        self._cached_types = frozenset()
        self._last_probe_time = now
        return self._cached_types

    def is_fragment_supported(self, fragment: dict[str, Any]) -> bool:
        """
        Check if fragment's required node types are installed on the server.

        Args:
            fragment: Fragment definition dictionary.

        Returns:
            True if all required nodes are available or probing disabled, False otherwise.
        """
        if not self.enabled or self.client is None:
            return True

        installed = self.installed_node_types()
        if not installed:
            # If probe failed or returned empty, fail soft and allow fragment
            return True

        meta = fragment.get("_meta", {})
        required = meta.get("required_node_types", [])

        # Also extract class_types directly from the fragment graph
        graph_nodes = fragment.get("graph", {})
        graph_class_types = [
            node.get("class_type")
            for node in graph_nodes.values()
            if isinstance(node, dict) and node.get("class_type")
        ]

        all_required = set(required).union(graph_class_types)
        missing = [nt for nt in all_required if nt not in installed]

        if missing:
            for node_type in missing:
                msg = f"Fragment dropped: required node type {node_type} not available in ComfyUI /object_info"
                logger.warning(msg)
                warnings.warn(msg, UnsupportedNodeTypeWarning)
            return False

        return True

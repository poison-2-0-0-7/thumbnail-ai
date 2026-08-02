"""
model_discovery_service.py
===========================

Discovers installed ComfyUI model files per loader node class directly from
ComfyUI's /object_info endpoint combo enumeration.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from config import (
    MODULE7_CAPABILITY_DISCOVERY_ENABLED,
    MODULE7_CAPABILITY_PROBE_CACHE_SECONDS,
    MODULE7_LOG_PATH,
)
from generation_components.capability_probe import CapabilityProbe
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


class ModelDiscoveryService:
    """
    Queries live ComfyUI server via /object_info to discover installed model filenames per node class.
    """

    def __init__(
        self,
        client: Any | None = None,
        probe: CapabilityProbe | None = None,
        enabled: bool = MODULE7_CAPABILITY_DISCOVERY_ENABLED,
        cache_ttl_seconds: float = MODULE7_CAPABILITY_PROBE_CACHE_SECONDS,
    ) -> None:
        self.client = client
        self.probe = probe
        self.enabled = enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_info: dict[str, Any] | None = None
        self._last_probe_time: float = 0.0

    def _get_object_info(self) -> dict[str, Any]:
        """Retrieve raw object info dictionary, reusing CapabilityProbe cache if available."""
        if not self.enabled:
            return {}

        if self.probe is not None and isinstance(self.probe, CapabilityProbe):
            return self.probe.get_raw_object_info()

        if self.client is None:
            return {}

        now = time.monotonic()
        if self._cached_info is not None and (now - self._last_probe_time) < self.cache_ttl_seconds:
            return self._cached_info

        try:
            info = self.client.object_info()
            if isinstance(info, dict):
                self._cached_info = info
                self._last_probe_time = now
                logger.info(
                    "ModelDiscoveryService cached /object_info ({count} node types)",
                    count=len(info),
                )
                return self._cached_info
        except Exception as exc:
            logger.warning("ModelDiscoveryService failed to fetch /object_info: {exc}", exc=exc)

        self._cached_info = {}
        self._last_probe_time = now
        return self._cached_info

    def installed_models_for(self, node_class: str, field_name: str) -> tuple[str, ...]:
        """
        Return tuple of installed model filenames reported by ComfyUI for node_class's field_name.

        Args:
            node_class: ComfyUI node class_type (e.g. 'ControlNetLoader', 'T2IAdapterLoader').
            field_name: Required input field name (e.g. 'control_net_name', 't2i_adapter_name').

        Returns:
            Tuple of installed filename strings, or empty tuple if unavailable/disabled.
        """
        if not self.enabled:
            return ()

        info = self._get_object_info()
        if not isinstance(info, dict) or node_class not in info:
            return ()

        node_info = info.get(node_class)
        if not isinstance(node_info, dict):
            return ()

        input_def = node_info.get("input")
        if not isinstance(input_def, dict):
            return ()

        required_def = input_def.get("required")
        if not isinstance(required_def, dict):
            return ()

        field_def = required_def.get(field_name)
        # ComfyUI INPUT_TYPES convention: field_def is [options_list, config_dict]
        if isinstance(field_def, (list, tuple)) and len(field_def) > 0:
            options = field_def[0]
            if isinstance(options, (list, tuple)):
                return tuple(str(opt) for opt in options if isinstance(opt, str))

        return ()

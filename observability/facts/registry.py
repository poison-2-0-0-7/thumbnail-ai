"""
observability/facts/registry.py
================================

Registry for fact definitions and fact extraction handlers in PORCE.
"""

from __future__ import annotations

from typing import Any, Callable
from observability.models import PipelineTrace


class FactRegistry:
    """
    Registry holding fact extraction functions and schema metadata.
    Allows registering custom handlers for specific fact categories.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[PipelineTrace], dict[str, Any]]] = {}

    def register(self, category: str, handler: Callable[[PipelineTrace], dict[str, Any]]) -> None:
        """Register a handler for a category of facts."""
        self._handlers[category] = handler

    def unregister(self, category: str) -> None:
        """Unregister a category handler if present."""
        self._handlers.pop(category, None)

    def get_registered_categories(self) -> list[str]:
        """Return list of all registered fact categories."""
        return list(self._handlers.keys())

    def extract_category_facts(self, category: str, trace: PipelineTrace) -> dict[str, Any]:
        """Extract facts for a specific registered category."""
        handler = self._handlers.get(category)
        if handler is None:
            return {}
        try:
            return handler(trace)
        except Exception:
            return {}

    def extract_all_custom_facts(self, trace: PipelineTrace) -> dict[str, dict[str, Any]]:
        """Run all registered handlers on trace and return results dictionary keyed by category."""
        results: dict[str, dict[str, Any]] = {}
        for cat, handler in self._handlers.items():
            try:
                results[cat] = handler(trace)
            except Exception:
                results[cat] = {}
        return results

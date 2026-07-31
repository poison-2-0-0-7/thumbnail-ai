"""Exceptions for Module 5.5 (Thumbnail Copywriter & Layout Planner)."""

from __future__ import annotations


class DesignBlueprintError(Exception):
    """Base exception for Module 5.5 failures."""


class InvalidRedesignSpecError(DesignBlueprintError):
    """Raised when a RedesignSpecification cannot support building a DesignBlueprint."""


class DesignBlueprintCacheError(DesignBlueprintError):
    """Raised when a DesignBlueprint cache file cannot be written or read."""

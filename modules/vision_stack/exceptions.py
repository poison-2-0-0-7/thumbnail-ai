"""Exceptions for AI Vision Stack V2.1 bootstrap components."""

from __future__ import annotations


class VisionStackError(Exception):
    """Base exception for Vision Stack bootstrap and lifecycle failures."""


class VisionStackConfigError(VisionStackError):
    """Raised when the Vision Stack YAML configuration is missing or invalid."""


class VisionStackRegistryError(VisionStackError):
    """Raised when registry state is invalid."""

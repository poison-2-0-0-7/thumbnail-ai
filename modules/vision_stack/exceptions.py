"""Exceptions for AI Vision Stack V2.1 bootstrap components."""

from __future__ import annotations


class VisionStackError(Exception):
    """Base exception for Vision Stack bootstrap and lifecycle failures."""


class VisionStackConfigError(VisionStackError):
    """Raised when the Vision Stack YAML configuration is missing or invalid."""


class VisionStackRegistryError(VisionStackError):
    """Raised when registry state is invalid."""


class VisionStackLoaderError(VisionStackError):
    """Raised when boot-time model metadata cannot be resolved."""


class VisionStackCheckpointError(VisionStackLoaderError):
    """Raised when a configured checkpoint or required sidecar file is missing."""


class VisionStackRuntimeError(VisionStackError):
    """Raised when runtime lifecycle coordination fails."""


class VisionStackResourceError(VisionStackRuntimeError):
    """Raised when GPU resource ownership or sequential execution is invalid."""

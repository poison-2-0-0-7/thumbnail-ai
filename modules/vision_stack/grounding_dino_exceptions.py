"""Typed exceptions for the GroundingDINO Vision Stack wrapper."""

from __future__ import annotations

from .exceptions import VisionStackError


class GroundingDINOError(VisionStackError):
    """Base exception for GroundingDINO wrapper failures."""


class GroundingDINOLoadError(GroundingDINOError):
    """Raised when GroundingDINO weights or model configuration cannot load."""


class GroundingDINOInferenceError(GroundingDINOError):
    """Raised when a GroundingDINO forward pass fails."""


class GroundingDINOOutOfMemoryError(GroundingDINOInferenceError):
    """Raised when GroundingDINO inference fails due to CUDA out-of-memory."""


class GroundingDINOParseError(GroundingDINOError):
    """Raised when raw GroundingDINO output cannot be normalized."""

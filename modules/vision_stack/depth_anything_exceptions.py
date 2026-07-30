"""
depth_anything_exceptions.py
============================

Exception hierarchy for Vision Stack V2.1 DepthAnything wrapper.
Subclasses vision_stack.exceptions.VisionStackError.
"""

from .exceptions import VisionStackError


class DepthAnythingError(VisionStackError):
    """Base exception for DepthAnything wrapper errors."""


class DepthAnythingLoadError(DepthAnythingError):
    """Raised when DepthAnything checkpoint or model load fails."""


class DepthAnythingInferenceError(DepthAnythingError):
    """Raised when DepthAnything depth estimation inference fails."""


class DepthAnythingOutOfMemoryError(DepthAnythingError):
    """Raised when DepthAnything inference encounters CUDA Out of Memory."""


class DepthAnythingParseError(DepthAnythingError):
    """Raised when DepthAnything depth output tensors cannot be parsed."""

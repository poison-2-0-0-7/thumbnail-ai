"""
birefnet_exceptions.py
======================

Exception hierarchy for Vision Stack V2.1 BiRefNet wrapper.
Subclasses vision_stack.exceptions.VisionStackError.
"""

from .exceptions import VisionStackError


class BiRefNetError(VisionStackError):
    """Base exception for BiRefNet wrapper errors."""


class BiRefNetLoadError(BiRefNetError):
    """Raised when BiRefNet checkpoint or model load fails."""


class BiRefNetInferenceError(BiRefNetError):
    """Raised when BiRefNet matting inference fails."""


class BiRefNetOutOfMemoryError(BiRefNetError):
    """Raised when BiRefNet inference encounters CUDA Out of Memory."""


class BiRefNetParseError(BiRefNetError):
    """Raised when BiRefNet output mask tensors cannot be parsed."""

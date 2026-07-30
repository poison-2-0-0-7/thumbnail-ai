"""
bisenet_exceptions.py
=====================

Exception hierarchy for Vision Stack V2.1 BiSeNet wrapper.
Subclasses vision_stack.exceptions.VisionStackError.
"""

from .exceptions import VisionStackError


class BiSeNetError(VisionStackError):
    """Base exception for BiSeNet wrapper errors."""


class BiSeNetLoadError(BiSeNetError):
    """Raised when BiSeNet checkpoint or model load fails."""


class BiSeNetInferenceError(BiSeNetError):
    """Raised when BiSeNet human parsing inference fails."""


class BiSeNetOutOfMemoryError(BiSeNetError):
    """Raised when BiSeNet inference encounters CUDA Out of Memory."""


class BiSeNetParseError(BiSeNetError):
    """Raised when BiSeNet segmentation logits cannot be parsed."""

"""
sam2_exceptions.py
==================

Exception hierarchy for Vision Stack V2.1 SAM2 wrapper.
Subclasses vision_stack.exceptions.VisionStackError.
"""

from .exceptions import VisionStackError


class SAM2Error(VisionStackError):
    """Base exception for SAM2 wrapper errors."""


class SAM2LoadError(SAM2Error):
    """Raised when SAM2 checkpoint or model load fails."""


class SAM2InferenceError(SAM2Error):
    """Raised when SAM2 segmentation inference fails."""


class SAM2OutOfMemoryError(SAM2Error):
    """Raised when SAM2 inference encounters CUDA Out of Memory."""


class SAM2ParseError(SAM2Error):
    """Raised when SAM2 output tensors cannot be parsed into mask arrays."""

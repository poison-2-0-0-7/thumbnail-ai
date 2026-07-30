"""
openclip_exceptions.py
======================

Exception hierarchy for Vision Stack V2.1 OpenCLIP wrapper.
Subclasses vision_stack.exceptions.VisionStackError.
"""

from .exceptions import VisionStackError


class OpenCLIPError(VisionStackError):
    """Base exception for OpenCLIP wrapper errors."""


class OpenCLIPLoadError(OpenCLIPError):
    """Raised when OpenCLIP model load fails."""


class OpenCLIPInferenceError(OpenCLIPError):
    """Raised when OpenCLIP text/image encoding inference fails."""


class OpenCLIPOutOfMemoryError(OpenCLIPError):
    """Raised when OpenCLIP inference encounters CUDA Out of Memory."""


class OpenCLIPParseError(OpenCLIPError):
    """Raised when OpenCLIP output tensors or inputs cannot be parsed."""

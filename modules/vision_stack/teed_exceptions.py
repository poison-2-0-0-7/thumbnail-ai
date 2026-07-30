"""
teed_exceptions.py
==================

Exception hierarchy for Vision Stack V2.1 TEED wrapper.
Subclasses vision_stack.exceptions.VisionStackError.
"""

from .exceptions import VisionStackError


class TEEDError(VisionStackError):
    """Base exception for TEED wrapper errors."""


class TEEDLoadError(TEEDError):
    """Raised when TEED checkpoint or model load fails."""


class TEEDInferenceError(TEEDError):
    """Raised when TEED edge detection inference fails."""


class TEEDOutOfMemoryError(TEEDError):
    """Raised when TEED inference encounters CUDA Out of Memory."""


class TEEDParseError(TEEDError):
    """Raised when TEED edge output tensors cannot be parsed."""

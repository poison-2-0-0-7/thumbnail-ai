"""Typed exception hierarchy for Module 6.5 Visual Reference Engine."""


class VREBaseError(Exception):
    """Base exception for every recoverable VRE failure."""


class SourceImageNotFoundError(VREBaseError):
    """Raised when a source image path is missing or unreadable."""


class FaceDetectionFailedWarning(Warning):
    """Signals optional face conditioning could not be prepared."""


class SegmentationInferenceError(VREBaseError):
    """Raised when foreground or salient-object extraction fails."""


class TopologyExtractionError(VREBaseError):
    """Raised when topology maps cannot be generated from the source image."""


class AssetWriteError(VREBaseError):
    """Raised when generated assets cannot be atomically persisted."""


class ManifestValidationError(VREBaseError):
    """Raised when a reference manifest cannot satisfy the VRE schema."""

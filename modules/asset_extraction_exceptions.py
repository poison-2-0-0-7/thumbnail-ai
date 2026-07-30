"""
asset_extraction_exceptions.py
================================

Exception hierarchy for Module 8 (Asset Extraction Engine).
This module has zero project-internal dependencies so it can be safely imported
by any component.
"""

from typing import Optional


class AssetExtractionError(Exception):
    """Base exception for every recoverable Module 8 failure."""


class SourceImageNotFoundError(AssetExtractionError):
    """Raised when a source image path is missing or unreadable."""


class IntelligenceReportInvalidError(AssetExtractionError):
    """Raised when the supplied ThumbnailIntelligence cannot seed extraction
    (e.g. status == 'error', or a referenced bbox is out of range)."""


class AssetFamilyModelError(AssetExtractionError):
    """Raised when a vision-stack-backed family exhausts its configured
    fallback policy. Carries family_name and model_name for logging."""

    def __init__(
        self,
        message: str,
        *,
        family_name: str,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.family_name = family_name
        self.model_name = model_name


class AssetFamilyDegradedWarning(Warning):
    """Signals one family fell back to a lower-fidelity result (e.g.
    cpu_fallback) but still produced usable output."""


class AssetWriteError(AssetExtractionError):
    """Raised when generated assets cannot be atomically persisted."""


class ManifestValidationError(AssetExtractionError):
    """Raised when the assembled manifest fails Pydantic validation."""


class ManifestNotFoundError(AssetExtractionError):
    """Raised by load_asset_manifest() when no manifest exists for a video_id."""


class CacheCorruptError(AssetExtractionError):
    """Raised (caught internally, never surfaced) when a cached manifest or
    asset file is unreadable; triggers a full or partial recompute."""

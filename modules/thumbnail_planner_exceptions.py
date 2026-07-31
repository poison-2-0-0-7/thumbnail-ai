"""
thumbnail_planner_exceptions.py
================================

Typed exception hierarchy for Module 10.5 (Thumbnail Planner).
Leaf module with zero project-internal dependencies.
"""


class ThumbnailPlannerError(Exception):
    """Base exception for all Module 10.5 Thumbnail Planner errors."""


class UpstreamArtifactMissingError(ThumbnailPlannerError):
    """Raised when a required upstream artifact file is missing."""


class PlanValidationError(ThumbnailPlannerError):
    """Raised when a generation plan fails validation."""


class PlanCacheError(ThumbnailPlannerError):
    """Raised when generation plan cache operations fail."""


class PlanPersistError(ThumbnailPlannerError):
    """Raised when generation plan cannot be saved to disk."""

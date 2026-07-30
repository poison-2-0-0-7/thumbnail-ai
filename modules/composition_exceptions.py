"""
composition_exceptions.py
=========================

Typed exception hierarchy for Module 10 (Asset Composer).

Mirrors the flat-hierarchy pattern established in vre_exceptions.py and
module7_exceptions.py.
"""

from __future__ import annotations


class CompositionBaseError(Exception):
    """Base exception for every Asset Composer failure."""


class CompositionInputInvalidError(CompositionBaseError):
    """Raised when the upstream RedesignSpecification or PromptPackage is unusable."""


class AssetRegistryError(CompositionBaseError):
    """Raised when a referenced VRE asset is missing, unreadable, or checksum-mismatched."""


class LayerPlacementError(CompositionBaseError):
    """Raised when geometry resolution produces an invalid or out-of-canvas placement."""


class MaskResolutionError(CompositionBaseError):
    """Raised when a required mask cannot be bound to its layer."""


class WorkspaceValidationError(CompositionBaseError):
    """Raised when CompositionValidator finds structural or referential defects."""


class WorkspacePersistenceError(CompositionBaseError):
    """Raised when the workspace cannot be atomically written to disk."""


class GenerationBundleError(CompositionBaseError):
    """Raised when a validated workspace cannot be flattened into a GenerationBundle."""

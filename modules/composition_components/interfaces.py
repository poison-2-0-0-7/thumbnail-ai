"""
interfaces.py
=============

Abstract base classes for Module 10 Asset Composer components.
Enables dependency injection and isolated component testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from models import (
    AssetMetadata,
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    GenerationBundle,
    LayerDecision,
    LayerGroup,
    LayerRole,
    LayerTransform,
    MaskReference,
    PromptPackage,
    RedesignSpecification,
    TextPlacement,
    VisualBoundingBox,
    VisualReferenceManifest,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


class IAssetRegistry(ABC):
    """Interface for indexing and resolving VRE assets."""

    @abstractmethod
    def index(self, manifest: VisualReferenceManifest) -> dict[str, AssetMetadata]:
        """Index all assets from a VisualReferenceManifest."""
        pass

    @abstractmethod
    def resolve(self, role: str) -> Optional[AssetMetadata]:
        """Resolve AssetMetadata by asset key/role."""
        pass

    @abstractmethod
    def verify_integrity(self) -> list[str]:
        """Verify that indexed assets exist and match checksums. Returns list of invalid asset_ids."""
        pass


class IDecisionResolver(ABC):
    """Interface for resolving KEEP/REMOVE/REPLACE/ENHANCE/ADD decisions."""

    @abstractmethod
    def resolve(
        self,
        spec: RedesignSpecification,
        decision_manifest: Optional[DecisionManifest] = None,
    ) -> list[tuple[str, LayerRole, LayerDecision, str]]:
        """Return (element_key, role, decision, rationale) tuples."""
        pass


class IPlacementEngine(ABC):
    """Interface for converting normalized bboxes to pixel-space geometry."""

    @abstractmethod
    def place(
        self, spec: RedesignSpecification, canvas: CanvasTransform
    ) -> dict[str, VisualBoundingBox]:
        """Convert normalized BoundingBox entries on spec to pixel VisualBoundingBox."""
        pass

    @abstractmethod
    def resolve_focal_zone(
        self, spec: RedesignSpecification, canvas: CanvasTransform
    ) -> Optional[VisualBoundingBox]:
        """Resolve focal zone to pixel VisualBoundingBox."""
        pass

    @abstractmethod
    def resolve_text_zones(
        self, spec: RedesignSpecification, canvas: CanvasTransform
    ) -> TextPlacement:
        """Resolve text placement and avoid zones to pixel space."""
        pass


class ITransformEngine(ABC):
    """Interface for layer transformation resolution."""

    @abstractmethod
    def resolve(
        self,
        pixel_bbox: Optional[VisualBoundingBox],
        decision: LayerDecision,
        crop_tighter: bool,
    ) -> LayerTransform:
        """Resolve transform for a given pixel bounding box and decision."""
        pass


class IMaskManager(ABC):
    """Interface for binding and configuring masks."""

    @abstractmethod
    def bind(self, registry: IAssetRegistry, role: LayerRole) -> Optional[MaskReference]:
        """Bind VRE mask path and checksum for a layer role."""
        pass

    @abstractmethod
    def feather(self, mask_ref: MaskReference, feather_px: int) -> MaskReference:
        """Return a copy of MaskReference with updated feathering."""
        pass


class ILayerManager(ABC):
    """Interface for layer ordering and grouping."""

    @abstractmethod
    def order(self, layers: list[CompositionLayer]) -> list[CompositionLayer]:
        """Order layers by z-index and deterministically."""
        pass

    @abstractmethod
    def group(self, layers: list[CompositionLayer]) -> list[LayerGroup]:
        """Group layers by role."""
        pass


class ICompositionValidator(ABC):
    """Interface for workspace validation."""

    @abstractmethod
    def validate(self, workspace: CompositionWorkspace) -> list[str]:
        """Validate workspace integrity. Returns empty list if valid, else error strings."""
        pass


class IGenerationBundleBuilder(ABC):
    """Interface for building GenerationBundle from CompositionWorkspace."""

    @abstractmethod
    def build_generation_bundle(
        self, workspace: CompositionWorkspace, prompt_package_hash: str = ""
    ) -> GenerationBundle:
        """Flatten CompositionWorkspace to ComfyUI-ready GenerationBundle."""
        pass


class IMetadataBuilder(ABC):
    """Interface for workspace metadata and statistics assembly."""

    @abstractmethod
    def build(
        self,
        video_id: str,
        manifest: VisualReferenceManifest,
        spec: RedesignSpecification,
        package: PromptPackage,
    ) -> WorkspaceMetadata:
        """Build WorkspaceMetadata from upstream artifacts."""
        pass

    @abstractmethod
    def statistics(self, layers: list[CompositionLayer]) -> WorkspaceStatistics:
        """Compute WorkspaceStatistics from layer list."""
        pass


class IWorkspaceManager(ABC):
    """Interface for workspace persistence, loading, and cache/resume management."""

    @abstractmethod
    def target_dir(self, video_id: str) -> Path:
        """Get target workspace directory for video_id."""
        pass

    @abstractmethod
    def persist(self, workspace: CompositionWorkspace) -> Path:
        """Atomically persist workspace to disk."""
        pass

    @abstractmethod
    def load(self, video_id: str) -> CompositionWorkspace:
        """Load workspace from disk."""
        pass

    @abstractmethod
    def resume(
        self, video_id: str, expected_hashes: dict[str, str]
    ) -> Optional[CompositionWorkspace]:
        """Resume cached workspace if manifest exists and hashes match."""
        pass

    @abstractmethod
    def purge(self, video_id: str) -> bool:
        """Purge workspace directory for video_id."""
        pass

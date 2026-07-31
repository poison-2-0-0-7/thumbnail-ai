"""
interfaces.py
=============

Abstract interfaces for Module 7 Phase 3 (Generation Integration) components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models import CompositionWorkspace, GenerationBundle, GenerationPlan, GenerationProfile
    from image_generator import ReferenceAssets
    from generation_components.conditioning_asset_resolver import GenerationConditioningContext


class IGenerationBundleLoader(ABC):
    """Loads a persisted GenerationBundle by video_id."""

    @abstractmethod
    def load(self, video_id: str) -> GenerationBundle:
        """Load a GenerationBundle for the given video_id."""
        ...


class ICompositionWorkspaceLoader(ABC):
    """Loads a persisted CompositionWorkspace by video_id."""

    @abstractmethod
    def load(self, video_id: str) -> CompositionWorkspace:
        """Load a CompositionWorkspace for the given video_id."""
        ...


class IConditioningAssetResolver(ABC):
    """Normalizes bundle, workspace, legacy reference assets into a GenerationConditioningContext."""

    @abstractmethod
    def resolve(
        self,
        bundle: GenerationBundle | None = None,
        workspace: CompositionWorkspace | None = None,
        reference_assets: ReferenceAssets | None = None,
        profile: GenerationProfile | None = None,
        plan: GenerationPlan | None = None,
    ) -> GenerationConditioningContext:
        """Resolve all conditioning inputs into a unified value object."""
        ...


class INodeFragmentLibrary(ABC):
    """Discovers and loads declarative graph fragments."""

    @abstractmethod
    def discover(self) -> list[Path]:
        """Discover available fragment JSON files."""
        ...

    @abstractmethod
    def load(self, fragment_id: str) -> dict[str, Any]:
        """Load a declarative fragment dictionary by ID."""
        ...


class IWorkflowGraphAssembler(ABC):
    """Assembles base workflow graph and conditioning fragments into a materialized graph."""

    @abstractmethod
    def assemble(
        self,
        base_graph: dict[str, Any],
        fragments: list[dict[str, Any]],
        conditioning: GenerationConditioningContext,
        profile: GenerationProfile,
    ) -> dict[str, Any]:
        """Merge base graph and fragments into a final ComfyUI node graph."""
        ...


class ICapabilityProbe(ABC):
    """Queries ComfyUI server capabilities to verify custom node availability."""

    @abstractmethod
    def installed_node_types(self) -> frozenset[str]:
        """Return the set of installed node class_types from ComfyUI."""
        ...

    @abstractmethod
    def is_fragment_supported(self, fragment: dict[str, Any]) -> bool:
        """Check if all node types required by fragment are installed."""
        ...

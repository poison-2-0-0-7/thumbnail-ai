"""
modules/generation_components
==============================

Module 7 Phase 3: Generation Integration components.
"""

from generation_components.conditioning_asset_resolver import (
    ConditioningAssetResolver,
    GenerationConditioningContext,
    LayerConditioning,
)
from generation_components.generation_bundle_loader import GenerationBundleLoader
from generation_components.interfaces import (
    ICapabilityProbe,
    IConditioningAssetResolver,
    ICompositionWorkspaceLoader,
    IGenerationBundleLoader,
    INodeFragmentLibrary,
    IWorkflowGraphAssembler,
)
from generation_components.workspace_loader import CompositionWorkspaceLoader

from generation_components.node_fragment_library import NodeFragmentLibrary
from generation_components.workflow_graph_assembler import WorkflowGraphAssembler
from generation_components.capability_probe import CapabilityProbe
from generation_components.strategy_pack_resolver import StrategyPackLibrary, StrategyPackResolver
from generation_components.candidate_strategy_planner import CandidateStrategyPlanner
from generation_components.workflow_graph_cache import WorkflowGraphCache
from generation_components.region_plan_validator import RegionPlanValidator
from generation_components.staged_edit_stages import (
    BackgroundEditStage,
    BaseLatentStage,
    HarmonizationStage,
    MaskedCompositeStage,
    ObjectEditStage,
    TypographyStage,
)


__all__ = [
    "GenerationConditioningContext",
    "LayerConditioning",
    "ConditioningAssetResolver",
    "GenerationBundleLoader",
    "CompositionWorkspaceLoader",
    "NodeFragmentLibrary",
    "WorkflowGraphAssembler",
    "CapabilityProbe",
    "StrategyPackLibrary",
    "StrategyPackResolver",
    "CandidateStrategyPlanner",
    "WorkflowGraphCache",
    "RegionPlanValidator",
    "BaseLatentStage",
    "MaskedCompositeStage",
    "BackgroundEditStage",
    "ObjectEditStage",
    "TypographyStage",
    "HarmonizationStage",
    "IGenerationBundleLoader",
    "ICompositionWorkspaceLoader",
    "IConditioningAssetResolver",
    "INodeFragmentLibrary",
    "IWorkflowGraphAssembler",
    "ICapabilityProbe",
]





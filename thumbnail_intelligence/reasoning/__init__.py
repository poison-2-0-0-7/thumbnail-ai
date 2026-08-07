"""
reasoning
=========

Strategic Reasoning Coordinator Foundation for Thumbnail AI (Phase 3.4A).
Orchestrates multi-reasoner strategic intelligence over the grounded NormalizedEvidenceGraph.

Provides:
- Strongly typed reasoning models, contracts, and trace steps
- Abstract interfaces for Narrative, Audience, Creator, Brand, Priority, Risk, and Strategy reasoners
- Pluggable ReasonerRegistry with topological dependency resolution
- ReasoningCoordinator orchestrating reasoner execution and context synthesis
- ReasoningPipeline high-level execution facade
- Comprehensive structured exception hierarchy
"""

from __future__ import annotations

from thumbnail_intelligence.reasoning.config import ReasoningConfig
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.exceptions import (
    CircularDependencyError,
    ContextConstructionError,
    CoordinatorError,
    CoordinatorTimeoutError,
    DuplicateReasonerError,
    EmptyEvidenceGraphError,
    GroundingEnforcementError,
    InvalidReasonerError,
    MissingDependencyError,
    PipelineError,
    PipelineExecutionError,
    ReasonerExecutionError,
    ReasonerNotFoundError,
    ReasonerValidationError,
    ReasoningError,
    RegistryError,
)
from thumbnail_intelligence.reasoning.interfaces import (
    AudienceReasoner,
    BaseReasoner,
    BrandReasoner,
    CreatorReasoner,
    NarrativeReasoner,
    PriorityReasoner,
    RiskReasoner,
    StrategyRanker,
)
from thumbnail_intelligence.reasoning.models import (
    AudienceReasoningOutput,
    BrandReasoningOutput,
    CreatorReasoningOutput,
    DecisionTree,
    DecisionTreeNode,
    NarrativeReasoningOutput,
    PriorityReasoningOutput,
    RankedStrategy,
    ReasonerContract,
    ReasonerType,
    ReasoningRisk,
    ReasoningTraceStep,
    RiskReasoningOutput,
    StrategyRankingOutput,
)
from thumbnail_intelligence.reasoning.pipeline import ReasoningPipeline
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry

from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
    ArcStep,
    CandidateNarrative,
    NarrativeArc,
    NarrativeResult,
    NarrativeType,
    VisualFocusCandidate,
)
from thumbnail_intelligence.reasoning.narrative_reasoner import (
    NarrativeReasoner as ConcreteNarrativeReasoner,
)

from thumbnail_intelligence.reasoning.audience_models import (
    AudienceResult,
    CandidateAudience,
    CognitiveLoadLevel,
    ViewerIntent,
    ViewerKnowledgeLevel,
    ViewerPersona,
)
from thumbnail_intelligence.reasoning.audience_reasoner import (
    AudienceReasoner as ConcreteAudienceReasoner,
)

from thumbnail_intelligence.reasoning.creator_models import (
    CandidateCreatorStyle,
    CreatorArchetype,
    CreatorResult,
    VisualIdentityStyle,
)
from thumbnail_intelligence.reasoning.creator_reasoner import (
    CreatorReasoner as ConcreteCreatorReasoner,
)

from thumbnail_intelligence.reasoning.brand_models import (
    BrandPreservationPriority,
    BrandResult,
    CandidateBrandInterpretation,
    VisualElementPreservation,
)
from thumbnail_intelligence.reasoning.brand_reasoner import (
    BrandReasoner as ConcreteBrandReasoner,
)

from thumbnail_intelligence.reasoning.priority_models import (
    AttentionFlowStep,
    BackgroundPriority,
    CandidateHierarchy,
    ElementPriorityLevel,
    HierarchyTier,
    PriorityResult,
    VisualHierarchyNode,
)
from thumbnail_intelligence.reasoning.priority_reasoner import (
    PriorityReasoner as ConcretePriorityReasoner,
)

from thumbnail_intelligence.reasoning.risk_models import (
    CandidateRiskProfile,
    DetectedRisk,
    RiskCategory,
    RiskLikelihood,
    RiskResult,
    RiskSeverity,
)
from thumbnail_intelligence.reasoning.risk_reasoner import (
    RiskReasoner as ConcreteRiskReasoner,
)

from thumbnail_intelligence.reasoning.strategy_models import (
    StrategyArchetype,
    StrategyCandidate,
    StrategyDecision,
    TradeoffAnalysis,
)
from thumbnail_intelligence.reasoning.strategy_ranker import (
    StrategyRanker as ConcreteStrategyRanker,
)
from thumbnail_intelligence.reasoning.interfaces import (
    StrategicReasoningValidator,
)
from thumbnail_intelligence.reasoning.validator_models import (
    ConflictType,
    DetectedConflict,
    ReasoningValidation,
    ValidatedReasoningPackage,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
    ValidationTraceStep,
)
from thumbnail_intelligence.reasoning.validator import (
    StrategicReasoningValidator as ConcreteStrategicReasoningValidator,
)
from thumbnail_intelligence.reasoning.interfaces import (
    DesignBriefGeneratorInterface,
)
from thumbnail_intelligence.reasoning.design_brief_models import (
    AudienceBrief,
    BrandBrief,
    BriefMetadata,
    CameraBrief,
    ColorBrief,
    CompositionBrief,
    CreatorBrief,
    DesignBrief,
    ExecutionConstraintsBrief,
    LightingBrief,
    NarrativeBrief,
    ObjectsBrief,
    TypographyBrief,
    ValidationBrief,
)
from thumbnail_intelligence.reasoning.design_brief_generator import (
    DesignBriefGenerator,
    DesignBriefGenerator as ConcreteDesignBriefGenerator,
)
from thumbnail_intelligence.reasoning.execution_plan_models import (
    ExecutionGraph,
    ExecutionMetadata,
    ExecutionPlan,
    ExecutionStep,
    ExecutionStepType,
    ResourceEstimates,
    RetryPolicy,
)
from thumbnail_intelligence.reasoning.execution_planner import (
    ExecutionPlanner,
    ExecutionPlanner as ConcreteExecutionPlanner,
)
from thumbnail_intelligence.reasoning.spatial_composition_models import (
    AnchorPoint,
    BoundingBox,
    CanvasSpecification,
    CompositionEdge,
    CompositionGraph,
    CompositionLayerPlane,
    CompositionRelationshipType,
    CompositionRule,
    PlacementInstructions,
    SafeZone,
    SpatialComposition,
    TypographyLayout,
    VisualElementPlacement,
)
from thumbnail_intelligence.reasoning.spatial_composition_planner import (
    SpatialCompositionPlanner,
    SpatialCompositionPlanner as ConcreteSpatialCompositionPlanner,
)
from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    PixelBoundingBox,
    RenderAssetReference,
    RenderBackgroundInstruction,
    RenderExecutionPackage,
    RenderLayerEntry,
    RenderLightingInstruction,
    RenderMaskInstruction,
    RenderOperation,
    RenderOperationType,
    RenderPackageMetadata,
    RenderPlacementCoordinate,
    RenderSceneGraph,
    RenderSceneGraphNode,
    RenderTypographyInstruction,
)
from thumbnail_intelligence.reasoning.renderer_adapter import (
    BaseRendererAdapter,
    FutureComfyUIAdapter,
    FutureCustomAdapter,
    FutureFluxAdapter,
    FutureImagenAdapter,
    RendererAdapter,
    RendererV2Adapter,
)

__all__ = [
    # Coordinator & Pipeline
    "ReasoningCoordinator",
    "ReasoningPipeline",
    "ReasonerRegistry",
    "ReasoningConfig",
    "ReasoningContext",
    # Abstract Interfaces
    "BaseReasoner",
    "NarrativeReasoner",
    "AudienceReasoner",
    "CreatorReasoner",
    "BrandReasoner",
    "PriorityReasoner",
    "RiskReasoner",
    "StrategyRanker",
    "StrategicReasoningValidator",
    "DesignBriefGeneratorInterface",
    "ExecutionPlannerInterface",
    "SpatialCompositionPlannerInterface",
    "RendererAdapterInterface",
    # Phase 3.4B Narrative Reasoner & Models
    "ConcreteNarrativeReasoner",
    "NarrativeResult",
    "CandidateNarrative",
    "NarrativeType",
    "NarrativeArc",
    "ArcStage",
    "ArcStep",
    "VisualFocusCandidate",
    # Phase 3.4C Audience & Creator Reasoners & Models
    "ConcreteAudienceReasoner",
    "AudienceResult",
    "CandidateAudience",
    "ViewerIntent",
    "ViewerKnowledgeLevel",
    "CognitiveLoadLevel",
    "ViewerPersona",
    "ConcreteCreatorReasoner",
    "CreatorResult",
    "CandidateCreatorStyle",
    "CreatorArchetype",
    "VisualIdentityStyle",
    # Phase 3.4D Brand Reasoner & Models
    "ConcreteBrandReasoner",
    "BrandResult",
    "CandidateBrandInterpretation",
    "VisualElementPreservation",
    "BrandPreservationPriority",
    # Phase 3.4E Priority Reasoner & Models
    "ConcretePriorityReasoner",
    "PriorityResult",
    "CandidateHierarchy",
    "VisualHierarchyNode",
    "AttentionFlowStep",
    "HierarchyTier",
    "ElementPriorityLevel",
    "BackgroundPriority",
    # Phase 3.4F Risk Reasoner & Models
    "ConcreteRiskReasoner",
    "RiskResult",
    "DetectedRisk",
    "CandidateRiskProfile",
    "RiskCategory",
    "RiskSeverity",
    "RiskLikelihood",
    # Phase 3.4G Strategy Ranker & Models
    "ConcreteStrategyRanker",
    "StrategyDecision",
    "StrategyCandidate",
    "StrategyArchetype",
    "TradeoffAnalysis",
    # Phase 3.4H Strategic Reasoning Validator & Models
    "ConcreteStrategicReasoningValidator",
    "ReasoningValidation",
    "ValidatedReasoningPackage",
    "ValidationIssue",
    "ValidationIssueType",
    "ValidationSeverity",
    "ValidationStatus",
    "ValidationTraceStep",
    "DetectedConflict",
    "ConflictType",
    # Phase 3.5 DesignBrief Generator & Models
    "DesignBriefGenerator",
    "ConcreteDesignBriefGenerator",
    "DesignBrief",
    "BriefMetadata",
    "NarrativeBrief",
    "AudienceBrief",
    "CreatorBrief",
    "BrandBrief",
    "CompositionBrief",
    "TypographyBrief",
    "ColorBrief",
    "LightingBrief",
    "CameraBrief",
    "ObjectsBrief",
    "ExecutionConstraintsBrief",
    "ValidationBrief",
    # Phase 3.6 Execution Planner & Models
    "ExecutionPlanner",
    "ConcreteExecutionPlanner",
    "ExecutionPlan",
    "ExecutionGraph",
    "ExecutionStep",
    "ExecutionStepType",
    "ExecutionMetadata",
    "ResourceEstimates",
    "RetryPolicy",
    # Phase 3.7 Spatial Composition Planner & Models
    "SpatialCompositionPlanner",
    "ConcreteSpatialCompositionPlanner",
    "SpatialComposition",
    "CompositionGraph",
    "PlacementInstructions",
    "TypographyLayout",
    "CanvasSpecification",
    "VisualElementPlacement",
    "BoundingBox",
    "AnchorPoint",
    "SafeZone",
    "CompositionRule",
    "CompositionLayerPlane",
    "CompositionRelationshipType",
    "CompositionEdge",
    # Phase 3.8 Renderer Adapter & Models
    "BaseRendererAdapter",
    "RendererV2Adapter",
    "RendererAdapter",
    "FutureComfyUIAdapter",
    "FutureFluxAdapter",
    "FutureImagenAdapter",
    "FutureCustomAdapter",
    "RenderExecutionPackage",
    "RenderSceneGraph",
    "RenderSceneGraphNode",
    "RenderOperation",
    "RenderOperationType",
    "RenderAssetReference",
    "RenderMaskInstruction",
    "RenderTypographyInstruction",
    "RenderBackgroundInstruction",
    "RenderLightingInstruction",
    "RenderLayerEntry",
    "RenderPlacementCoordinate",
    "PixelBoundingBox",
    "RenderPackageMetadata",
    # Domain Models & Outputs
    "ReasonerType",
    "ReasonerContract",
    "ReasoningTraceStep",
    "ReasoningRisk",
    "RankedStrategy",
    "DecisionTreeNode",
    "DecisionTree",
    "NarrativeReasoningOutput",
    "AudienceReasoningOutput",
    "CreatorReasoningOutput",
    "BrandReasoningOutput",
    "PriorityReasoningOutput",
    "RiskReasoningOutput",
    "StrategyRankingOutput",
    # Exceptions
    "ReasoningError",
    "RegistryError",
    "ReasonerNotFoundError",
    "DuplicateReasonerError",
    "MissingDependencyError",
    "CircularDependencyError",
    "InvalidReasonerError",
    "CoordinatorError",
    "ReasonerExecutionError",
    "ReasonerValidationError",
    "ContextConstructionError",
    "EmptyEvidenceGraphError",
    "CoordinatorTimeoutError",
    "PipelineError",
    "PipelineExecutionError",
    "GroundingEnforcementError",
]


"""
Execution Engine Package (Phase 4.1 — Execution Engine Foundation).
"""

from renderer_v2.execution.adapters import (
    AssetLoaderAdapter,
    BackgroundGeneratorAdapter,
    ExporterAdapter,
    ImageValidatorAdapter,
    LayerComposerAdapter,
    LightingEngineAdapter,
    QualityValidatorAdapter,
    SubjectExtractorAdapter,
    TypographyRendererAdapter,
)
from renderer_v2.execution.context import RenderJobContext
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.engine import ExecutionEngine
from renderer_v2.execution.exceptions import (
    ExecutionEngineError,
    GraphValidationError,
    JobCancellationError,
    OperationExecutionError,
    PackageValidationError,
    StageExecutionError,
    WorkspaceValidationError,
)
from renderer_v2.execution.fsm import ExecutionFSM, ExecutionState, InvalidStateTransitionError
from renderer_v2.execution.graph import ExecutionGraph, ExecutionNode, ExecutionScheduler, NodeStatus
from renderer_v2.execution.models import LayerBuffer, SceneInstance, WorkspaceSnapshot
from renderer_v2.execution.reports import (
    CritiqueReport,
    RenderJobReport,
    RenderJobStatus,
    StageExecutionReport,
    StageStatus,
)
from renderer_v2.execution.stages import (
    AssetLoader,
    BackgroundGenerator,
    BaseExecutionStage,
    Exporter,
    ImageValidator,
    LayerComposer,
    LightingEngine,
    ModelManager,
    QualityValidator,
    SubjectEnhancer,
    SubjectExtractor,
    TypographyRenderer,
)
from renderer_v2.execution.validation import (
    ExecutionValidator,
    GraphValidator,
    OperationValidator,
    PackageValidator,
    WorkspaceValidator,
)
from renderer_v2.execution.workspace import RenderWorkspace, WorkspaceState

__all__ = [
    # Engine
    "ExecutionEngine",
    # Context & Workspace
    "RenderJobContext",
    "RenderWorkspace",
    "WorkspaceState",
    "LayerBuffer",
    "SceneInstance",
    "WorkspaceSnapshot",
    # Reports
    "StageExecutionReport",
    "RenderJobReport",
    "CritiqueReport",
    "StageStatus",
    "RenderJobStatus",
    # FSM
    "ExecutionFSM",
    "ExecutionState",
    "InvalidStateTransitionError",
    # Graph & Scheduler
    "ExecutionGraph",
    "ExecutionScheduler",
    "ExecutionNode",
    "NodeStatus",
    # Dispatcher & Stages & Adapters
    "ExecutionDispatcher",
    "BaseExecutionStage",
    "AssetLoader",
    "ModelManager",
    "BackgroundGenerator",
    "SubjectExtractor",
    "SubjectEnhancer",
    "LightingEngine",
    "TypographyRenderer",
    "LayerComposer",
    "ImageValidator",
    "QualityValidator",
    "Exporter",
    "AssetLoaderAdapter",
    "BackgroundGeneratorAdapter",
    "SubjectExtractorAdapter",
    "LightingEngineAdapter",
    "TypographyRendererAdapter",
    "LayerComposerAdapter",
    "ImageValidatorAdapter",
    "QualityValidatorAdapter",
    "ExporterAdapter",
    # Validation
    "PackageValidator",
    "GraphValidator",
    "WorkspaceValidator",
    "OperationValidator",
    "ExecutionValidator",
    # Exceptions
    "ExecutionEngineError",
    "PackageValidationError",
    "GraphValidationError",
    "WorkspaceValidationError",
    "OperationExecutionError",
    "StageExecutionError",
    "JobCancellationError",
]

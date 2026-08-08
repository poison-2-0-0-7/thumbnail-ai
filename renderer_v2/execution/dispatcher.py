"""
dispatcher.py
=============

Execution Dispatcher for Phase 4.1 Execution Engine.
Dispatches RenderOperation primitives to corresponding placeholder stage handlers.
Maps all 14 RenderOperationType primitives to stage instances.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderOperation, RenderOperationType
from renderer_v2.execution.context import RenderJobContext
from renderer_v2.execution.exceptions import OperationExecutionError
from renderer_v2.execution.reports import StageExecutionReport, StageStatus
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
from renderer_v2.execution.workspace import RenderWorkspace

logger = logging.getLogger(__name__)


class ExecutionDispatcher:
    """
    Dispatcher routing RenderOperations to registered renderer stage adapters.
    Supports all 14 RenderOperationType primitives per architecture specification.
    """

    def __init__(
        self,
        use_placeholders: bool = False,
        runtime_manager: Optional[Any] = None,
    ) -> None:
        self._stages: Dict[str, BaseExecutionStage] = {}
        self._type_mapping: Dict[RenderOperationType, BaseExecutionStage] = {}
        self.runtime_manager = runtime_manager
        self._register_default_stages(use_placeholders=use_placeholders)

    def _register_default_stages(self, use_placeholders: bool = False) -> None:
        """Instantiate and register stage implementations for all 14 operation primitives."""
        if use_placeholders:
            asset_loader = AssetLoader()
            bg_generator = BackgroundGenerator()
            subject_extractor = SubjectExtractor()
            lighting_engine = LightingEngine()
            typography_renderer = TypographyRenderer()
            layer_composer = LayerComposer()
            image_validator = ImageValidator()
            quality_validator = QualityValidator()
            exporter = Exporter()
        else:
            asset_loader = AssetLoaderAdapter()
            bg_generator = BackgroundGeneratorAdapter(runtime_manager=self.runtime_manager)
            subject_extractor = SubjectExtractorAdapter(runtime_manager=self.runtime_manager)
            lighting_engine = LightingEngineAdapter()
            typography_renderer = TypographyRendererAdapter()
            layer_composer = LayerComposerAdapter()
            image_validator = ImageValidatorAdapter()
            quality_validator = QualityValidatorAdapter()
            exporter = ExporterAdapter()

        model_manager = ModelManager()  # Placeholder per Phase 4.2 scope
        subject_enhancer = SubjectEnhancer()  # Placeholder per Phase 4.2 scope

        # Register stages
        for stage in [
            asset_loader,
            model_manager,
            bg_generator,
            subject_extractor,
            subject_enhancer,
            lighting_engine,
            typography_renderer,
            layer_composer,
            image_validator,
            quality_validator,
            exporter,
        ]:
            self.register_stage(stage)

        # Map RenderOperationType primitives to stages
        self._type_mapping = {
            RenderOperationType.LOAD_ASSET: asset_loader,
            RenderOperationType.PREPARE_CANVAS: layer_composer,
            RenderOperationType.GENERATE_BACKGROUND: bg_generator,
            RenderOperationType.EXTRACT_SUBJECT: subject_extractor,
            RenderOperationType.ENHANCE_SUBJECT: subject_enhancer,
            RenderOperationType.APPLY_LIGHTING: lighting_engine,
            RenderOperationType.GENERATE_SHADOW: lighting_engine,
            RenderOperationType.RENDER_TYPOGRAPHY: typography_renderer,
            RenderOperationType.COMPOSE_LAYER: layer_composer,
            RenderOperationType.APPLY_COLOR_GRADE: layer_composer,
            RenderOperationType.ADJUST_CONTRAST: layer_composer,
            RenderOperationType.EVALUATE_QUALITY: quality_validator,
            RenderOperationType.COMPOSITE_FINAL: layer_composer,
            RenderOperationType.CLEANUP_BUFFERS: model_manager,
        }

    def register_stage(self, stage: BaseExecutionStage) -> None:
        """Register a stage handler instance."""
        self._stages[stage.stage_name] = stage

    def map_operation_type(self, op_type: RenderOperationType, stage: BaseExecutionStage) -> None:
        """Override mapping for an operation primitive to a custom stage handler."""
        self.register_stage(stage)
        self._type_mapping[op_type] = stage

    def get_stage(self, stage_name: str) -> Optional[BaseExecutionStage]:
        """Retrieve registered stage handler by stage_name."""
        return self._stages.get(stage_name)

    def get_stage_for_type(self, op_type: RenderOperationType) -> BaseExecutionStage:
        """Retrieve stage handler registered for an operation primitive."""
        if op_type not in self._type_mapping:
            raise OperationExecutionError(f"No stage registered for RenderOperationType '{op_type.value}'")
        return self._type_mapping[op_type]

    def dispatch(
        self,
        operation: RenderOperation,
        context: RenderJobContext,
        workspace: RenderWorkspace,
    ) -> StageExecutionReport:
        """
        Dispatch a RenderOperation to its mapped placeholder stage.
        Invokes stage.execute(), runs stage.validate(), and records workspace status.
        """
        stage = self.get_stage_for_type(operation.op_type)
        logger.debug(f"Dispatching operation '{operation.op_id}' ({operation.op_type.value}) -> {stage.stage_name}")

        try:
            report = stage.execute(operation, context, workspace)

            # Post-execution stage validation
            val_errors = stage.validate(operation, workspace)
            if val_errors:
                report.validation_notes.extend([f"Validation issue: {e}" for e in val_errors])
                if report.status == StageStatus.SUCCESS:
                    report.status = StageStatus.SUCCESS_WITH_DEGRADATION

            # Record in workspace
            workspace.record_stage_report(report)
            workspace.record_operation(
                op_id=operation.op_id,
                op_type=operation.op_type.value,
                status=report.status.value,
                latency_s=report.latency_s,
            )
            return report

        except Exception as e:
            logger.exception(f"Error dispatching operation '{operation.op_id}' to stage '{stage.stage_name}': {e}")
            err_report = StageExecutionReport(
                stage=stage.stage_name,
                op_id=operation.op_id,
                status=StageStatus.FAILED_FATAL,
                latency_s=0.0,
                error_message=str(e),
            )
            workspace.record_stage_report(err_report)
            workspace.record_operation(
                op_id=operation.op_id,
                op_type=operation.op_type.value,
                status=StageStatus.FAILED_FATAL.value,
                latency_s=0.0,
                details={"error": str(e)},
            )
            return err_report

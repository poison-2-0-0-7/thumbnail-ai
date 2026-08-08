"""
pipeline.py
===========

Renderer V2 Main Pipeline Orchestrator for Phase 4.5 Final Integration.
Encapsulates the complete architectural stack:
RenderExecutionPackage -> ExecutionEngine -> ExecutionDispatcher -> Stage Adapters -> ModelRuntimeManager -> Renderer Modules -> RenderWorkspace -> RendererV2Adapter -> Final Composite -> Image Export.

Zero architecture layers bypassed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage
from thumbnail_intelligence.reasoning.spatial_composition_models import SpatialComposition
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.engine import ExecutionEngine
from renderer_v2.execution.reports import RenderJobReport, RenderJobStatus
from renderer_v2.execution.workspace import RenderWorkspace
from renderer_v2.runtime.manager import ModelRuntimeManager

logger = logging.getLogger(__name__)


class RendererV2PipelineError(RuntimeError):
    """Exception raised for high-level Renderer V2 pipeline failures."""
    pass


class RendererV2Pipeline:
    """Production Renderer V2 Pipeline orchestrating the full execution stack end-to-end."""

    def __init__(
        self,
        runtime_manager: Optional[ModelRuntimeManager] = None,
        dispatcher: Optional[ExecutionDispatcher] = None,
        engine: Optional[ExecutionEngine] = None,
    ) -> None:
        self.runtime_manager = runtime_manager or ModelRuntimeManager()
        self.dispatcher = dispatcher or ExecutionDispatcher(
            use_placeholders=False,
            runtime_manager=self.runtime_manager,
        )
        self.engine = engine or ExecutionEngine(dispatcher=self.dispatcher)
        self.adapter = RendererV2Adapter()

    def render(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
        output_path: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> RenderJobReport:
        """Translate SpatialComposition + ExecutionPlan and render final thumbnail end-to-end.

        Args:
            composition: Renderer-independent spatial layout.
            plan: Renderer-independent execution plan.
            output_path: Destination file path for rendered thumbnail raster.
            context_overrides: Additional runtime execution metadata overrides.

        Returns:
            RenderJobReport containing final status, stage reports, latency, and output image path.
        """
        package = self.adapter.translate(composition, plan)
        return self.render_package(package, output_path=output_path, context_overrides=context_overrides)

    def render_package(
        self,
        package: RenderExecutionPackage,
        output_path: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> RenderJobReport:
        """Execute a RenderExecutionPackage through ExecutionEngine and return RenderJobReport."""
        if not package:
            raise RendererV2PipelineError("Cannot execute rendering pipeline with None package.")

        overrides = (context_overrides or {}).copy()
        if output_path is not None:
            overrides["output_path"] = str(output_path)

        logger.info(f"=== Starting Renderer V2 Pipeline Run for package '{package.metadata.package_id}' ===")
        report = self.engine.execute(package, context_overrides=overrides)

        # Validate pipeline output consistency
        self.validate_report_output(report)
        logger.info(f"=== Renderer V2 Pipeline Completed with status '{report.status.value}' (Output: '{report.output_image_path}') ===")
        return report

    def validate_report_output(self, report: RenderJobReport) -> None:
        """Validate rendered thumbnail output file integrity and dimensions on disk."""
        if report.status == RenderJobStatus.FAILED_FATAL:
            logger.warning(f"Pipeline finished with FAILED_FATAL status: {report.errors}")
            return

        out_path = report.output_image_path
        if not out_path or not os.path.exists(out_path):
            raise RendererV2PipelineError(f"Rendered thumbnail file not found at expected path '{out_path}'")

        if os.path.getsize(out_path) == 0:
            raise RendererV2PipelineError(f"Exported thumbnail file '{out_path}' is empty (0 bytes)")

        img = cv2.imread(out_path)
        if img is None:
            raise RendererV2PipelineError(f"Failed to decode exported thumbnail raster from '{out_path}'")

        if img.ndim != 3 or img.shape[2] != 3:
            raise RendererV2PipelineError(f"Invalid exported image shape {img.shape}; expected 3-channel RGB image.")


# Production alias
RendererV2 = RendererV2Pipeline

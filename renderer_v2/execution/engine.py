"""
engine.py
=========

Execution Engine entry point for Phase 4.1 Execution Engine Foundation.
Consumes ONLY a RenderExecutionPackage (and nothing else), orchestrates state machine,
job context, workspace, execution graph, scheduling, operation dispatching, and reports.
NO AI model execution, NO image generation, NO diffusion, NO segmentation, NO renderer integration.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    RenderExecutionPackage,
    RenderOperation,
    RenderOperationType,
)
from renderer_v2.execution.context import RenderJobContext
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.exceptions import (
    ExecutionEngineError,
    JobCancellationError,
    PackageValidationError,
)
from renderer_v2.execution.fsm import ExecutionFSM, ExecutionState
from renderer_v2.execution.graph import ExecutionGraph, ExecutionScheduler
from renderer_v2.execution.reports import (
    RenderJobReport,
    RenderJobStatus,
    StageExecutionReport,
    StageStatus,
)
from renderer_v2.execution.validation import (
    ExecutionValidator,
    GraphValidator,
    PackageValidator,
)
from renderer_v2.execution.workspace import RenderWorkspace

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Master ExecutionEngine orchestrator for Renderer V2 Phase 4.1.
    Strictly consumes ONLY RenderExecutionPackage data contracts.
    """

    def __init__(self, dispatcher: Optional[ExecutionDispatcher] = None) -> None:
        self.dispatcher = dispatcher or ExecutionDispatcher()

    def execute(
        self,
        package: RenderExecutionPackage,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> RenderJobReport:
        """
        Main execution entry point.
        Consumes RenderExecutionPackage, runs validation, constructs context and workspace,
        builds execution graph, dispatches operations to stage handlers, and returns RenderJobReport.

        Raises TypeError if package is not a RenderExecutionPackage instance.
        """
        t0 = time.time()

        # Invariant Enforcement: Consume ONLY RenderExecutionPackage
        if not isinstance(package, RenderExecutionPackage):
            raise TypeError(
                f"ExecutionEngine.execute() consumes ONLY RenderExecutionPackage instances. Received invalid type: {type(package).__name__}"
            )

        fsm = ExecutionFSM()

        try:
            fsm.transition_to(ExecutionState.INITIALIZING)

            # Step 1: Package Validation
            fsm.transition_to(ExecutionState.VALIDATING)
            pkg_errors = PackageValidator.validate(package)
            if pkg_errors:
                fsm.transition_to(ExecutionState.FAILED, {"reason": "Package validation failed"})
                return RenderJobReport(
                    job_id=f"job_invalid_{int(t0)}",
                    correlation_id="corr_invalid",
                    attempt=1,
                    status=RenderJobStatus.FAILED_FATAL,
                    total_latency_s=time.time() - t0,
                    errors=pkg_errors,
                    validation_summary={"valid": False, "errors": pkg_errors},
                )

            # Step 2: Context & Workspace Initialization
            overrides = context_overrides or {}
            job_id = overrides.get("job_id")
            correlation_id = overrides.get("correlation_id")
            attempt = overrides.get("attempt", 1)

            context = RenderJobContext(
                package=package,
                job_id=job_id,
                correlation_id=correlation_id,
                attempt=attempt,
                execution_metadata=overrides,
            )

            workspace = RenderWorkspace(
                job_id=context.job_id,
                canvas_width_px=package.scene_graph.canvas_width_px,
                canvas_height_px=package.scene_graph.canvas_height_px,
            )
            workspace.initialize(package)

            # Step 3: Build & Validate Execution Graph
            fsm.transition_to(ExecutionState.SCHEDULING)
            graph = ExecutionGraph.build_from_package(package)
            GraphValidator.validate_or_raise(graph)

            scheduler = ExecutionScheduler(graph)
            stage_reports: List[StageExecutionReport] = []
            peak_vram_gb: float = 0.0

            # Step 4: Dispatch Loop
            fsm.transition_to(ExecutionState.DISPATCHING)

            while not scheduler.is_complete():
                # Cancellation Check
                context.check_cancellation()

                runnable_nodes = scheduler.get_next_runnable_operations()
                if not runnable_nodes:
                    break

                for node in runnable_nodes:
                    context.check_cancellation()
                    scheduler.mark_started(node.op_id)

                    fsm.transition_to(
                        ExecutionState.RUNNING_STAGE,
                        {"op_id": node.op_id, "op_type": node.op_type.value},
                    )

                    # Dispatch operation to placeholder stage
                    report = self.dispatcher.dispatch(
                        operation=node.operation,
                        context=context,
                        workspace=workspace,
                    )
                    stage_reports.append(report)

                    if report.vram_peak_gb > peak_vram_gb:
                        peak_vram_gb = report.vram_peak_gb

                    # Stage Validation & Status Check
                    fsm.transition_to(
                        ExecutionState.VALIDATING_STAGE,
                        {"op_id": node.op_id, "status": report.status.value},
                    )

                    if report.status == StageStatus.FAILED_FATAL:
                        scheduler.mark_failed(node.op_id)
                        fsm.transition_to(
                            ExecutionState.FAILED,
                            {"failed_op": node.op_id, "error": report.error_message},
                        )
                        total_latency = context.mark_completed()
                        return RenderJobReport(
                            job_id=context.job_id,
                            correlation_id=context.correlation_id,
                            attempt=context.attempt,
                            status=RenderJobStatus.FAILED_FATAL,
                            stage_reports=stage_reports,
                            total_latency_s=total_latency,
                            vram_peak_gb=peak_vram_gb,
                            errors=[f"Operation '{node.op_id}' failed: {report.error_message}"],
                            validation_summary={"valid": False},
                        )

                    scheduler.mark_completed(node.op_id)

                    # Checkpoint Workspace after stage
                    fsm.transition_to(
                        ExecutionState.CHECKPOINTING,
                        {"op_id": node.op_id, "stage": report.stage},
                    )
                    workspace.checkpoint(f"after_{node.op_id}")

                    fsm.transition_to(ExecutionState.DISPATCHING)

            # Step 5: Finalize & Export
            exporter = self.dispatcher.get_stage("Exporter")
            if exporter is not None:
                export_op = RenderOperation(
                    op_id=f"op_export_{context.job_id}",
                    op_type=RenderOperationType.COMPOSITE_FINAL,
                )
                export_report = exporter.execute(export_op, context, workspace)
                stage_reports.append(export_report)

            val_summary = ExecutionValidator.validate(context, workspace)
            total_latency = context.mark_completed()

            has_degraded_stage = any(
                r.status == StageStatus.SUCCESS_WITH_DEGRADATION for r in stage_reports
            )
            final_status = (
                RenderJobStatus.SUCCESS_WITH_DEGRADATION
                if has_degraded_stage
                else RenderJobStatus.SUCCESS
            )

            fsm.transition_to(
                ExecutionState.COMPLETED_WITH_DEGRADATION
                if has_degraded_stage
                else ExecutionState.COMPLETED
            )

            out_sink = workspace.intermediate_artifacts.get(
                "exporter_sink", context.get_meta("output_path", f"output/{context.job_id}_final.jpg")
            )

            return RenderJobReport(
                job_id=context.job_id,
                correlation_id=context.correlation_id,
                attempt=context.attempt,
                status=final_status,
                stage_reports=stage_reports,
                total_latency_s=total_latency,
                vram_peak_gb=peak_vram_gb,
                output_image_path=out_sink if isinstance(out_sink, str) else str(out_sink),
                validation_summary=val_summary,
            )

        except JobCancellationError as e:
            fsm.transition_to(ExecutionState.CANCELLED, {"reason": str(e)})
            total_latency = time.time() - t0
            return RenderJobReport(
                job_id=job_id or "job_cancelled",
                correlation_id=correlation_id or "corr_cancelled",
                attempt=attempt if 'attempt' in locals() else 1,
                status=RenderJobStatus.CANCELLED,
                stage_reports=stage_reports if 'stage_reports' in locals() else [],
                total_latency_s=total_latency,
                errors=[str(e)],
                validation_summary={"valid": False, "cancelled": True},
            )

        except Exception as e:
            logger.exception(f"Unhandled execution error in ExecutionEngine: {e}")
            if not fsm.is_terminal:
                fsm.transition_to(ExecutionState.FAILED, {"error": str(e)})
            total_latency = time.time() - t0
            return RenderJobReport(
                job_id=job_id if 'job_id' in locals() and job_id else f"job_err_{int(t0)}",
                correlation_id=correlation_id if 'correlation_id' in locals() and correlation_id else "corr_err",
                attempt=attempt if 'attempt' in locals() else 1,
                status=RenderJobStatus.FAILED_FATAL,
                stage_reports=stage_reports if 'stage_reports' in locals() else [],
                total_latency_s=total_latency,
                errors=[str(e)],
                validation_summary={"valid": False, "error": str(e)},
            )

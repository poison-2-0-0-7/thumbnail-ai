"""
validation.py
=============

Validation suite for the Phase 4.1 Execution Engine.
Includes validators for packages, graphs, workspaces, operations, and final execution outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    RenderExecutionPackage,
    RenderOperation,
)
from renderer_v2.execution.context import RenderJobContext
from renderer_v2.execution.exceptions import (
    GraphValidationError,
    OperationExecutionError,
    PackageValidationError,
    WorkspaceValidationError,
)
from renderer_v2.execution.graph import ExecutionGraph
from renderer_v2.execution.workspace import RenderWorkspace

logger = logging.getLogger(__name__)


class PackageValidator:
    """Validator for incoming RenderExecutionPackage contracts."""

    @staticmethod
    def validate(package: RenderExecutionPackage) -> List[str]:
        """
        Validate RenderExecutionPackage structural integrity, coordinate bounds,
        metadata references, and layer stack non-emptiness.
        """
        errors: List[str] = []

        # Rely on package's built-in validation
        pkg_errors = package.validate_package()
        errors.extend(pkg_errors)

        if not package.render_operations:
            errors.append("RenderExecutionPackage contains zero render_operations.")

        if package.scene_graph.canvas_width_px <= 0 or package.scene_graph.canvas_height_px <= 0:
            errors.append(
                f"Invalid scene graph canvas dimensions: {package.scene_graph.canvas_width_px}x{package.scene_graph.canvas_height_px}"
            )

        # Validate unique op_ids
        seen_ops = set()
        for op in package.render_operations:
            if op.op_id in seen_ops:
                errors.append(f"Duplicate RenderOperation op_id found: '{op.op_id}'")
            seen_ops.add(op.op_id)

        return errors

    @classmethod
    def validate_or_raise(cls, package: RenderExecutionPackage) -> None:
        """Validate package and raise PackageValidationError if non-empty errors."""
        errors = cls.validate(package)
        if errors:
            raise PackageValidationError(
                message=f"RenderExecutionPackage failed validation with {len(errors)} error(s).",
                errors=errors,
            )


class GraphValidator:
    """Validator for ExecutionGraph DAG structures."""

    @staticmethod
    def validate(graph: ExecutionGraph) -> List[str]:
        """Validate graph dependencies and topological ordering."""
        errors = graph.validate_dependencies()
        try:
            graph.validate_and_sort()
        except GraphValidationError as e:
            errors.append(str(e))
        return errors

    @classmethod
    def validate_or_raise(cls, graph: ExecutionGraph) -> None:
        """Validate graph and raise GraphValidationError if invalid."""
        errors = cls.validate(graph)
        if errors:
            raise GraphValidationError(
                message=f"ExecutionGraph failed validation: {'; '.join(errors)}"
            )


class WorkspaceValidator:
    """Validator for RenderWorkspace state integrity."""

    @staticmethod
    def validate(workspace: RenderWorkspace) -> List[str]:
        """Validate workspace dimensions and layer buffer alignment."""
        return workspace.validate_workspace()

    @classmethod
    def validate_or_raise(cls, workspace: RenderWorkspace) -> None:
        """Validate workspace state and raise WorkspaceValidationError if invalid."""
        errors = cls.validate(workspace)
        if errors:
            raise WorkspaceValidationError(
                f"RenderWorkspace failed validation for job {workspace.job_id}: {'; '.join(errors)}"
            )


class OperationValidator:
    """Validator for individual RenderOperation parameters and prerequisite keys."""

    @staticmethod
    def validate(operation: RenderOperation, workspace: RenderWorkspace) -> List[str]:
        """Validate operation parameters and input key availability in workspace."""
        errors: List[str] = []
        if not operation.op_id:
            errors.append("RenderOperation op_id must be non-empty.")

        # Check input keys
        for in_key in operation.input_keys:
            if not workspace.has_layer(in_key) and in_key not in workspace.masks and f"asset:{in_key}" not in workspace.intermediate_artifacts:
                logger.debug(f"Operation '{operation.op_id}' input_key '{in_key}' not yet materialized in workspace.")

        return errors


class ExecutionValidator:
    """Final post-execution validator for assembled job outputs."""

    @staticmethod
    def validate(context: RenderJobContext, workspace: RenderWorkspace) -> Dict[str, Any]:
        """Validate final job state, output layer availability, and report summary."""
        summary: Dict[str, Any] = {
            "valid": True,
            "layers_count": len(workspace.layers),
            "masks_count": len(workspace.masks),
            "instances_count": len(workspace.scene_instances),
            "stage_reports_count": len(workspace.stage_reports),
            "errors": [],
        }

        if not workspace.layers:
            summary["valid"] = False
            summary["errors"].append("RenderWorkspace contains zero layer buffers upon execution completion.")

        return summary

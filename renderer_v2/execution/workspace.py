"""
workspace.py
============

RenderWorkspace implementation for Phase 4.1 Execution Engine.
Serves as the central mutable working state for a single render job.
Stores layer buffers, masks, scene instances, depth map placeholders,
intermediate artifacts, stage reports, operation history, and checkpoints.
"""

from __future__ import annotations

import copy
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np

from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage, RenderLayerEntry
from renderer_v2.execution.exceptions import WorkspaceValidationError
from renderer_v2.execution.models import LayerBuffer, SceneInstance, WorkspaceSnapshot
from renderer_v2.execution.reports import StageExecutionReport

logger = logging.getLogger(__name__)


class WorkspaceState(str, Enum):
    """Workspace lifecycle state enum."""

    CREATED = "CREATED"
    INITIALIZED = "INITIALIZED"
    ACTIVE = "ACTIVE"
    CHECKPOINTED = "CHECKPOINTED"
    CLEANED = "CLEANED"


class RenderWorkspace:
    """
    Mutable job-scoped workspace for accumulating intermediate render state,
    rasters, masks, scene instances, depth maps, stage reports, and checkpoints.
    """

    def __init__(self, job_id: str, canvas_width_px: int = 1280, canvas_height_px: int = 720) -> None:
        self._job_id = job_id
        self._canvas_width_px = max(1, canvas_width_px)
        self._canvas_height_px = max(1, canvas_height_px)
        self._state = WorkspaceState.CREATED

        # Core Storage
        self.layers: Dict[str, LayerBuffer] = {}
        self.masks: Dict[str, Any] = {}
        self.scene_instances: Dict[str, SceneInstance] = {}
        self.depth_map: Optional[Any] = None
        self.intermediate_artifacts: Dict[str, Any] = {}
        self.stage_reports: List[StageExecutionReport] = []
        self.op_history: List[Dict[str, Any]] = []
        self.checkpoints: Dict[str, WorkspaceSnapshot] = {}
        self._checkpoint_data: Dict[str, Dict[str, Any]] = {}

    @property
    def job_id(self) -> str:
        """Return workspace job ID."""
        return self._job_id

    @property
    def canvas_width_px(self) -> int:
        """Return canvas width in pixels."""
        return self._canvas_width_px

    @property
    def canvas_height_px(self) -> int:
        """Return canvas height in pixels."""
        return self._canvas_height_px

    @property
    def state(self) -> WorkspaceState:
        """Return current workspace lifecycle state."""
        return self._state

    def initialize(self, package: Optional[RenderExecutionPackage] = None) -> None:
        """
        Initialize workspace with dimensions and empty layer buffers
        from a RenderExecutionPackage layer_stack.
        """
        if package:
            self._canvas_width_px = package.scene_graph.canvas_width_px
            self._canvas_height_px = package.scene_graph.canvas_height_px

            # Pre-register layers from package layer_stack
            for layer_entry in package.layer_stack:
                if layer_entry.layer_id not in self.layers:
                    self.layers[layer_entry.layer_id] = LayerBuffer.from_entry(
                        entry=layer_entry,
                        width=self._canvas_width_px,
                        height=self._canvas_height_px,
                    )

        self._state = WorkspaceState.INITIALIZED
        logger.debug(
            f"Workspace initialized for job {self._job_id} ({self._canvas_width_px}x{self._canvas_height_px})"
        )

    # ---------------------------------------------------------------------------
    # Layer Operations
    # ---------------------------------------------------------------------------

    def add_layer(self, layer_id: str, buffer: Union[LayerBuffer, Any], entry: Optional[RenderLayerEntry] = None) -> None:
        """Add or overwrite a layer buffer in workspace."""
        if isinstance(buffer, LayerBuffer):
            self.layers[layer_id] = buffer
        else:
            if entry:
                layer_buf = LayerBuffer.from_entry(
                    entry=entry,
                    width=self._canvas_width_px,
                    height=self._canvas_height_px,
                    buffer_data=buffer,
                )
            else:
                layer_buf = LayerBuffer(
                    layer_id=layer_id,
                    layer_name=layer_id,
                    width_px=self._canvas_width_px,
                    height_px=self._canvas_height_px,
                    buffer_data=buffer,
                )
            self.layers[layer_id] = layer_buf
        self._state = WorkspaceState.ACTIVE

    def get_layer(self, layer_id: str) -> Optional[LayerBuffer]:
        """Retrieve layer buffer by layer_id."""
        return self.layers.get(layer_id)

    def has_layer(self, layer_id: str) -> bool:
        """Check if layer exists in workspace."""
        return layer_id in self.layers

    # ---------------------------------------------------------------------------
    # Mask Operations
    # ---------------------------------------------------------------------------

    def add_mask(self, mask_id: str, mask_data: Any) -> None:
        """Store a mask raster/ndarray buffer."""
        self.masks[mask_id] = mask_data
        self._state = WorkspaceState.ACTIVE

    def get_mask(self, mask_id: str) -> Optional[Any]:
        """Retrieve mask buffer by mask_id."""
        return self.masks.get(mask_id)

    # ---------------------------------------------------------------------------
    # Scene Instance Operations
    # ---------------------------------------------------------------------------

    def add_scene_instance(self, instance_id: str, instance: SceneInstance) -> None:
        """Store a scene instance object."""
        self.scene_instances[instance_id] = instance
        self._state = WorkspaceState.ACTIVE

    def get_scene_instance(self, instance_id: str) -> Optional[SceneInstance]:
        """Retrieve scene instance by instance_id."""
        return self.scene_instances.get(instance_id)

    # ---------------------------------------------------------------------------
    # Depth Map Operations
    # ---------------------------------------------------------------------------

    def set_depth_map(self, depth_map_data: Any) -> None:
        """Set full-frame depth map placeholder."""
        self.depth_map = depth_map_data
        self._state = WorkspaceState.ACTIVE

    # ---------------------------------------------------------------------------
    # Intermediate Artifacts & Reports
    # ---------------------------------------------------------------------------

    def add_artifact(self, name: str, artifact_data: Any) -> None:
        """Store a named debug or intermediate artifact blob."""
        self.intermediate_artifacts[name] = artifact_data

    def record_stage_report(self, report: StageExecutionReport) -> None:
        """Append a StageExecutionReport to workspace history."""
        self.stage_reports.append(report)

    def record_operation(self, op_id: str, op_type: str, status: str, latency_s: float, details: Optional[Dict[str, Any]] = None) -> None:
        """Record operation execution detail in operation history."""
        self.op_history.append({
            "op_id": op_id,
            "op_type": op_type,
            "status": status,
            "latency_s": latency_s,
            "details": details or {},
        })

    # ---------------------------------------------------------------------------
    # Checkpointing & Lifecycle
    # ---------------------------------------------------------------------------

    def checkpoint(self, name: str) -> WorkspaceSnapshot:
        """Snapshot current workspace state for point-in-time recovery."""
        snapshot = WorkspaceSnapshot(
            checkpoint_id=f"cp_{name}_{len(self.checkpoints)}",
            checkpoint_name=name,
            layer_ids=list(self.layers.keys()),
            mask_ids=list(self.masks.keys()),
            instance_ids=list(self.scene_instances.keys()),
            has_depth_map=self.depth_map is not None,
            stage_reports_count=len(self.stage_reports),
            op_history_count=len(self.op_history),
        )
        self.checkpoints[name] = snapshot

        # Deepcopy light structures for restoration
        self._checkpoint_data[name] = {
            "layers": copy.deepcopy(self.layers),
            "masks": copy.deepcopy(self.masks),
            "scene_instances": copy.deepcopy(self.scene_instances),
            "depth_map": copy.deepcopy(self.depth_map),
            "intermediate_artifacts": copy.deepcopy(self.intermediate_artifacts),
        }
        self._state = WorkspaceState.CHECKPOINTED
        return snapshot

    def restore_checkpoint(self, name: str) -> bool:
        """Restore workspace state from a named snapshot."""
        if name not in self._checkpoint_data:
            logger.warning(f"Checkpoint '{name}' not found in workspace for job {self._job_id}")
            return False

        saved = self._checkpoint_data[name]
        self.layers = copy.deepcopy(saved["layers"])
        self.masks = copy.deepcopy(saved["masks"])
        self.scene_instances = copy.deepcopy(saved["scene_instances"])
        self.depth_map = copy.deepcopy(saved["depth_map"])
        self.intermediate_artifacts = copy.deepcopy(saved["intermediate_artifacts"])
        self._state = WorkspaceState.ACTIVE
        logger.info(f"Workspace restored to checkpoint '{name}' for job {self._job_id}")
        return True

    def validate_workspace(self) -> List[str]:
        """Perform workspace structural validation."""
        errors: List[str] = []
        if self._canvas_width_px <= 0 or self._canvas_height_px <= 0:
            errors.append(f"Invalid canvas dimensions: {self._canvas_width_px}x{self._canvas_height_px}")

        for layer_id, buf in self.layers.items():
            if buf.width_px != self._canvas_width_px or buf.height_px != self._canvas_height_px:
                errors.append(
                    f"Layer '{layer_id}' dimensions ({buf.width_px}x{buf.height_px}) mismatch canvas ({self._canvas_width_px}x{self._canvas_height_px})"
                )
        return errors

    def cleanup(self) -> None:
        """Clean up buffers and transition state to CLEANED."""
        self.layers.clear()
        self.masks.clear()
        self.scene_instances.clear()
        self.depth_map = None
        self.intermediate_artifacts.clear()
        self._checkpoint_data.clear()
        self._state = WorkspaceState.CLEANED
        logger.debug(f"Workspace cleaned for job {self._job_id}")

    def reset(self) -> None:
        """Reset workspace storage for job retry."""
        self.cleanup()
        self._state = WorkspaceState.CREATED

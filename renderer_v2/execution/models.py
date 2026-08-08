"""
models.py
=========

Internal execution models for LayerBuffer, SceneInstance, and WorkspaceSnapshot.
Scoped entirely inside the execution layer (Phase 4.1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderLayerEntry, RenderPlacementCoordinate
from renderer_v2.execution.reports import StageExecutionReport


class LayerBuffer(BaseKBModel):
    """
    In-memory representation of a layer buffer within the workspace.
    Combines raster pixel buffer metadata with original RenderLayerEntry configuration.
    """

    layer_id: str = Field(description="Unique layer identifier")
    layer_name: str = Field(default="", description="Descriptive layer name")
    layer_type: str = Field(default="background", description="Layer type classification")
    z_index: int = Field(default=0, ge=0, description="Rendering order z-index")
    blend_mode: str = Field(default="normal", description="Blend mode e.g. normal, multiply, screen")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Layer opacity")
    visible: bool = Field(default=True, description="Visibility toggle")
    width_px: int = Field(default=1280, gt=0, description="Layer width in pixels")
    height_px: int = Field(default=720, gt=0, description="Layer height in pixels")
    channels: int = Field(default=4, ge=1, le=4, description="Color channels (3 for RGB, 4 for RGBA)")
    buffer_data: Optional[Any] = Field(default=None, description="Raster array buffer (ndarray or list)")

    model_config = {
        "arbitrary_types_allowed": True
    }

    @classmethod
    def from_entry(cls, entry: RenderLayerEntry, width: int = 1280, height: int = 720, buffer_data: Optional[Any] = None) -> LayerBuffer:
        """Construct LayerBuffer from RenderLayerEntry model."""
        return cls(
            layer_id=entry.layer_id,
            layer_name=entry.layer_name,
            layer_type=entry.layer_type,
            z_index=entry.z_index,
            blend_mode=entry.blend_mode,
            opacity=entry.opacity,
            visible=entry.visible,
            width_px=width,
            height_px=height,
            channels=4,
            buffer_data=buffer_data,
        )


class SceneInstance(BaseKBModel):
    """
    Decomposition scene instance record holding instance mask, alpha matte,
    bbox coordinates, depth layer, and locked status flag.
    """

    instance_id: str = Field(description="Unique scene instance identifier")
    class_label: str = Field(default="subject", description="Semantic class label e.g. person, logo, product")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")
    bbox: Tuple_BBox = Field(default=(0, 0, 100, 100), description="Bounding box (x, y, w, h)")
    is_locked: bool = Field(default=True, description="Locked instance flag — never regenerated")
    mask_buffer: Optional[Any] = Field(default=None, description="Binary mask array")
    alpha_matte: Optional[Any] = Field(default=None, description="Soft alpha matte array")
    depth_layer: Optional[Any] = Field(default=None, description="Instance depth layer array")

    model_config = {
        "arbitrary_types_allowed": True
    }


# Type alias for BBox tuple
Tuple_BBox = tuple[int, int, int, int]


class WorkspaceSnapshot(BaseKBModel):
    """
    Snapshot of workspace state captured at a specific execution checkpoint.
    Allows point-in-time recovery for retry loops.
    """

    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    checkpoint_name: str = Field(description="Name or label of checkpoint stage")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp")
    layer_ids: List[str] = Field(default_factory=list, description="Layer IDs present at snapshot time")
    mask_ids: List[str] = Field(default_factory=list, description="Mask IDs present at snapshot time")
    instance_ids: List[str] = Field(default_factory=list, description="Scene instance IDs present at snapshot time")
    has_depth_map: bool = Field(default=False, description="Flag indicating depth map presence")
    stage_reports_count: int = Field(default=0, ge=0, description="Number of stage reports logged")
    op_history_count: int = Field(default=0, ge=0, description="Number of operations logged")

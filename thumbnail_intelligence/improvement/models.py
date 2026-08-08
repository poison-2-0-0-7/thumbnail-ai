"""
models.py
=========

Data Models and Contracts for Phase 5.5 Automatic Improvement Engine.
Defines:
- ImprovementStrategyType (Enum: CONSERVATIVE, BALANCED, AGGRESSIVE)
- ModificationRule (Rule mapping suggestion action types to targeted transformations)
- LayerModification (Record of a specific layer/element modification)
- ImprovementExecutionPlan (Plan tracking modified vs preserved layers)
- ModificationReport (Report summarizing changes, preserved assets, expected gain, render cost)
- UpdatedRenderExecutionPackage (Container wrapping modified package, plan, and report)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage


class ImprovementStrategyType(str, Enum):
    """Strategies governing improvement aggressiveness and layer modification limits."""

    CONSERVATIVE = "conservative"  # Applies top 2 high-confidence suggestions only
    BALANCED = "balanced"        # Applies top 4 suggestions (default)
    AGGRESSIVE = "aggressive"      # Applies all valid suggestions


class LayerModification(BaseKBModel):
    """Record of a targeted modification applied to a specific layer or element."""

    modification_id: str = Field(..., description="Unique modification identifier")
    layer_id: str = Field(..., description="Target layer ID e.g. layer_elem_02_primary_subject")
    element_name: str = Field(..., description="Target element name e.g. Primary Hero Subject")
    modification_type: str = Field(..., description="Type of change e.g. face_scaling, typography_resizing, recoloring")
    action_type: str = Field(..., description="Originating suggestion action type")
    original_params: Dict[str, Any] = Field(default_factory=dict, description="Parameter values before modification")
    new_params: Dict[str, Any] = Field(default_factory=dict, description="Parameter values after modification")


class ImprovementExecutionPlan(BaseKBModel):
    """Execution plan detailing which layers are modified versus preserved."""

    plan_id: str = Field(..., description="Unique improvement execution plan identifier")
    base_package_id: str = Field(..., description="Source RenderExecutionPackage package_id")
    strategy_used: ImprovementStrategyType = Field(ImprovementStrategyType.BALANCED, description="Strategy applied")
    modified_layer_ids: List[str] = Field(default_factory=list, description="List of layer IDs modified")
    preserved_layer_ids: List[str] = Field(default_factory=list, description="List of layer IDs preserved without change")
    layer_modifications: List[LayerModification] = Field(default_factory=list, description="Detailed layer modification records")


class ModificationReport(BaseKBModel):
    """Summary report detailing layer preservation, expected CTR lift, and estimated render cost."""

    report_id: str = Field(..., description="Unique modification report identifier")
    base_package_id: str = Field(..., description="Base RenderExecutionPackage package_id")
    updated_package_id: str = Field(..., description="New RenderExecutionPackage package_id")
    total_layers_count: int = Field(..., ge=0, description="Total layer count in canvas")
    modified_layers_count: int = Field(..., ge=0, description="Number of layers modified")
    preserved_layers_count: int = Field(..., ge=0, description="Number of layers preserved without re-render")
    preservation_ratio: float = Field(..., ge=0.0, le=1.0, description="Percentage of layers preserved")
    expected_ctr_gain_pts: float = Field(..., ge=0.0, description="Total expected score lift in points")
    estimated_render_cost: str = Field("LOW", description="Estimated render cost (LOW, MEDIUM, HIGH)")
    modified_layer_names: List[str] = Field(default_factory=list, description="Names of modified layers")
    preserved_layer_names: List[str] = Field(default_factory=list, description="Names of preserved layers")


class UpdatedRenderExecutionPackage(BaseKBModel):
    """Container wrapping the updated RenderExecutionPackage, ImprovementExecutionPlan, and ModificationReport."""

    package: RenderExecutionPackage = Field(..., description="Revised, validated RenderExecutionPackage")
    execution_plan: ImprovementExecutionPlan = Field(..., description="Detailed improvement execution plan")
    report: ModificationReport = Field(..., description="Summary modification report")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of package update")

    def to_json(self, indent: int = 2) -> str:
        """Serialize UpdatedRenderExecutionPackage to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> UpdatedRenderExecutionPackage:
        """Deserialize UpdatedRenderExecutionPackage from JSON string."""
        return cls.model_validate(json.loads(json_str))

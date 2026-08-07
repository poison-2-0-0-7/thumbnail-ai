"""
renderer_adapter_models.py
==========================

Strongly typed Pydantic data contracts for the Renderer Adapter (Phase 3.8).
Defines the RenderExecutionPackage, RenderSceneGraph, RenderOperation, RenderAssetReference,
RenderMaskInstruction, RenderTypographyInstruction, RenderBackgroundInstruction,
RenderLightingInstruction, and RenderLayerEntry models.

The RenderExecutionPackage is the ONLY data contract consumed downstream by Renderer V2 or future rendering backends.
Intelligence Engine internal objects (DesignBrief, ReasoningContext, ExecutionPlan, SpatialComposition) MUST NEVER
be passed directly to renderers.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import Field, field_validator

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso


class PixelBoundingBox(BaseKBModel):
    """Absolute 2D Bounding Box in target canvas pixel coordinates."""

    x_px: int = Field(ge=0, description="Top-left x position in pixels")
    y_px: int = Field(ge=0, description="Top-left y position in pixels")
    width_px: int = Field(gt=0, description="Width in pixels")
    height_px: int = Field(gt=0, description="Height in pixels")

    @property
    def x2_px(self) -> int:
        """Right edge pixel coordinate."""
        return self.x_px + self.width_px

    @property
    def y2_px(self) -> int:
        """Bottom edge pixel coordinate."""
        return self.y_px + self.height_px

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return tuple (x_px, y_px, width_px, height_px)."""
        return (self.x_px, self.y_px, self.width_px, self.height_px)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return bounding box in (x1, y1, x2, y2) format."""
        return (self.x_px, self.y_px, self.x2_px, self.y2_px)


class RenderPlacementCoordinate(BaseKBModel):
    """Detailed renderer placement coordinates combining normalized and pixel bounding boxes."""

    element_id: str = Field(description="Unique visual element identifier")
    element_name: str = Field(description="Descriptive element name")
    bbox_normalized: Tuple[float, float, float, float] = Field(description="Normalized (x, y, w, h) tuple")
    bbox_pixels: PixelBoundingBox = Field(description="Pixel bounding box in target resolution")
    anchor_x_px: float = Field(default=0.0, description="Anchor x coordinate in pixels")
    anchor_y_px: float = Field(default=0.0, description="Anchor y coordinate in pixels")
    rotation_deg: float = Field(default=0.0, description="Rotation angle in degrees")
    scale: float = Field(default=1.0, gt=0.0, description="Scale factor")
    z_index: int = Field(default=0, ge=0, description="Layer z-index order")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Layer opacity")


class RenderAssetReference(BaseKBModel):
    """Manifest reference for required input assets and state keys."""

    asset_id: str = Field(description="Unique asset identifier")
    asset_type: str = Field(description="Asset type classification e.g. image_hero, image_prop, font, logo")
    source_key: str = Field(description="Internal asset source key e.g. asset:primary_subject")
    file_path: Optional[str] = Field(default=None, description="Optional local file path or URI")
    is_required: bool = Field(default=True, description="Strict requirement flag")


class RenderMaskInstruction(BaseKBModel):
    """In-painting or subject isolation mask instructions."""

    mask_id: str = Field(description="Unique mask identifier")
    target_element_id: str = Field(description="Target element ID for mask application")
    mask_type: str = Field(description="Mask type e.g. subject_matte, background_depth_mask, face_protect_mask")
    feather_px: float = Field(default=2.0, ge=0.0, description="Edge feathering in pixels")
    invert: bool = Field(default=False, description="Invert mask flag")


class RenderTypographyInstruction(BaseKBModel):
    """Renderer typography overlay rendering instructions."""

    text_id: str = Field(description="Unique text object identifier")
    content: str = Field(default="", description="Text string content to render")
    placement: RenderPlacementCoordinate = Field(description="Text placement coordinates")
    font_family: str = Field(default="Sans-Serif", description="Target font family")
    font_size_px: int = Field(default=48, gt=0, description="Font size in pixels")
    font_weight: str = Field(default="bold", description="Font weight e.g. bold, heavy")
    font_color_hex: str = Field(default="#FFFFFF", description="Text color hex code")
    stroke_color_hex: Optional[str] = Field(default="#000000", description="Text outline stroke hex code")
    stroke_width_px: int = Field(default=4, ge=0, description="Stroke width in pixels")
    drop_shadow_blur_px: int = Field(default=8, ge=0, description="Drop shadow blur radius in pixels")
    alignment: str = Field(default="left", description="Text alignment e.g. left, center, right")
    max_word_count: int = Field(default=4, ge=0, description="Maximum word count limit")


class RenderBackgroundInstruction(BaseKBModel):
    """Renderer background synthesis/replacement instructions."""

    action: str = Field(default="replace", description="Action e.g. replace, retain, inpaint")
    style_prompt_direction: str = Field(default="modern neon studio", description="Target background style direction")
    dominant_colors: List[str] = Field(default_factory=list, description="Target color palette hex codes")
    depth_treatment: str = Field(default="shallow", description="Depth of field style hint")
    sourced_from_step: str = Field(default="step_04_background_generation", description="Originating execution step")


class RenderLightingInstruction(BaseKBModel):
    """Renderer lighting and relighting instructions."""

    target_element_id: str = Field(description="Target element ID to receive relighting")
    mood: str = Field(default="high_key_dramatic", description="Lighting mood direction")
    key_light_direction: str = Field(default="top_left", description="Key light direction hint")
    key_light_intensity: float = Field(default=0.8, ge=0.0, le=1.0, description="Key light intensity multiplier")
    rim_light_enabled: bool = Field(default=True, description="Rim light flag")
    rim_light_color_temp: int = Field(default=5600, ge=1000, le=10000, description="Rim light Kelvin temperature")
    shadow_cast_enabled: bool = Field(default=True, description="Cast shadow flag")


class RenderLayerEntry(BaseKBModel):
    """Layer entry in the renderer layer stack."""

    layer_id: str = Field(description="Unique layer ID")
    layer_name: str = Field(description="Descriptive layer name")
    layer_type: str = Field(description="Layer type e.g. background, subject, lighting, shadow, typography, overlay")
    z_index: int = Field(default=0, ge=0, description="Rendering order z-index")
    blend_mode: str = Field(default="normal", description="Layer blend mode e.g. normal, multiply, screen")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Layer opacity")
    visible: bool = Field(default=True, description="Visibility toggle")


class RenderOperationType(str, Enum):
    """Classified renderer operation primitives."""

    LOAD_ASSET = "load_asset"
    PREPARE_CANVAS = "prepare_canvas"
    GENERATE_BACKGROUND = "generate_background"
    EXTRACT_SUBJECT = "extract_subject"
    ENHANCE_SUBJECT = "enhance_subject"
    APPLY_LIGHTING = "apply_lighting"
    GENERATE_SHADOW = "generate_shadow"
    COMPOSE_LAYER = "compose_layer"
    RENDER_TYPOGRAPHY = "render_typography"
    APPLY_COLOR_GRADE = "apply_color_grade"
    ADJUST_CONTRAST = "adjust_contrast"
    EVALUATE_QUALITY = "evaluate_quality"
    COMPOSITE_FINAL = "composite_final"
    CLEANUP_BUFFERS = "cleanup_buffers"


class RenderOperation(BaseKBModel):
    """Renderer primitive execution operation."""

    op_id: str = Field(description="Unique operation ID")
    op_type: RenderOperationType = Field(description="Operation primitive type")
    target_layer_id: str = Field(default="", description="Target layer ID")
    input_keys: List[str] = Field(default_factory=list, description="Input keys")
    output_keys: List[str] = Field(default_factory=list, description="Output keys")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Renderer operation parameters")
    sourced_from_step_id: str = Field(default="", description="Source ExecutionStep ID")


class RenderSceneGraphNode(BaseKBModel):
    """Renderer scene graph node."""

    node_id: str = Field(description="Unique scene node ID")
    node_name: str = Field(description="Node name")
    placement: RenderPlacementCoordinate = Field(description="Pixel placement coordinates")
    children: List[str] = Field(default_factory=list, description="Child node IDs")


class RenderSceneGraph(BaseKBModel):
    """Renderer scene graph containing resolution and hierarchical nodes."""

    scene_id: str = Field(default_factory=lambda: f"scene_{uuid.uuid4().hex[:8]}", description="Scene graph ID")
    canvas_width_px: int = Field(default=1280, gt=0, description="Canvas width in pixels")
    canvas_height_px: int = Field(default=720, gt=0, description="Canvas height in pixels")
    nodes: Dict[str, RenderSceneGraphNode] = Field(default_factory=dict, description="Scene graph nodes")


class RenderPackageMetadata(BaseKBModel):
    """Metadata tracking render package identity, references, target renderer, and semver."""

    package_id: str = Field(
        default_factory=lambda: f"pkg_render_{uuid.uuid4().hex[:8]}",
        description="Unique render package ID",
    )
    comp_ref: str = Field(description="Reference to SpatialComposition composition_id")
    plan_ref: str = Field(description="Reference to ExecutionPlan plan_id")
    brief_ref: str = Field(description="Reference to DesignBrief brief_id")
    target_renderer: str = Field(default="RendererV2", description="Target renderer engine name")
    schema_version: str = Field(default="1.0.0", description="Package semver version")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp")


class RenderExecutionPackage(BaseKBModel):
    """
    Master RenderExecutionPackage data contract.
    The ONLY contract emitted by the RendererAdapter to Renderer V2 or future rendering backends.
    """

    metadata: RenderPackageMetadata = Field(description="Render package metadata")
    scene_graph: RenderSceneGraph = Field(description="Target resolution scene graph")
    render_operations: List[RenderOperation] = Field(default_factory=list, description="Ordered render operations")
    asset_references: List[RenderAssetReference] = Field(default_factory=list, description="Required asset references")
    masks: List[RenderMaskInstruction] = Field(default_factory=list, description="Mask instructions")
    placement_coordinates: List[RenderPlacementCoordinate] = Field(default_factory=list, description="Pixel placements")
    typography_instructions: List[RenderTypographyInstruction] = Field(default_factory=list, description="Typography specs")
    background_instruction: RenderBackgroundInstruction = Field(
        default_factory=RenderBackgroundInstruction, description="Background specs"
    )
    lighting_instructions: List[RenderLightingInstruction] = Field(
        default_factory=list, description="Lighting/relighting specs"
    )
    layer_stack: List[RenderLayerEntry] = Field(default_factory=list, description="Ordered layer stack")

    # ---------------------------------------------------------------------------
    # Serialization Methods
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert RenderExecutionPackage to python dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RenderExecutionPackage:
        """Construct RenderExecutionPackage from dictionary representation."""
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize RenderExecutionPackage to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> RenderExecutionPackage:
        """Deserialize RenderExecutionPackage from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """Serialize RenderExecutionPackage to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> RenderExecutionPackage:
        """Deserialize RenderExecutionPackage from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def validate_package(self) -> List[str]:
        """Validate metadata, coordinate bounds, layer stack integrity, and asset references."""
        errors: List[str] = []
        if not self.metadata.package_id:
            errors.append("RenderPackageMetadata package_id must be non-empty.")
        if not self.metadata.comp_ref:
            errors.append("RenderPackageMetadata comp_ref must reference source SpatialComposition.")

        w_px = self.scene_graph.canvas_width_px
        h_px = self.scene_graph.canvas_height_px

        # Validate placement coordinates bounds
        for p in self.placement_coordinates:
            b = p.bbox_pixels
            if b.x_px < 0 or b.y_px < 0 or b.x2_px > w_px or b.y2_px > h_px:
                errors.append(f"Placement for '{p.element_id}' exceeds canvas bounds: ({b.x_px}, {b.y_px}, {b.x2_px}, {b.y2_px}).")

        # Validate asset references non-empty
        for ref in self.asset_references:
            if not ref.asset_id:
                errors.append(f"Asset reference with source_key '{ref.source_key}' has empty asset_id.")

        return errors

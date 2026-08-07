"""
spatial_composition_models.py
=============================

Strongly typed Pydantic data contracts for the Spatial Composition Planner (Phase 3.7).
Defines the renderer-independent SpatialComposition, CompositionGraph, BoundingBox,
CanvasSpecification, TypographyLayout, and PlacementInstructions models.

Decides WHERE visual elements belong in 2D/3D layout space (bounding box, rotation, scale,
z-index, layer depth, safe zones, alignment) without specifying HOW they are rendered.
Contains ZERO renderer-specific parameters (no ComfyUI, SD, SAM, YOLO, or Diffusers code).
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import Field, field_validator

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso


class CompositionRule(str, Enum):
    """Extensible taxonomy of professional visual composition rules."""

    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    CENTER_COMPOSITION = "center_composition"
    DIAGONAL_FLOW = "diagonal_flow"
    TRIANGULAR_COMPOSITION = "triangular_composition"
    RADIAL_BALANCE = "radial_balance"
    DYNAMIC_BALANCE = "dynamic_balance"
    CUSTOM = "custom"


class CompositionLayerPlane(str, Enum):
    """Z-plane classification for spatial depth layers."""

    BACKGROUND = "background"
    MIDGROUND = "midground"
    FOREGROUND = "foreground"
    OVERLAY = "overlay"


class CompositionRelationshipType(str, Enum):
    """Edge types for spatial relationship graph."""

    CONTAINMENT = "containment"
    OVERLAP = "overlap"
    ADJACENCY = "adjacency"
    ALIGNMENT = "alignment"
    DEPTH_ORDER = "depth_order"


class AnchorPoint(BaseKBModel):
    """Normalized origin anchor point within bounding box [0.0, 1.0]."""

    x_pct: float = Field(default=0.5, ge=0.0, le=1.0, description="Anchor x position percentage")
    y_pct: float = Field(default=0.5, ge=0.0, le=1.0, description="Anchor y position percentage")
    preset: str = Field(default="center", description="Anchor preset name e.g. center, top_left")


class BoundingBox(BaseKBModel):
    """Normalized 2D Bounding Box in canvas coordinates [0.0, 1.0]."""

    x: float = Field(ge=0.0, le=1.0, description="Normalized top-left x position")
    y: float = Field(ge=0.0, le=1.0, description="Normalized top-left y position")
    width: float = Field(gt=0.0, le=1.0, description="Normalized width fraction")
    height: float = Field(gt=0.0, le=1.0, description="Normalized height fraction")

    @property
    def aspect_ratio(self) -> float:
        """Calculate width-to-height aspect ratio."""
        return self.width / max(self.height, 1e-6)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center coordinate (cx, cy)."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def x2(self) -> float:
        """Right edge coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge coordinate."""
        return self.y + self.height

    def overlaps(self, other: BoundingBox, margin: float = 0.0) -> bool:
        """Check if this box overlaps another box (with optional margin)."""
        no_overlap_x = (self.x2 + margin <= other.x) or (other.x2 + margin <= self.x)
        no_overlap_y = (self.y2 + margin <= other.y) or (other.y2 + margin <= self.y)
        return not (no_overlap_x or no_overlap_y)

    def intersection_area(self, other: BoundingBox) -> float:
        """Calculate intersection area between two boxes in normalized units."""
        inter_x1 = max(self.x, other.x)
        inter_y1 = max(self.y, other.y)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) ratio [0.0, 1.0]."""
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        area_a = self.width * self.height
        area_b = other.width * other.height
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)


class VisualElementPlacement(BaseKBModel):
    """Full spatial placement attributes for a visual element in 2D/3D canvas space."""

    element_id: str = Field(description="Unique element ID e.g. elem_01_primary_subject")
    element_name: str = Field(description="Descriptive element name e.g. Host Face")
    element_category: str = Field(default="subject", description="Category e.g. face, subject, text, logo, prop")
    bbox: BoundingBox = Field(description="Normalized 2D bounding box")
    anchor_point: AnchorPoint = Field(default_factory=AnchorPoint, description="Anchor point origin")
    rotation_deg: float = Field(default=0.0, ge=-360.0, le=360.0, description="Rotation angle in degrees")
    scale: float = Field(default=1.0, gt=0.0, description="Relative scale factor")
    z_index: int = Field(default=0, ge=0, description="Rendering layer z-index order")
    layer_plane: CompositionLayerPlane = Field(
        default=CompositionLayerPlane.MIDGROUND, description="Layer plane classification"
    )
    depth_z: float = Field(default=0.5, ge=0.0, le=1.0, description="Normalized 3D depth coordinate")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Layer opacity [0.0, 1.0]")
    alignment: str = Field(default="center", description="Alignment e.g. left, center, right")
    safe_zone_restricted: bool = Field(default=True, description="Strictly restricted from overlapping UI safe zones")
    priority_tier: str = Field(default="PRIMARY", description="Visual priority tier: PRIMARY, SECONDARY, TERTIARY")
    sourced_from_step: str = Field(default="", description="ExecutionStep ID that placed this element")


class SafeZone(BaseKBModel):
    """Represent UI or platform overlay safe zones on canvas."""

    zone_id: str = Field(description="Unique safe zone identifier e.g. timestamp_overlay")
    bbox: BoundingBox = Field(description="Normalized safe zone bounding box")
    description: str = Field(default="", description="Description of UI overlay constraint")
    is_forbidden: bool = Field(default=True, description="If True, visual elements must NOT overlap")


class CanvasSpecification(BaseKBModel):
    """Canvas dimension, aspect ratio, safe zones, and margin guidelines."""

    width_px: int = Field(default=1280, gt=0, description="Canvas pixel width")
    height_px: int = Field(default=720, gt=0, description="Canvas pixel height")
    aspect_ratio: str = Field(default="16:9", description="Canvas aspect ratio string")
    mobile_crop_safe_zone: SafeZone = Field(
        default_factory=lambda: SafeZone(
            zone_id="mobile_crop",
            bbox=BoundingBox(x=0.125, y=0.0, width=0.75, height=1.0),
            description="Central 4:3 crop area safe for mobile feed cards",
            is_forbidden=False,
        )
    )
    timestamp_safe_zone: SafeZone = Field(
        default_factory=lambda: SafeZone(
            zone_id="timestamp_overlay",
            bbox=BoundingBox(x=0.80, y=0.85, width=0.18, height=0.12),
            description="Bottom-right YouTube duration badge area",
            is_forbidden=True,
        )
    )
    title_safe_zone: SafeZone = Field(
        default_factory=lambda: SafeZone(
            zone_id="title_overlay",
            bbox=BoundingBox(x=0.02, y=0.02, width=0.96, height=0.96),
            description="General canvas margin safety boundary",
            is_forbidden=False,
        )
    )
    margin_rules: Dict[str, float] = Field(
        default_factory=lambda: {"top_margin": 0.05, "bottom_margin": 0.05, "left_margin": 0.05, "right_margin": 0.05}
    )


class TypographyLayout(BaseKBModel):
    """Detailed spatial layout, collision avoidance, and contrast targets for text overlays."""

    text_element_id: str = Field(description="Reference to text VisualElementPlacement element_id")
    text_content: str = Field(default="", description="Planned text headline content")
    text_region_bbox: BoundingBox = Field(description="Bounding box allocated for text region")
    maximum_width_fraction: float = Field(default=0.80, ge=0.1, le=1.0, description="Max width fraction of canvas")
    maximum_height_fraction: float = Field(default=0.40, ge=0.1, le=1.0, description="Max height fraction of canvas")
    alignment: str = Field(default="left", description="Text alignment e.g. left, center, right")
    padding_px: float = Field(default=10.0, ge=0.0, description="Internal text region padding in pixels")
    face_avoidance_margin: float = Field(default=0.05, ge=0.0, description="Safety margin from host face region")
    collision_free: bool = Field(default=True, description="True if verified 0% collision with host faces and safe zones")
    contrast_target_background: str = Field(default="high_contrast", description="Target background contrast treatment")


class CompositionEdge(BaseKBModel):
    """Spatial relationship edge between two visual element nodes."""

    source_element_id: str = Field(description="Origin element ID")
    target_element_id: str = Field(description="Destination element ID")
    relationship_type: CompositionRelationshipType = Field(description="Classified spatial relationship")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship priority/weight")
    description: str = Field(default="", description="Relationship summary")


class CompositionGraph(BaseKBModel):
    """
    Spatial graph representation of the composition.
    Nodes represent VisualElementPlacements, edges represent spatial relationships.
    Provides collision detection and layout validation algorithms.
    """

    graph_id: str = Field(
        default_factory=lambda: f"spatial_graph_{uuid.uuid4().hex[:8]}",
        description="Unique spatial composition graph ID",
    )
    nodes: Dict[str, VisualElementPlacement] = Field(default_factory=dict, description="Nodes: element_id -> VisualElementPlacement")
    edges: List[CompositionEdge] = Field(default_factory=list, description="Edges: spatial relationships")

    def add_element(self, element: VisualElementPlacement) -> None:
        """Add or replace a visual element node in the spatial graph."""
        self.nodes[element.element_id] = element

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: CompositionRelationshipType,
        description: str = "",
    ) -> None:
        """Add a spatial relationship edge between two nodes."""
        edge = CompositionEdge(
            source_element_id=source_id,
            target_element_id=target_id,
            relationship_type=rel_type,
            description=description,
        )
        self.edges.append(edge)

    def detect_collisions(self, margin: float = 0.0) -> List[Dict[str, Any]]:
        """
        Detect overlapping pairs of visual elements.
        Returns a list of collision descriptors.
        """
        collisions: List[Dict[str, Any]] = []
        node_list = list(self.nodes.values())

        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                a, b = node_list[i], node_list[j]
                # Background layer elements don't collide
                if a.layer_plane == CompositionLayerPlane.BACKGROUND or b.layer_plane == CompositionLayerPlane.BACKGROUND:
                    continue

                if a.bbox.overlaps(b.bbox, margin=margin):
                    iou = a.bbox.intersection_over_union(b.bbox)
                    area = a.bbox.intersection_area(b.bbox)
                    collisions.append(
                        {
                            "element_a": a.element_id,
                            "element_b": b.element_id,
                            "category_a": a.element_category,
                            "category_b": b.element_category,
                            "intersection_area": area,
                            "iou": iou,
                        }
                    )

        return collisions

    def validate_composition_graph(self, canvas: CanvasSpecification) -> List[str]:
        """
        Validate spatial composition graph:
        - Check canvas overflow
        - Check forbidden safe zone violations (e.g. timestamp overlay)
        - Check text/face collisions
        """
        errors: List[str] = []

        # Check canvas overflow
        for elem in self.nodes.values():
            b = elem.bbox
            if b.x < 0.0 or b.y < 0.0 or b.x2 > 1.0 or b.y2 > 1.0:
                errors.append(f"Element '{elem.element_id}' exceeds canvas boundaries: ({b.x:.2f}, {b.y:.2f}, {b.x2:.2f}, {b.y2:.2f}).")

        # Check forbidden safe zone overlaps
        if canvas.timestamp_safe_zone and canvas.timestamp_safe_zone.is_forbidden:
            ts_box = canvas.timestamp_safe_zone.bbox
            for elem in self.nodes.values():
                if elem.safe_zone_restricted and elem.layer_plane != CompositionLayerPlane.BACKGROUND:
                    if elem.bbox.overlaps(ts_box):
                        errors.append(f"Element '{elem.element_id}' overlaps forbidden timestamp safe zone.")

        # Check face vs text collisions
        faces = [e for e in self.nodes.values() if e.element_category == "face"]
        texts = [e for e in self.nodes.values() if e.element_category == "text"]

        for f in faces:
            for t in texts:
                if f.bbox.overlaps(t.bbox, margin=0.02):
                    errors.append(f"Collision detected between face '{f.element_id}' and text '{t.element_id}'.")

        return errors


class PlacementInstructions(BaseKBModel):
    """High-level graphic design instructions, focal points, eye path, and balance metrics."""

    primary_focal_point: str = Field(default="", description="element_id of primary hero focal point")
    secondary_focal_point: str = Field(default="", description="element_id of secondary focal point")
    applied_composition_rule: CompositionRule = Field(
        default=CompositionRule.RULE_OF_THIRDS, description="Dominant graphic design composition rule applied"
    )
    visual_flow_path: List[str] = Field(
        default_factory=list, description="Ordered element_ids defining expected viewer eye scan path"
    )
    negative_space_fraction: float = Field(default=0.30, ge=0.0, le=1.0, description="Uncluttered canvas fraction")
    visual_balance_score: float = Field(default=0.85, ge=0.0, le=1.0, description="Left/right moment balance score")
    is_asymmetrical: bool = Field(default=True, description="True if composition uses intentional dynamic asymmetry")


class SpatialComposition(BaseKBModel):
    """
    Master SpatialComposition data contract.
    Contains CanvasSpecification, CompositionGraph, PlacementInstructions, and TypographyLayout.
    Emitted by SpatialCompositionPlanner (Phase 3.7).
    """

    composition_id: str = Field(
        default_factory=lambda: f"comp_{uuid.uuid4().hex[:8]}",
        description="Unique spatial composition ID",
    )
    plan_ref: str = Field(description="Reference to source ExecutionPlan plan_id")
    brief_ref: str = Field(description="Reference to source DesignBrief brief_id")
    schema_version: str = Field(default="1.0.0", description="Schema semver version")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC creation timestamp")
    canvas: CanvasSpecification = Field(default_factory=CanvasSpecification, description="Canvas & safe zone specs")
    composition_graph: CompositionGraph = Field(description="Spatial graph with nodes and relationships")
    placement_instructions: PlacementInstructions = Field(description="Composition rules and focal points")
    typography_layout: Optional[TypographyLayout] = Field(default=None, description="Detailed typography layout specs")

    # ---------------------------------------------------------------------------
    # Serialization Methods
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert SpatialComposition to python dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpatialComposition:
        """Construct SpatialComposition from dictionary representation."""
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize SpatialComposition to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> SpatialComposition:
        """Deserialize SpatialComposition from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """Serialize SpatialComposition to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SpatialComposition:
        """Deserialize SpatialComposition from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def validate_spatial_composition(self) -> List[str]:
        """Validate metadata, canvas bounds, safe zones, and graph integrity."""
        errors: List[str] = []
        if not self.composition_id:
            errors.append("SpatialComposition composition_id must be non-empty.")
        if not self.plan_ref:
            errors.append("SpatialComposition plan_ref must specify source ExecutionPlan ID.")

        graph_errors = self.composition_graph.validate_composition_graph(self.canvas)
        errors.extend(graph_errors)
        return errors

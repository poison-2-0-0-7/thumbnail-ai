"""
relationship_reasoner.py
========================

Spatial & Semantic Relationship Reasoner (Phase 6 & Phase 7).

Computes pairwise spatial and semantic relationships between scene elements using
deterministic spatial geometry (above, below, left, right, overlapping, occluding,
inside, behind, in front of) and type-informed relational heuristics.
"""

from __future__ import annotations

from models import BoundingBox
from thumbnail_understanding.schemas import (
    ElementRelationship,
    ElementType,
    SceneElement,
    SpatialRelation,
)


class RelationshipReasoner:
    """Computes pairwise spatial and semantic relationships between SceneElements."""

    @staticmethod
    def _compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x_left = max(b1.x_min, b2.x_min)
        y_top = max(b1.y_min, b2.y_min)
        x_right = min(b1.x_max, b2.x_max)
        y_bottom = min(b1.y_max, b2.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        b1_area = (b1.x_max - b1.x_min) * (b1.y_max - b1.y_min)
        b2_area = (b2.x_max - b2.x_min) * (b2.y_max - b2.y_min)
        union_area = b1_area + b2_area - intersection_area

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def _compute_containment(inner: BoundingBox, outer: BoundingBox) -> float:
        """Fraction of `inner` box area contained within `outer` box."""
        x_left = max(inner.x_min, outer.x_min)
        y_top = max(inner.y_min, outer.y_min)
        x_right = min(inner.x_max, outer.x_max)
        y_bottom = min(inner.y_max, outer.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        inner_area = (inner.x_max - inner.x_min) * (inner.y_max - inner.y_min)
        if inner_area <= 0:
            return 0.0
        return intersection_area / inner_area

    @classmethod
    def analyze_relationships(
        cls,
        elements: list[SceneElement],
    ) -> list[ElementRelationship]:
        """
        Compute all valid pairwise relationships among the provided scene elements.
        """
        relationships: list[ElementRelationship] = []
        non_bg = [e for e in elements if e.element_type != ElementType.BACKGROUND]
        bg = [e for e in elements if e.element_type == ElementType.BACKGROUND]

        # 1. Background relationships
        for e in non_bg:
            if bg:
                relationships.append(
                    ElementRelationship(
                        subject_element_id=e.element_id,
                        relation=SpatialRelation.IN_FRONT_OF,
                        object_element_id=bg[0].element_id,
                        confidence=1.0,
                        spatial_direction="foreground",
                        provenance="deterministic_spatial_engine",
                    )
                )

        # 2. Pairwise relationships among non-background elements
        n = len(non_bg)
        for i in range(n):
            e1 = non_bg[i]
            for j in range(i + 1, n):
                e2 = non_bg[j]

                iou = cls._compute_iou(e1.bbox, e2.bbox)
                c1_in_2 = cls._compute_containment(e1.bbox, e2.bbox)
                c2_in_1 = cls._compute_containment(e2.bbox, e1.bbox)

                # Overlap / Occlusion / Containment
                if iou > 0.05 or c1_in_2 > 0.2 or c2_in_1 > 0.2:
                    if c1_in_2 > 0.8:
                        relationships.append(
                            ElementRelationship(
                                subject_element_id=e1.element_id,
                                relation=SpatialRelation.INSIDE,
                                object_element_id=e2.element_id,
                                confidence=round(c1_in_2, 2),
                                spatial_direction="contained",
                                provenance="deterministic_spatial_engine",
                            )
                        )
                    elif c2_in_1 > 0.8:
                        relationships.append(
                            ElementRelationship(
                                subject_element_id=e2.element_id,
                                relation=SpatialRelation.INSIDE,
                                object_element_id=e1.element_id,
                                confidence=round(c2_in_1, 2),
                                spatial_direction="contained",
                                provenance="deterministic_spatial_engine",
                            )
                        )

                    # Text relative to subject/person
                    if e1.element_type == ElementType.TEXT and e2.element_type in (ElementType.PERSON, ElementType.OBJECT):
                        relationships.append(
                            ElementRelationship(
                                subject_element_id=e1.element_id,
                                relation=SpatialRelation.TEXT_RELATIVE_TO,
                                object_element_id=e2.element_id,
                                confidence=0.9,
                                spatial_direction="overlapping",
                                provenance="text_spatial_engine",
                            )
                        )
                    elif e2.element_type == ElementType.TEXT and e1.element_type in (ElementType.PERSON, ElementType.OBJECT):
                        relationships.append(
                            ElementRelationship(
                                subject_element_id=e2.element_id,
                                relation=SpatialRelation.TEXT_RELATIVE_TO,
                                object_element_id=e1.element_id,
                                confidence=0.9,
                                spatial_direction="overlapping",
                                provenance="text_spatial_engine",
                            )
                        )

                    # Person holding prop
                    if e1.element_type == ElementType.PERSON and e2.element_type in (ElementType.PROP, ElementType.OBJECT):
                        relationships.append(
                            ElementRelationship(
                                subject_element_id=e1.element_id,
                                relation=SpatialRelation.HOLDING,
                                object_element_id=e2.element_id,
                                confidence=0.85,
                                spatial_direction="interacting",
                                provenance="person_prop_engine",
                            )
                        )
                    elif e2.element_type == ElementType.PERSON and e1.element_type in (ElementType.PROP, ElementType.OBJECT):
                        relationships.append(
                            ElementRelationship(
                                subject_element_id=e2.element_id,
                                relation=SpatialRelation.HOLDING,
                                object_element_id=e1.element_id,
                                confidence=0.85,
                                spatial_direction="interacting",
                                provenance="person_prop_engine",
                            )
                        )

                # Positional relative spatial direction (Left/Right, Above/Below)
                e1_cx = (e1.bbox.x_min + e1.bbox.x_max) / 2.0
                e1_cy = (e1.bbox.y_min + e1.bbox.y_max) / 2.0
                e2_cx = (e2.bbox.x_min + e2.bbox.x_max) / 2.0
                e2_cy = (e2.bbox.y_min + e2.bbox.y_max) / 2.0

                if abs(e1_cx - e2_cx) > abs(e1_cy - e2_cy):
                    rel = SpatialRelation.LEFT_OF if e1_cx < e2_cx else SpatialRelation.RIGHT_OF
                    dir_str = "left_of" if e1_cx < e2_cx else "right_of"
                else:
                    rel = SpatialRelation.ABOVE if e1_cy < e2_cy else SpatialRelation.BELOW
                    dir_str = "above" if e1_cy < e2_cy else "below"

                relationships.append(
                    ElementRelationship(
                        subject_element_id=e1.element_id,
                        relation=rel,
                        object_element_id=e2.element_id,
                        confidence=0.95,
                        spatial_direction=dir_str,
                        provenance="deterministic_spatial_engine",
                    )
                )

        return relationships

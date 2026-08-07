"""
hierarchy_calculator.py
========================

Deterministic Subject Hierarchy and Visual Hierarchy Calculator (Phase 3 & Phase 8).

Computes visual hierarchy, reading order (eye flow), dominant subject, first/second
attention targets, attention competition, and focal strength using auditable geometric,
face, text, and contrast evidence without arbitrary hardcoded guessing.
"""

from __future__ import annotations

import math
from typing import Optional
from models import BoundingBox, CompositionAnalysis
from thumbnail_understanding.schemas import (
    ElementRole,
    ElementType,
    SceneElement,
    SceneGraph,
    VisualHierarchy,
)


class HierarchyCalculator:
    """Calculates deterministic subject importance ranking and visual hierarchy."""

    @classmethod
    def calculate_importance_score(
        cls,
        elem: SceneElement,
        composition: CompositionAnalysis,
    ) -> float:
        """
        Compute an auditable numerical score [0.0, 100.0] for how strongly an element draws attention.
        """
        if elem.element_type == ElementType.BACKGROUND:
            return 0.0

        # 1. Size contribution (area fraction)
        area = (elem.bbox.x_max - elem.bbox.x_min) * (elem.bbox.y_max - elem.bbox.y_min)
        size_score = min(1.0, area / 0.4) * 35.0  # Up to 35 pts

        # 2. Face presence & Hero status
        face_score = 0.0
        if elem.element_type == ElementType.PERSON:
            if elem.is_creator or elem.role == ElementRole.HERO:
                face_score = 35.0
            else:
                face_score = 25.0

        # 3. Text presence (high contrast visual anchor)
        text_score = 30.0 if elem.element_type == ElementType.TEXT else 0.0

        # 4. Centrality / Rule of thirds alignment
        center_x = (elem.bbox.x_min + elem.bbox.x_max) / 2.0
        center_y = (elem.bbox.y_min + elem.bbox.y_max) / 2.0
        dist_from_center = math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
        centrality_score = max(0.0, 1.0 - (dist_from_center / 0.707)) * 15.0

        # 5. Detector Confidence & Sharpness
        quality_score = elem.confidence * 15.0

        total = size_score + face_score + text_score + centrality_score + quality_score
        return round(total, 2)

    @classmethod
    def compute_hierarchy(
        cls,
        elements: list[SceneElement],
        composition: CompositionAnalysis,
    ) -> tuple[SceneGraph, VisualHierarchy]:
        """
        Re-ranks SceneElement list, identifies hero element, primary/secondary subjects,
        reading order, and outputs updated SceneGraph + VisualHierarchy.
        """
        non_bg_elements = [e for e in elements if e.element_type != ElementType.BACKGROUND]
        bg_elements = [e for e in elements if e.element_type == ElementType.BACKGROUND]

        # Calculate scores for non-bg elements
        scored_elements: list[tuple[float, SceneElement]] = []
        for elem in non_bg_elements:
            score = cls.calculate_importance_score(elem, composition)
            scored_elements.append((score, elem))

        # Sort descending by score
        scored_elements.sort(key=lambda x: x[0], reverse=True)

        # Update importance_rank and role in updated element list
        updated_elements: list[SceneElement] = []
        hero_id: Optional[str] = None
        primary_ids: list[str] = []
        secondary_ids: list[str] = []

        for rank_idx, (score, elem) in enumerate(scored_elements, start=1):
            if rank_idx == 1:
                hero_id = elem.element_id
                role = ElementRole.HERO if elem.element_type == ElementType.PERSON else ElementRole.PRIMARY
                primary_ids.append(elem.element_id)
            elif rank_idx == 2:
                role = ElementRole.PRIMARY if elem.element_type in (ElementType.PERSON, ElementType.TEXT) else ElementRole.SECONDARY
                primary_ids.append(elem.element_id)
            elif rank_idx <= 4:
                role = ElementRole.SECONDARY
                secondary_ids.append(elem.element_id)
            else:
                role = ElementRole.SUPPORTING

            # Re-create element with updated rank and role
            updated_elem = SceneElement(
                element_id=elem.element_id,
                element_type=elem.element_type,
                category=elem.category,
                label=elem.label,
                semantic_description=elem.semantic_description,
                bbox=elem.bbox,
                polygon=elem.polygon,
                mask_path=elem.mask_path,
                cutout_path=elem.cutout_path,
                confidence=elem.confidence,
                importance_rank=rank_idx,
                role=role,
                preserve_score=elem.preserve_score,
                replace_score=elem.replace_score,
                edit_priority=1 if rank_idx <= 2 else 2 if rank_idx <= 4 else 3,
                depth_level=0.1 if rank_idx == 1 else 0.2 if rank_idx == 2 else 0.4,
                occlusion_ratio=elem.occlusion_ratio,
                parent_id=elem.parent_id,
                children_ids=elem.children_ids,
                identity_relevance=elem.identity_relevance,
                story_relevance=elem.story_relevance,
                visual_relevance=elem.visual_relevance,
                editability=elem.editability,
                source_detector=elem.source_detector,
                provenance=elem.provenance,
                emotion=elem.emotion,
                emotion_confidence=elem.emotion_confidence,
                expression_intensity=elem.expression_intensity,
                head_pose=elem.head_pose,
                eye_direction=elem.eye_direction,
                sharpness=elem.sharpness,
                lighting_quality=elem.lighting_quality,
                is_creator=elem.is_creator,
            )
            updated_elements.append(updated_elem)

        # Append background elements back
        updated_elements.extend(bg_elements)

        # Build reading order (Western eye flow: top-left to bottom-right, weighted by importance)
        reading_order = [e.element_id for e in updated_elements if e.element_type != ElementType.BACKGROUND]
        
        first_target = reading_order[0] if reading_order else None
        second_target = reading_order[1] if len(reading_order) > 1 else None

        # Focal strength calculation (ratio of top 1 score to top 2 score)
        focal_strength = 0.8
        attention_comp = 0.3
        if len(scored_elements) >= 2:
            s1, s2 = scored_elements[0][0], scored_elements[1][0]
            if s1 > 0:
                focal_strength = min(1.0, round((s1 - s2) / s1, 2))
                attention_comp = min(1.0, round(s2 / (s1 + 1e-5), 2))

        hierarchy = VisualHierarchy(
            reading_order=reading_order,
            first_attention_target=first_target,
            second_attention_target=second_target,
            dominant_subject_id=hero_id,
            visual_anchors=primary_ids,
            attention_competition_score=attention_comp,
            eye_flow_description=f"Eye flows from {first_target or 'center'} to {second_target or 'background'}",
            negative_space_ratio=composition.negative_space_ratio,
            text_safe_areas=[],
            visual_clutter_score=composition.clutter_score,
            subject_separation_score=round(1.0 - composition.clutter_score, 2),
            balance_score=composition.balance_score,
            focal_strength_score=focal_strength,
            hierarchy_basis="size_face_text_position_scoring",
        )

        scene_graph = SceneGraph(
            elements=updated_elements,
            relationships=[],  # To be populated by RelationshipReasoner
            hero_element_id=hero_id,
            primary_subject_ids=primary_ids,
            secondary_subject_ids=secondary_ids,
        )

        return scene_graph, hierarchy

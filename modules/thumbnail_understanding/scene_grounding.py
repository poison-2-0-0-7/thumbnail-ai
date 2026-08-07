"""
scene_grounding.py
==================

Deterministic scene element grounding component (Phase 2 & Phase 5).

Grounds raw detections from ThumbnailIntelligence (faces, objects, OCR text,
colors, composition) into structured, strongly-typed SceneElement records with
deterministic element_ids, initial importance ranks, and semantic roles.
"""

from __future__ import annotations

import hashlib
from typing import Optional
from models import (
    BoundingBox,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    OCRResult,
    ThumbnailIntelligence,
)
from thumbnail_understanding.schemas import (
    EditabilityStatus,
    ElementRole,
    ElementType,
    SceneElement,
)


class SceneGrounder:
    """Grounds raw detections into unified SceneElement instances."""

    @staticmethod
    def _make_element_id(prefix: str, index: int, label: str, bbox: BoundingBox) -> str:
        """Generate a stable, deterministic element ID based on label and bbox."""
        raw_key = f"{prefix}_{index}_{label}_{bbox.x_min:.3f}_{bbox.y_min:.3f}_{bbox.x_max:.3f}_{bbox.y_max:.3f}"
        short_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:8]
        clean_label = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
        return f"elem_{prefix}_{clean_label}_{short_hash}"

    @classmethod
    def ground_elements(
        cls,
        intelligence: ThumbnailIntelligence,
    ) -> list[SceneElement]:
        """
        Produce a grounded, ranked list of SceneElement instances from ThumbnailIntelligence.
        """
        elements: list[SceneElement] = []

        # 1. Background element (always present)
        bg_bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        bg_element = SceneElement(
            element_id="elem_bg_plate_00",
            element_type=ElementType.BACKGROUND,
            category="background",
            label="Background Plate",
            semantic_description=f"Overall background environment ({intelligence.colors.warm_or_cool} temperature)",
            bbox=bg_bbox,
            confidence=1.0,
            importance_rank=999,
            role=ElementRole.BACKGROUND,
            preserve_score=0.1,
            replace_score=0.9,
            edit_priority=5,
            depth_level=1.0,
            occlusion_ratio=0.0,
            editability=EditabilityStatus.EDITABLE,
            source_detector="color_profile",
            provenance="deterministic_grounder",
        )
        elements.append(bg_element)

        # 2. Face Grounding (Phase 4)
        if intelligence.faces.has_face:
            for idx, face in enumerate(intelligence.faces.faces):
                area = (face.bbox.x_max - face.bbox.x_min) * (face.bbox.y_max - face.bbox.y_min)
                is_largest = face.is_largest or (idx == 0)
                role = ElementRole.HERO if is_largest else ElementRole.PRIMARY if idx == 1 else ElementRole.SECONDARY
                
                # High identity relevance for faces
                preserve_score = 0.95 if is_largest else 0.85
                replace_score = 0.05 if is_largest else 0.15

                elem_id = cls._make_element_id("face", idx, "person_face", face.bbox)
                elements.append(
                    SceneElement(
                        element_id=elem_id,
                        element_type=ElementType.PERSON,
                        category="person",
                        label=f"Person Face {idx + 1}" + (" (Hero)" if is_largest else ""),
                        semantic_description=f"Human face with emotion={face.emotion or 'neutral'}, position={face.position_label}",
                        bbox=face.bbox,
                        confidence=face.detection_confidence,
                        importance_rank=idx + 1,
                        role=role,
                        preserve_score=preserve_score,
                        replace_score=replace_score,
                        edit_priority=1 if is_largest else 2,
                        depth_level=0.2,  # Faces sit near front
                        occlusion_ratio=0.0,
                        identity_relevance=0.95 if is_largest else 0.8,
                        story_relevance=0.9,
                        visual_relevance=0.9,
                        editability=EditabilityStatus.PRESERVE if is_largest else EditabilityStatus.EDITABLE,
                        source_detector="insightface_multi",
                        provenance="face_analysis_stage",
                        emotion=face.emotion,
                        emotion_confidence=face.emotion_confidence,
                        head_pose=face.head_pose,
                        eye_direction=face.eye_direction,
                        is_creator=is_largest,
                    )
                )

        # 3. Object Grounding (Phase 5)
        # Filter out objects that heavily overlap faces to avoid double-counting face bbox as generic 'person'
        for idx, obj in enumerate(intelligence.objects):
            area = (obj.bbox.x_max - obj.bbox.x_min) * (obj.bbox.y_max - obj.bbox.y_min)
            
            # Check if this object is a 'person' that corresponds to an already grounded face
            is_face_person = False
            if obj.label.lower() == "person" and intelligence.faces.has_face:
                for face in intelligence.faces.faces:
                    # Check overlap / containment
                    if (
                        obj.bbox.x_min <= face.bbox.x_min <= obj.bbox.x_max
                        and obj.bbox.y_min <= face.bbox.y_min <= obj.bbox.y_max
                    ):
                        is_face_person = True
                        break

            elem_type = ElementType.PERSON if obj.label.lower() == "person" else ElementType.OBJECT
            if obj.label.lower() in {"microphone", "camera", "phone", "bag", "cup", "book", "trophy", "box", "car"}:
                elem_type = ElementType.PROP

            rank = len(elements) + 1
            preserve_score = 0.7 if elem_type == ElementType.PERSON else 0.4
            replace_score = 0.3 if elem_type == ElementType.PERSON else 0.6
            role = ElementRole.PRIMARY if (area > 0.15 and not is_face_person) else ElementRole.SUPPORTING

            elem_id = cls._make_element_id("obj", idx, obj.label, obj.bbox)
            elements.append(
                SceneElement(
                    element_id=elem_id,
                    element_type=elem_type,
                    category=obj.label.lower(),
                    label=f"{obj.label.title()} ({idx + 1})",
                    semantic_description=f"Detected {obj.label} with confidence {obj.confidence:.2f}",
                    bbox=obj.bbox,
                    confidence=obj.confidence,
                    importance_rank=rank,
                    role=role,
                    preserve_score=preserve_score,
                    replace_score=replace_score,
                    edit_priority=2 if elem_type == ElementType.PERSON else 3,
                    depth_level=0.4,
                    occlusion_ratio=0.0,
                    identity_relevance=0.7 if elem_type == ElementType.PERSON else 0.2,
                    story_relevance=0.6,
                    visual_relevance=0.6,
                    editability=EditabilityStatus.EDITABLE,
                    source_detector="yolo_grounding_dino",
                    provenance="object_detection_stage",
                )
            )

        # 4. Text Grounding
        if intelligence.ocr.text_regions:
            for idx, region in enumerate(intelligence.ocr.text_regions):
                elem_id = cls._make_element_id("text", idx, "ocr_text", region.bbox)
                elements.append(
                    SceneElement(
                        element_id=elem_id,
                        element_type=ElementType.TEXT,
                        category="text",
                        label=f"Text Region: '{region.text}'",
                        semantic_description=f"OCR Text reading '{region.text}'",
                        bbox=region.bbox,
                        confidence=region.confidence,
                        importance_rank=len(elements) + 1,
                        role=ElementRole.PRIMARY,
                        preserve_score=0.8,
                        replace_score=0.2,
                        edit_priority=1,
                        depth_level=0.1,  # Text overlay on top
                        occlusion_ratio=0.0,
                        identity_relevance=0.1,
                        story_relevance=0.85,
                        visual_relevance=0.85,
                        editability=EditabilityStatus.EDITABLE,
                        source_detector="easyocr",
                        provenance="ocr_stage",
                    )
                )

        return elements

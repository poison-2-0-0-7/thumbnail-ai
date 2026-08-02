"""
StyleExtractor component for Phase 1 of Module 10 Creator Style Learning.

Extracts structured ThumbnailStyleSignature from existing Module 4 ThumbnailIntelligence
and VRE outputs without introducing new CV models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from modules.models import ThumbnailIntelligence, ThumbnailStyleSignature


class StyleExtractor:
    """
    Read-only derivation stage extracting normalized style signatures from Module 4 outputs.
    """

    @staticmethod
    def extract_signature(
        video_id: str,
        channel_id: str,
        intelligence: ThumbnailIntelligence | dict[str, Any],
    ) -> ThumbnailStyleSignature:
        """
        Extract a ThumbnailStyleSignature from a video's ThumbnailIntelligence.
        """
        extracted_at = datetime.now(timezone.utc).isoformat()

        # Handle dict vs Pydantic object
        if isinstance(intelligence, dict):
            colors_data = intelligence.get("colors", {})
            comp_data = intelligence.get("composition", {})
            faces_data = intelligence.get("faces", {})
            ocr_data = intelligence.get("ocr", {})
            objects_data = intelligence.get("objects", [])

            dominant_colors = colors_data.get("dominant_colors", [])
            brightness = colors_data.get("brightness", 0.5)
            contrast = colors_data.get("contrast", 0.5)
            saturation = colors_data.get("saturation", 0.5)
            warm_or_cool = colors_data.get("warm_or_cool", "neutral")
            harmony_score = colors_data.get("harmony_score", 0.5)

            subject_placement = comp_data.get("subject_placement", "center")
            neg_space = comp_data.get("negative_space_ratio", 0.3)
            balance = comp_data.get("balance_score", 0.5)
            symmetry = comp_data.get("symmetry_score", 0.5)

            has_face = faces_data.get("has_face", False)
            faces_list = faces_data.get("faces", [])
            face_scale_ratio: Optional[float] = None
            if has_face and faces_list:
                total_area = 0.0
                for f in faces_list:
                    bbox = f.get("bbox", {}) if isinstance(f, dict) else getattr(f, "bbox", None)
                    if isinstance(bbox, dict):
                        w = max(0.0, bbox.get("x_max", 0.0) - bbox.get("x_min", 0.0))
                        h = max(0.0, bbox.get("y_max", 0.0) - bbox.get("y_min", 0.0))
                        total_area += w * h
                    elif bbox is not None:
                        w = max(0.0, getattr(bbox, "x_max", 0.0) - getattr(bbox, "x_min", 0.0))
                        h = max(0.0, getattr(bbox, "y_max", 0.0) - getattr(bbox, "y_min", 0.0))
                        total_area += w * h
                face_scale_ratio = min(1.0, total_area)

            text_cov = ocr_data.get("text_coverage_ratio", 0.0)
            text_regions = ocr_data.get("text_regions", [])
            text_region_count = len(text_regions)

            object_classes = []
            for obj in objects_data:
                label = obj.get("class_label") if isinstance(obj, dict) else getattr(obj, "class_label", "")
                if label and label not in object_classes:
                    object_classes.append(label)

        else: # Pydantic object
            dominant_colors = intelligence.colors.dominant_colors
            brightness = intelligence.colors.brightness
            contrast = intelligence.colors.contrast
            saturation = intelligence.colors.saturation
            warm_or_cool = intelligence.colors.warm_or_cool
            harmony_score = intelligence.colors.harmony_score

            subject_placement = intelligence.composition.subject_placement
            neg_space = intelligence.composition.negative_space_ratio
            balance = intelligence.composition.balance_score
            symmetry = intelligence.composition.symmetry_score

            face_scale_ratio = None
            if intelligence.faces.has_face and intelligence.faces.faces:
                total_area = 0.0
                for f in intelligence.faces.faces:
                    bbox = f.bbox
                    w = max(0.0, bbox.x_max - bbox.x_min)
                    h = max(0.0, bbox.y_max - bbox.y_min)
                    total_area += w * h
                face_scale_ratio = min(1.0, total_area)

            text_cov = intelligence.ocr.text_coverage_ratio
            text_region_count = len(intelligence.ocr.text_regions)

            object_classes = []
            for obj in intelligence.objects:
                label = getattr(obj, "class_label", "") or getattr(obj, "label", "")
                if label and label not in object_classes:
                    object_classes.append(label)

        return ThumbnailStyleSignature(
            video_id=video_id,
            channel_id=channel_id,
            dominant_colors=dominant_colors,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            warm_or_cool=warm_or_cool,
            color_harmony_score=harmony_score,
            subject_placement=subject_placement,
            negative_space_ratio=neg_space,
            balance_score=balance,
            symmetry_score=symmetry,
            face_scale_ratio=face_scale_ratio,
            text_coverage_ratio=text_cov,
            text_region_count=text_region_count,
            object_classes_present=object_classes,
            extracted_at=extracted_at,
        )

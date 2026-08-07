"""
scene_decomposer.py
===================

Scene Decomposition and Editable Layer Engine (Phase 12).

Decomposes a thumbnail scene into independent, editable layers (Background,
Primary Subject, Secondary Subjects, Foreground Objects, Props, Text, Logos) with
depth priorities, preservation requirements, mask paths, and cutouts.
"""

from __future__ import annotations

from typing import Optional
from models import AssetExtractionManifest
from thumbnail_understanding.schemas import (
    DecomposedScene,
    ElementType,
    LayerCategory,
    SceneElement,
    SceneGraph,
    SceneLayer,
)
from thumbnail_understanding.background_plate_builder import BackgroundPlateBuilder
from thumbnail_understanding.mask_validator import MaskValidator


class SceneDecomposer:
    """Decomposes scene graph into a complete multi-layer editable structure."""

    @classmethod
    def decompose_scene(
        cls,
        source_thumbnail_path: str,
        scene_graph: SceneGraph,
        asset_manifest: Optional[AssetExtractionManifest] = None,
    ) -> DecomposedScene:
        """
        Decompose scene graph into ordered SceneLayer objects.
        """
        layers: list[SceneLayer] = []

        # 1. Background Layer (depth_priority = 0)
        bg_path, is_reconstructed = BackgroundPlateBuilder.resolve_background_plate(
            source_thumbnail_path, asset_manifest
        )
        layers.append(
            SceneLayer(
                layer_id="layer_00_background",
                category=LayerCategory.BACKGROUND,
                image_path=bg_path,
                depth_priority=0,
                preservation_requirement="replaceable",
                editability="editable",
                source_provenance="background_plate_builder",
            )
        )

        # 2. Convert grounded SceneElements to SceneLayers
        for elem in scene_graph.elements:
            if elem.element_type == ElementType.BACKGROUND:
                continue

            # Categorize layer
            if elem.element_type == ElementType.PERSON:
                category = LayerCategory.PRIMARY_SUBJECT if elem.importance_rank <= 2 else LayerCategory.SECONDARY_SUBJECT
                preservation = "must_preserve" if elem.is_creator or elem.importance_rank == 1 else "preserve_if_possible"
            elif elem.element_type == ElementType.TEXT:
                category = LayerCategory.TEXT
                preservation = "preserve_if_possible"
            elif elem.element_type == ElementType.LOGO:
                category = LayerCategory.LOGO
                preservation = "must_preserve"
            elif elem.element_type == ElementType.PROP:
                category = LayerCategory.PROP
                preservation = "replaceable"
            else:
                category = LayerCategory.FOREGROUND_OBJECT
                preservation = "replaceable"

            # Match mask/cutout from asset_manifest if available
            mask_p: Optional[str] = elem.mask_path
            cutout_p: Optional[str] = elem.cutout_path

            if asset_manifest:
                if elem.element_type == ElementType.PERSON and asset_manifest.people:
                    # Match by index or bbox
                    person_asset = asset_manifest.people[0] if asset_manifest.people else None
                    if person_asset and person_asset.mask:
                        mask_p = person_asset.mask.file_path
                    if person_asset and person_asset.cutout:
                        cutout_p = person_asset.cutout.file_path

            # Depth priority: 1 = lowest non-bg, higher = further forward
            depth_pri = max(1, 10 - elem.importance_rank)

            layer = SceneLayer(
                layer_id=f"layer_{elem.element_id}",
                category=category,
                element_ref_id=elem.element_id,
                image_path=cutout_p,
                mask_path=mask_p,
                bounding_region=elem.bbox,
                polygon=elem.polygon,
                depth_priority=depth_pri,
                relationships=[
                    r.object_element_id for r in scene_graph.relationships if r.subject_element_id == elem.element_id
                ],
                preservation_requirement=preservation,
                editability="editable",
                source_provenance=elem.source_detector,
            )

            # Validate mask
            MaskValidator.validate_layer(layer)
            layers.append(layer)

        # Sort layers by depth_priority ascending (Background first, Foreground last)
        layers.sort(key=lambda l: l.depth_priority)

        return DecomposedScene(
            layers=layers,
            background_plate_path=bg_path,
            background_reconstructed=is_reconstructed,
            layer_count=len(layers),
        )

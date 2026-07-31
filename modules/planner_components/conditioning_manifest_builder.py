"""
conditioning_manifest_builder.py
================================

Converts a resolved GenerationConditioningContext into a list[PlanConditioningAsset]
for GenerationPlan persistence per §11 and §13.
"""

from __future__ import annotations

from typing import Literal, Optional

from generation_components.conditioning_asset_resolver import GenerationConditioningContext
from models import AssetExtractionManifest, PlanConditioningAsset
from planner_components.interfaces import IConditioningManifestBuilder


class ConditioningManifestBuilder(IConditioningManifestBuilder):
    """Converts GenerationConditioningContext into structured PlanConditioningAssets."""

    def build_manifest(
        self,
        context: GenerationConditioningContext,
        extraction_manifest: Optional[AssetExtractionManifest] = None,
    ) -> list[PlanConditioningAsset]:
        assets: list[PlanConditioningAsset] = []

        # 1. Role image paths
        for role, p in context.role_image_paths.items():
            source = self._determine_source(str(p), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role=role,
                    asset_id=role,
                    path=str(p),
                    kind="reference_image",
                    source_module=source,
                )
            )

        # 2. Role mask paths
        for role, p in context.role_mask_paths.items():
            source = self._determine_source(str(p), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role=role,
                    asset_id=f"mask_{role}",
                    path=str(p),
                    kind="mask",
                    source_module=source,
                )
            )

        # 3. Depth path
        if context.depth_path:
            source = self._determine_source(str(context.depth_path), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role="depth_map",
                    asset_id="depth_map",
                    path=str(context.depth_path),
                    kind="depth",
                    source_module=source,
                )
            )

        # 4. Canny path
        if context.canny_path:
            source = self._determine_source(str(context.canny_path), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role="canny_map",
                    asset_id="canny_map",
                    path=str(context.canny_path),
                    kind="canny",
                    source_module=source,
                )
            )

        # 5. Segmentation path
        if context.segmentation_path:
            source = self._determine_source(str(context.segmentation_path), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role="segmentation_map",
                    asset_id="segmentation_map",
                    path=str(context.segmentation_path),
                    kind="segmentation",
                    source_module=source,
                )
            )

        # 6. IP-Adapter reference paths
        for key, p in context.ip_adapter_reference_paths.items():
            source = self._determine_source(str(p), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role=key,
                    asset_id=f"ip_adapter_{key}",
                    path=str(p),
                    kind="ip_adapter_reference",
                    source_module=source,
                )
            )

        # 7. Text exclusion mask path
        if context.text_exclusion_mask_path:
            source = self._determine_source(str(context.text_exclusion_mask_path), extraction_manifest)
            assets.append(
                PlanConditioningAsset(
                    role="text_exclusion",
                    asset_id="text_exclusion_mask",
                    path=str(context.text_exclusion_mask_path),
                    kind="text_exclusion_mask",
                    source_module=source,
                )
            )

        return assets

    @staticmethod
    def _determine_source(
        file_path_str: str,
        extraction_manifest: Optional[AssetExtractionManifest],
    ) -> Literal["module8", "vre", "module10"]:
        if "asset_extraction" in file_path_str:
            return "module8"
        elif "visual_references" in file_path_str:
            return "vre"
        else:
            return "module10"

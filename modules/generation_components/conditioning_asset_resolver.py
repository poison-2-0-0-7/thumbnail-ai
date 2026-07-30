"""
conditioning_asset_resolver.py
===============================

Normalizes GenerationBundle, CompositionWorkspace, and legacy ReferenceAssets
into a unified GenerationConditioningContext for Module 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from composition_components.generation_bundle_builder import GenerationBundleBuilder
from config import MODULE7_LOG_PATH
from generation_components.interfaces import IConditioningAssetResolver
from module7_exceptions import ConditioningResolutionError
from models import CompositionWorkspace, GenerationBundle, GenerationProfile, LayerDecision
from loguru import logger

if TYPE_CHECKING:
    from image_generator import ReferenceAssets


def _configure_logger() -> None:
    """Ensure Loguru sink is configured for Module 7."""
    try:
        logger.add(
            MODULE7_LOG_PATH,
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            enqueue=True,
        )
    except ValueError:
        pass

_configure_logger()


@dataclass(frozen=True)
class LayerConditioning:
    """Per-layer conditioning details extracted from a CompositionWorkspace."""

    role: str
    decision: str
    mask_path: Path | None = None
    feather_px: int = 0
    z_index: int = 0
    crop_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class GenerationConditioningContext:
    """Frozen, unified conditioning context value object for Module 7."""

    source_thumbnail_path: Path | None = None
    canvas_width: int = 1280
    canvas_height: int = 720
    aspect_ratio: str = "16:9"
    role_image_paths: dict[str, Path] = field(default_factory=dict)
    role_mask_paths: dict[str, Path] = field(default_factory=dict)
    depth_path: Path | None = None
    canny_path: Path | None = None
    segmentation_path: Path | None = None
    ip_adapter_reference_paths: dict[str, Path] = field(default_factory=dict)
    text_exclusion_mask_path: Path | None = None
    layer_order: tuple[str, ...] = ()
    per_layer: dict[str, LayerConditioning] | None = None


class ConditioningAssetResolver(IConditioningAssetResolver):
    """Resolves and validates conditioning assets into a GenerationConditioningContext."""

    def resolve(
        self,
        bundle: GenerationBundle | None = None,
        workspace: CompositionWorkspace | None = None,
        reference_assets: ReferenceAssets | None = None,
        profile: GenerationProfile | None = None,
    ) -> GenerationConditioningContext:
        """
        Resolve all available inputs into a GenerationConditioningContext.

        Args:
            bundle: Optional GenerationBundle instance.
            workspace: Optional CompositionWorkspace instance.
            reference_assets: Optional legacy ReferenceAssets instance.
            profile: Optional GenerationProfile instance.

        Returns:
            Resolved GenerationConditioningContext instance.

        Raises:
            ConditioningResolutionError: If a referenced path does not exist on disk or is malformed.
        """
        # If no bundle provided but workspace is present, derive bundle using Module 10 builder
        if bundle is None and workspace is not None:
            try:
                bundle = GenerationBundleBuilder().build_generation_bundle(workspace)
            except Exception as exc:
                raise ConditioningResolutionError(f"Failed to derive GenerationBundle from workspace: {exc}") from exc

        # 1. Resolve source_thumbnail_path from reference_assets
        source_thumbnail_path: Path | None = None
        if reference_assets is not None and getattr(reference_assets, "source_thumbnail_path", None):
            p = Path(reference_assets.source_thumbnail_path)
            self._verify_file_exists(p, "source_thumbnail_path")
            source_thumbnail_path = p

        if bundle is None:
            # All-empty / legacy context
            context = GenerationConditioningContext(source_thumbnail_path=source_thumbnail_path)
            logger.info(
                "Resolved GenerationConditioningContext for legacy/empty input: source_thumbnail={thumb}",
                thumb=source_thumbnail_path,
            )
            return context

        # 2. Canvas parameters
        canvas_width = bundle.canvas.width if bundle.canvas else 1280
        canvas_height = bundle.canvas.height if bundle.canvas else 720
        aspect_ratio = bundle.canvas.aspect_ratio if bundle.canvas else "16:9"

        # 3. Role image paths
        role_image_paths: dict[str, Path] = {}
        for role, raw_path in bundle.reference_image_paths.items():
            if raw_path:
                p = Path(raw_path)
                self._verify_file_exists(p, f"reference_image_paths['{role}']")
                role_image_paths[role] = p

        # 4. Role mask paths
        role_mask_paths: dict[str, Path] = {}
        for role, raw_path in bundle.mask_paths.items():
            if raw_path:
                p = Path(raw_path)
                self._verify_file_exists(p, f"mask_paths['{role}']")
                role_mask_paths[role] = p

        # 5. Single ControlNet maps
        depth_path: Path | None = None
        if bundle.depth_path:
            p = Path(bundle.depth_path)
            self._verify_file_exists(p, "depth_path")
            depth_path = p

        canny_path: Path | None = None
        if bundle.canny_path:
            p = Path(bundle.canny_path)
            self._verify_file_exists(p, "canny_path")
            canny_path = p

        # 6. Defensive attribute access for potential future/extended fields (§7.3)
        segmentation_path: Path | None = None
        raw_seg = getattr(bundle, "segmentation_path", None)
        if raw_seg:
            p = Path(raw_seg)
            self._verify_file_exists(p, "segmentation_path")
            segmentation_path = p

        ip_adapter_reference_paths: dict[str, Path] = {}
        raw_ip = getattr(bundle, "ip_adapter_reference_paths", {})
        if isinstance(raw_ip, dict):
            for key, raw_path in raw_ip.items():
                if raw_path:
                    p = Path(raw_path)
                    self._verify_file_exists(p, f"ip_adapter_reference_paths['{key}']")
                    ip_adapter_reference_paths[key] = p

        text_exclusion_mask_path: Path | None = None
        raw_text_ex = getattr(bundle, "text_exclusion_mask_path", None)
        if raw_text_ex:
            p = Path(raw_text_ex)
            self._verify_file_exists(p, "text_exclusion_mask_path")
            text_exclusion_mask_path = p

        layer_order = tuple(bundle.layer_order)

        # 7. Extract per_layer from CompositionWorkspace if supplied
        per_layer: dict[str, LayerConditioning] | None = None
        if workspace is not None:
            per_layer = {}
            for layer in workspace.layers:
                m_path: Path | None = None
                feather = 0
                if layer.placement.mask and layer.placement.mask.mask_path:
                    m_path = Path(layer.placement.mask.mask_path)
                    self._verify_file_exists(m_path, f"layer '{layer.layer_id}' mask_path")
                    feather = layer.placement.mask.feather_px

                crop_box: tuple[int, int, int, int] | None = None
                if layer.placement.transform.crop_box:
                    cb = layer.placement.transform.crop_box
                    crop_box = (cb.x, cb.y, cb.width, cb.height)

                per_layer[layer.layer_id] = LayerConditioning(
                    role=layer.placement.role.value,
                    decision=layer.placement.decision.value,
                    mask_path=m_path,
                    feather_px=feather,
                    z_index=layer.placement.z_index,
                    crop_box=crop_box,
                )

        context = GenerationConditioningContext(
            source_thumbnail_path=source_thumbnail_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            aspect_ratio=aspect_ratio,
            role_image_paths=role_image_paths,
            role_mask_paths=role_mask_paths,
            depth_path=depth_path,
            canny_path=canny_path,
            segmentation_path=segmentation_path,
            ip_adapter_reference_paths=ip_adapter_reference_paths,
            text_exclusion_mask_path=text_exclusion_mask_path,
            layer_order=layer_order,
            per_layer=per_layer,
        )

        logger.info(
            "Resolved GenerationConditioningContext for video_id={video_id}: "
            "roles={roles}, masks={masks}, depth={has_depth}, canny={has_canny}, ip_adapter_refs={n_refs}",
            video_id=bundle.video_id,
            roles=list(role_image_paths.keys()),
            masks=list(role_mask_paths.keys()),
            has_depth=depth_path is not None,
            has_canny=canny_path is not None,
            n_refs=len(ip_adapter_reference_paths),
        )

        return context

    def _verify_file_exists(self, path: Path, field_name: str) -> None:
        """Helper to ensure a non-empty path exists on disk."""
        if not path.is_file():
            raise ConditioningResolutionError(
                f"Referenced conditioning asset for '{field_name}' does not exist on disk: '{path}'"
            )

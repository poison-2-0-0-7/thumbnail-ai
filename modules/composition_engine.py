"""
composition_engine.py
=====================

Asset Composer orchestrator (Module 10).

Resolves geometry, transforms, layer decisions, and mask bindings into a
versioned Composition Workspace and ComfyUI Generation Bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Optional

from loguru import logger

from composition_components.asset_registry import AssetRegistry
from composition_components.composition_validator import CompositionValidator
from composition_components.decision_resolver import DecisionResolver
from composition_components.generation_bundle_builder import GenerationBundleBuilder
from composition_components.interfaces import (
    IAssetRegistry,
    ICompositionValidator,
    IDecisionResolver,
    IGenerationBundleBuilder,
    ILayerManager,
    IMaskManager,
    IMetadataBuilder,
    IPlacementEngine,
    ITransformEngine,
    IWorkspaceManager,
)
from composition_components.layer_manager import LayerManager
from composition_components.mask_manager import MaskManager
from composition_components.metadata_builder import MetadataBuilder, compute_model_hash
from composition_components.placement_engine import PlacementEngine
from composition_components.transform_engine import TransformEngine
from composition_components.workspace_manager import WorkspaceManager
from composition_exceptions import (
    AssetRegistryError,
    CompositionInputInvalidError,
    WorkspaceValidationError,
)
from config import (
    COMPOSITION_CACHE_ENABLED,
    COMPOSITION_RESOLVE_CANNY_ASSET_KEY,
    COMPOSITION_SAFE_MARGIN_PX,
    COMPOSITION_WORKSPACE_ROOT,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    MODULE10_LOG_PATH,
)
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    DecisionManifest,
    GenerationBundle,
    LayerRole,
    LightingAdjustment,
    PlacementConstraints,
    PromptPackage,
    RedesignSpecification,
)
from visual_reference_engine import VisualReferenceEngine


def _configure_logger() -> None:
    """Configure Loguru logger sink for Module 10."""
    MODULE10_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        MODULE10_LOG_PATH,
        rotation="10 MB",
        retention="30 days",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        level="DEBUG",
    )


_configure_logger()


class AssetComposer:
    """Orchestrator for Module 10 Composition Workspace Preparation."""

    def __init__(
        self,
        storage_root: Optional[Path] = None,
        vre_engine: Optional[VisualReferenceEngine] = None,
        asset_registry: Optional[IAssetRegistry] = None,
        decision_resolver: Optional[IDecisionResolver] = None,
        placement_engine: Optional[IPlacementEngine] = None,
        transform_engine: Optional[ITransformEngine] = None,
        mask_manager: Optional[IMaskManager] = None,
        layer_manager: Optional[ILayerManager] = None,
        metadata_builder: Optional[IMetadataBuilder] = None,
        validator: Optional[ICompositionValidator] = None,
        workspace_manager: Optional[IWorkspaceManager] = None,
        bundle_builder: Optional[IGenerationBundleBuilder] = None,
    ) -> None:
        self._storage_root = Path(storage_root) if storage_root else COMPOSITION_WORKSPACE_ROOT
        self._vre_engine = vre_engine if vre_engine is not None else VisualReferenceEngine()
        self._asset_registry = asset_registry if asset_registry is not None else AssetRegistry()
        self._decision_resolver = (
            decision_resolver if decision_resolver is not None else DecisionResolver()
        )
        self._placement_engine = (
            placement_engine if placement_engine is not None else PlacementEngine()
        )
        self._transform_engine = (
            transform_engine if transform_engine is not None else TransformEngine()
        )
        self._mask_manager = mask_manager if mask_manager is not None else MaskManager()
        self._layer_manager = layer_manager if layer_manager is not None else LayerManager()
        self._metadata_builder = (
            metadata_builder if metadata_builder is not None else MetadataBuilder()
        )
        self._validator = validator if validator is not None else CompositionValidator()
        self._workspace_manager = (
            workspace_manager
            if workspace_manager is not None
            else WorkspaceManager(root_dir=self._storage_root)
        )
        self._bundle_builder = (
            bundle_builder if bundle_builder is not None else GenerationBundleBuilder()
        )

    def compose_workspace(
        self,
        video_id: str,
        options: Optional[dict] = None,
        decision_manifest: Optional[DecisionManifest] = None,
    ) -> CompositionWorkspace:
        """
        Compose a complete CompositionWorkspace for video_id.

        Args:
            video_id: YouTube video identifier.
            options: Optional execution configuration dict.
            decision_manifest: Optional DecisionManifest from Module 9.

        Returns:
            CompositionWorkspace instance.
        """
        start_time = time.monotonic()
        options = options or {}
        use_cache = options.get("use_cache", COMPOSITION_CACHE_ENABLED)

        logger.info(f"Starting workspace composition for video_id '{video_id}'")

        # 1. Load upstream RedesignSpecification
        spec_path = DEFAULT_REDESIGN_SPEC_DIR / f"{video_id}.json"
        if not spec_path.is_file():
            raise CompositionInputInvalidError(
                f"RedesignSpecification file not found for video_id '{video_id}' at '{spec_path}'."
            )
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
            spec = RedesignSpecification.model_validate_json(spec_content)
        except Exception as exc:
            raise CompositionInputInvalidError(
                f"Failed to parse RedesignSpecification for '{video_id}': {exc}"
            ) from exc

        # 2. Load upstream PromptPackage
        package_path = DEFAULT_PROMPT_PACKAGE_DIR / f"{video_id}.json"
        if not package_path.is_file():
            raise CompositionInputInvalidError(
                f"PromptPackage file not found for video_id '{video_id}' at '{package_path}'."
            )
        try:
            package_content = package_path.read_text(encoding="utf-8")
            package = PromptPackage.model_validate_json(package_content)
        except Exception as exc:
            raise CompositionInputInvalidError(
                f"Failed to parse PromptPackage for '{video_id}': {exc}"
            ) from exc

        # 3. Obtain VisualReferenceManifest via VRE
        manifest = self._vre_engine.prepare_assets(
            video_id, spec.source_thumbnail_path, options=options
        )

        # 4. Check cache resume
        spec_hash = compute_model_hash(spec)
        package_hash = compute_model_hash(package)
        expected_hashes = {
            "vre_source_hash": manifest.source_hash,
            "redesign_spec_hash": spec_hash,
            "prompt_package_hash": package_hash,
        }

        if use_cache:
            cached = self._workspace_manager.resume(video_id, expected_hashes)
            if cached is not None:
                logger.info(f"Cache hit for workspace '{video_id}'")
                return cached

        # 5. Index & verify VRE assets
        self._asset_registry.index(manifest)
        invalid_assets = self._asset_registry.verify_integrity()
        if invalid_assets:
            raise AssetRegistryError(
                f"VRE asset integrity failure for video_id '{video_id}': invalid asset(s) {invalid_assets}"
            )

        # 6. Canvas Transform
        canvas = CanvasTransform(
            width=package.generation_parameters.width,
            height=package.generation_parameters.height,
            aspect_ratio=package.generation_parameters.aspect_ratio,
        )

        # 7. Resolve decisions & placement geometry
        decisions = self._decision_resolver.resolve(
            spec, decision_manifest=decision_manifest
        )
        placements_dict = self._placement_engine.place(spec, canvas)
        text_placement = self._placement_engine.resolve_text_zones(spec, canvas)
        focal_zone_px = self._placement_engine.resolve_focal_zone(spec, canvas)

        # 8. Build layers
        layers: list[CompositionLayer] = []
        depth_asset = self._asset_registry.resolve("depth_map")
        depth_hint_path = depth_asset.file_path if depth_asset else None
        canny_asset = self._asset_registry.resolve(COMPOSITION_RESOLVE_CANNY_ASSET_KEY)
        canny_hint_path = canny_asset.file_path if canny_asset else None

        for element_key, role, decision, rationale in decisions:
            source_path = None
            if role == LayerRole.BACKGROUND:
                asset = self._asset_registry.resolve("background")
                source_path = asset.file_path if asset else None
            elif role == LayerRole.PERSON:
                asset = self._asset_registry.resolve("creator_face")
                source_path = asset.file_path if asset else None
            elif role == LayerRole.OBJECT:
                asset = self._asset_registry.resolve("object_crop")
                source_path = asset.file_path if asset else None

            mask_ref = self._mask_manager.bind(self._asset_registry, role)
            pixel_bbox = placements_dict.get(role.value)
            crop_tighter = (
                spec.subject_treatment.crop_tighter if role == LayerRole.PERSON else False
            )
            transform = self._transform_engine.resolve(pixel_bbox, decision, crop_tighter)
            z_index = LayerManager.get_role_z_index(role)

            placement = AssetPlacement(
                asset_id=element_key,
                role=role,
                decision=decision,
                source_path=source_path,
                mask=mask_ref,
                transform=transform,
                z_index=z_index,
                rationale=rationale,
            )

            layer = CompositionLayer(
                layer_id=f"layer_{element_key}",
                placement=placement,
                depth_hint_path=depth_hint_path,
                canny_hint_path=canny_hint_path,
            )
            layers.append(layer)

        ordered_layers = self._layer_manager.order(layers)
        groups = self._layer_manager.group(ordered_layers)

        lighting = LightingAdjustment(
            target_brightness=spec.color_direction.target_brightness,
            target_contrast=spec.color_direction.target_contrast,
            target_saturation=spec.color_direction.target_saturation,
            warm_or_cool=spec.color_direction.warm_or_cool,
        )

        constraints = PlacementConstraints(
            safe_margin_px=COMPOSITION_SAFE_MARGIN_PX,
            avoid_zones_px=text_placement.avoid_zones_px,
            focal_zone_px=focal_zone_px,
        )

        metadata = self._metadata_builder.build(video_id, manifest, spec, package)
        statistics = self._metadata_builder.statistics(ordered_layers)

        elapsed = time.monotonic() - start_time

        workspace = CompositionWorkspace(
            video_id=video_id,
            canvas=canvas,
            layers=ordered_layers,
            groups=groups,
            text_placement=text_placement,
            lighting=lighting,
            constraints=constraints,
            statistics=statistics,
            metadata=metadata,
            status="success",
            duration_seconds=elapsed,
        )

        # 9. Validate workspace
        validation_errors = self._validator.validate(workspace)
        if validation_errors:
            raise WorkspaceValidationError(
                f"Workspace validation failed for video_id '{video_id}': {validation_errors}"
            )

        logger.info(
            f"Successfully composed workspace for '{video_id}' with {len(ordered_layers)} layers in {elapsed:.3f}s"
        )
        return workspace

    def save_workspace(self, workspace: CompositionWorkspace) -> Path:
        """Atomically save workspace to disk."""
        return self._workspace_manager.persist(workspace)

    def load_workspace(self, video_id: str) -> CompositionWorkspace:
        """Load workspace from disk."""
        return self._workspace_manager.load(video_id)

    def validate_workspace(self, workspace: CompositionWorkspace) -> CompositionWorkspace:
        """Validate workspace object."""
        errors = self._validator.validate(workspace)
        if errors:
            raise WorkspaceValidationError(
                f"Workspace validation failed for '{workspace.video_id}': {errors}"
            )
        return workspace

    def resume_workspace(self, video_id: str) -> Optional[CompositionWorkspace]:
        """Attempt to resume a workspace if present."""
        return self._workspace_manager.resume(video_id, {})

    def build_generation_bundle(self, workspace: CompositionWorkspace) -> GenerationBundle:
        """Flatten workspace to GenerationBundle."""
        return self._bundle_builder.build_generation_bundle(workspace)

    def prepare_generation_workspace(
        self,
        video_id: str,
        options: Optional[dict] = None,
        decision_manifest: Optional[DecisionManifest] = None,
    ) -> GenerationBundle:
        """
        Convenience pipeline: compose_workspace -> validate_workspace -> save_workspace -> build_generation_bundle.
        """
        workspace = self.compose_workspace(
            video_id, options=options, decision_manifest=decision_manifest
        )
        self.validate_workspace(workspace)
        self.save_workspace(workspace)
        return self.build_generation_bundle(workspace)

    def clean_workspace(self, video_id: str) -> bool:
        """Symmetrical clean helper matching VisualReferenceEngine.clean_assets."""
        return self._workspace_manager.purge(video_id)

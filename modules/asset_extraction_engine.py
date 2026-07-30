"""
asset_extraction_engine.py
===========================

Module 8 Orchestrator (Asset Extraction Engine).
Extracts, persists, and manages reusable thumbnail visual assets.
Cache-aware, resumable, fine-grained family dispatch under single-model GPU lock.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from modules.config import (
    ASSET_EXTRACTION_CACHE_ENABLED,
    ASSET_EXTRACTION_ENGINE_VERSION,
    ASSET_EXTRACTION_FAMILY_ORDER,
    ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX,
    ASSET_MANIFEST_FILENAME,
    DEFAULT_ASSET_EXTRACTION_DIR,
    MODULE8_LOG_PATH,
)
from modules.asset_extraction_components.asset_writer import AssetExtractionWriter
from modules.asset_extraction_components.composition_processor import CompositionProcessor
from modules.asset_extraction_components.effects_processor import EffectsProcessor
from modules.asset_extraction_components.interfaces import (
    IAssetExtractionWriter,
    IAssetManifestBuilder,
    ICompositionAssetProcessor,
    IEffectsProcessor,
    IModelBridge,
    IObjectProcessor,
    IPersonProcessor,
    ISceneProcessor,
    ITypographyProcessor,
    IVisualPropertiesProcessor,
)
from modules.asset_extraction_components.manifest_builder import ManifestBuilder
from modules.asset_extraction_components.model_bridge import ModelBridge
from modules.asset_extraction_components.object_processor import ObjectProcessor
from modules.asset_extraction_components.person_processor import PersonProcessor
from modules.asset_extraction_components.scene_processor import SceneProcessor
from modules.asset_extraction_components.typography_processor import TypographyProcessor
from modules.asset_extraction_components.visual_properties_processor import VisualPropertiesProcessor
from modules.asset_extraction_exceptions import (
    AssetExtractionError,
    AssetFamilyModelError,
    AssetWriteError,
    IntelligenceReportInvalidError,
    ManifestNotFoundError,
    ManifestValidationError,
    SourceImageNotFoundError,
)
from modules.models import (
    AssetExtractionManifest,
    AssetExtractionStatus,
    AssetFileRef,
    CompositionAsset,
    EffectsAsset,
    ObjectAsset,
    PersonAsset,
    SceneAsset,
    ThumbnailIntelligence,
    TypographyAsset,
    VisualPropertiesAsset,
)

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    MODULE8_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE8_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


class AssetExtractionEngine:
    """Orchestrator class for Module 8 Asset Extraction Engine."""

    def __init__(
        self,
        storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
        person_processor: IPersonProcessor | None = None,
        scene_processor: ISceneProcessor | None = None,
        object_processor: IObjectProcessor | None = None,
        typography_processor: ITypographyProcessor | None = None,
        visual_properties_processor: IVisualPropertiesProcessor | None = None,
        composition_processor: ICompositionAssetProcessor | None = None,
        effects_processor: IEffectsProcessor | None = None,
        asset_writer: IAssetExtractionWriter | None = None,
        manifest_builder: IAssetManifestBuilder | None = None,
        model_bridge: IModelBridge | None = None,
        cache_enabled: bool = ASSET_EXTRACTION_CACHE_ENABLED,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.cache_enabled = cache_enabled
        self.model_bridge = model_bridge or ModelBridge()

        self.person_processor = person_processor or PersonProcessor(model_bridge=self.model_bridge)
        self.scene_processor = scene_processor or SceneProcessor(model_bridge=self.model_bridge)
        self.object_processor = object_processor or ObjectProcessor(model_bridge=self.model_bridge)
        self.typography_processor = typography_processor or TypographyProcessor()
        self.visual_properties_processor = (
            visual_properties_processor or VisualPropertiesProcessor()
        )
        self.composition_processor = composition_processor or CompositionProcessor()
        self.effects_processor = effects_processor or EffectsProcessor()

        self.asset_writer = asset_writer or AssetExtractionWriter()
        self.manifest_builder = manifest_builder or ManifestBuilder()

    def extract(
        self,
        video_id: str,
        source_image_path: str,
        intelligence: ThumbnailIntelligence,
        options: Optional[dict] = None,
    ) -> AssetExtractionManifest:
        """Extract every asset family for one thumbnail. Cache-aware, resumable."""
        started_time = time.monotonic()
        video_id = self._validate_video_id(video_id)
        source_path = Path(source_image_path)

        if not source_path.exists() or not source_path.is_file():
            raise SourceImageNotFoundError(f"Source thumbnail image missing: {source_path}")

        if intelligence is None or intelligence.status == "error":
            raise IntelligenceReportInvalidError(
                f"Cannot seed asset extraction for video {video_id} with invalid intelligence report"
            )

        if intelligence.status == "partial":
            logger.warning(
                "Extracting assets from partial intelligence report video_id={id}", id=video_id
            )

        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        intelligence_hash = hashlib.sha256(
            intelligence.model_dump_json().encode("utf-8")
        ).hexdigest()

        shard_dir = self.storage_root / video_id
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Cache check & resume verification
        is_fully_cached, valid_completed_families, existing_manifest = self._check_cache(
            video_id, source_hash, intelligence_hash, shard_dir
        )

        if self.cache_enabled and is_fully_cached and existing_manifest:
            logger.info("Full asset manifest cache hit for video_id={id}", id=video_id)
            return existing_manifest

        # Load image for processing
        image = cv2.imread(str(source_path))
        if image is None or image.size == 0:
            raise SourceImageNotFoundError(f"Could not decode image at {source_path}")

        image = self._downscale_if_needed(image)

        # Initialize family results state
        completed_families = list(valid_completed_families)
        partial_failure_reasons: list[str] = (
            list(existing_manifest.partial_failure_reasons) if existing_manifest else []
        )
        status = existing_manifest.status if existing_manifest else AssetExtractionStatus.SUCCESS

        people_assets: list[PersonAsset] = (
            list(existing_manifest.people) if existing_manifest else []
        )
        scene_asset: Optional[SceneAsset] = (
            existing_manifest.scene if existing_manifest else None
        )
        object_assets: list[ObjectAsset] = (
            list(existing_manifest.objects) if existing_manifest else []
        )
        typography_assets: list[TypographyAsset] = (
            list(existing_manifest.typography) if existing_manifest else []
        )
        visual_props_asset: Optional[VisualPropertiesAsset] = (
            existing_manifest.visual_properties if existing_manifest else None
        )
        comp_asset: Optional[CompositionAsset] = (
            existing_manifest.composition if existing_manifest else None
        )
        effects_asset: Optional[EffectsAsset] = (
            existing_manifest.effects if existing_manifest else None
        )

        # Execute missing families sequentially
        for family in ASSET_EXTRACTION_FAMILY_ORDER:
            if self.cache_enabled and family in completed_families:
                logger.debug(
                    "Skipping cached family family={family} video_id={id}",
                    family=family,
                    id=video_id,
                )
                continue

            logger.info(
                "Extracting asset family family={family} video_id={id}",
                family=family,
                id=video_id,
            )
            family_start = time.monotonic()

            try:
                if family == "typography":
                    typography_assets = self._extract_typography(
                        image, intelligence, shard_dir / "typography"
                    )
                elif family == "visual_properties":
                    visual_props_asset = self._extract_visual_properties(
                        image, intelligence, shard_dir / "visual"
                    )
                elif family == "composition":
                    comp_asset = self._extract_composition(
                        image, intelligence, shard_dir / "composition"
                    )
                elif family == "objects":
                    object_assets = self._extract_objects(
                        image, intelligence, shard_dir / "objects"
                    )
                elif family == "people":
                    people_assets = self._extract_people(
                        image, intelligence, shard_dir / "people"
                    )
                elif family == "scene":
                    scene_asset = self._extract_scene(image, shard_dir / "scene")
                elif family == "effects":
                    effects_asset = self._extract_effects(
                        image, shard_dir / "effects"
                    )

                completed_families.append(family)
                family_elapsed_ms = (time.monotonic() - family_start) * 1000.0
                logger.info(
                    "Family extraction complete family={family} elapsed_ms={ms:.2f}",
                    family=family,
                    ms=family_elapsed_ms,
                )
            except Exception as exc:
                status = AssetExtractionStatus.PARTIAL
                reason = f"Family '{family}' failed: {exc}"
                partial_failure_reasons.append(reason)
                logger.error(
                    "Asset family failed family={family} video_id={id} exc={exc}",
                    family=family,
                    id=video_id,
                    exc=str(exc),
                )
                completed_families.append(family)  # Mark terminal to allow resume

            # Incremental manifest state update after each family
            intermediate_manifest = self.manifest_builder.build(
                video_id=video_id,
                source_thumbnail_path=str(source_path),
                source_hash=source_hash,
                intelligence_hash=intelligence_hash,
                engine_version=ASSET_EXTRACTION_ENGINE_VERSION,
                people=people_assets,
                scene=scene_asset,
                objects=object_assets,
                typography=typography_assets,
                visual_properties=visual_props_asset,
                composition=comp_asset,
                effects=effects_asset,
                status=status,
                partial_failure_reasons=partial_failure_reasons,
                completed_families=completed_families,
                total_duration_seconds=round(time.monotonic() - started_time, 4),
            )
            self.manifest_builder.serialize_to_disk(
                intermediate_manifest, shard_dir / ASSET_MANIFEST_FILENAME
            )

        total_duration = time.monotonic() - started_time
        final_manifest = self.manifest_builder.build(
            video_id=video_id,
            source_thumbnail_path=str(source_path),
            source_hash=source_hash,
            intelligence_hash=intelligence_hash,
            engine_version=ASSET_EXTRACTION_ENGINE_VERSION,
            people=people_assets,
            scene=scene_asset,
            objects=object_assets,
            typography=typography_assets,
            visual_properties=visual_props_asset,
            composition=comp_asset,
            effects=effects_asset,
            status=status,
            partial_failure_reasons=partial_failure_reasons,
            completed_families=completed_families,
            total_duration_seconds=round(total_duration, 4),
        )
        self.manifest_builder.serialize_to_disk(
            final_manifest, shard_dir / ASSET_MANIFEST_FILENAME
        )
        logger.info(
            "Asset extraction complete video_id={id} status={status} duration={duration:.2f}s",
            id=video_id,
            status=status.value,
            duration=total_duration,
        )
        return final_manifest

    def clean_assets(self, video_id: str) -> bool:
        """Remove generated shard directory for one video_id."""
        video_id = self._validate_video_id(video_id)
        shard_dir = self.storage_root / video_id
        return self.asset_writer.purge_directory(shard_dir)

    def _check_cache(
        self,
        video_id: str,
        source_hash: str,
        intelligence_hash: str,
        shard_dir: Path,
    ) -> tuple[bool, list[str], Optional[AssetExtractionManifest]]:
        """Verify on-disk manifest cache integrity and return completed families."""
        manifest_path = shard_dir / ASSET_MANIFEST_FILENAME
        if not manifest_path.exists():
            return False, [], None

        try:
            manifest = load_asset_manifest(video_id, storage_root=self.storage_root)
            if (
                manifest.source_hash != source_hash
                or manifest.intelligence_hash != intelligence_hash
            ):
                logger.info(
                    "Cache miss (hash mismatch) for video_id={id}", id=video_id
                )
                return False, [], None

            valid_families: list[str] = []
            for family in manifest.completed_families:
                if self._verify_family_files_exist(manifest, family):
                    valid_families.append(family)

            is_full_hit = len(valid_families) == len(ASSET_EXTRACTION_FAMILY_ORDER)
            return is_full_hit, valid_families, manifest
        except Exception as exc:
            logger.warning(
                "Cache manifest corrupted for video_id={id}: {exc}",
                id=video_id,
                exc=str(exc),
            )
            return False, [], None

    def _verify_family_files_exist(
        self, manifest: AssetExtractionManifest, family: str
    ) -> bool:
        """Walk file references for a family and confirm existence and non-zero size."""

        def check_ref(ref: Optional[AssetFileRef]) -> bool:
            if ref is None:
                return True
            p = Path(ref.file_path)
            return p.exists() and p.is_file() and p.stat().st_size > 0

        if family == "people":
            for p in manifest.people:
                if not check_ref(p.face) or not check_ref(p.face_mask) or not check_ref(p.body_mask):
                    return False
        elif family == "scene":
            if manifest.scene:
                for ref in (
                    manifest.scene.background,
                    manifest.scene.foreground,
                    manifest.scene.depth_map,
                    manifest.scene.segmentation_map,
                    manifest.scene.sky_mask,
                    manifest.scene.ground_mask,
                ):
                    if not check_ref(ref):
                        return False
        elif family == "objects":
            for obj in manifest.objects:
                if not check_ref(obj.crop) or not check_ref(obj.mask):
                    return False
        elif family == "typography":
            for typo in manifest.typography:
                if not check_ref(typo.crop):
                    return False
        elif family == "composition":
            if manifest.composition:
                for ref in (
                    manifest.composition.eye_flow_map,
                    manifest.composition.negative_space_mask,
                    manifest.composition.visual_hierarchy_overlay,
                ):
                    if not check_ref(ref):
                        return False
        return True

    def _extract_typography(
        self, image: np.ndarray, intelligence: ThumbnailIntelligence, family_dir: Path
    ) -> list[TypographyAsset]:
        family_dir.mkdir(parents=True, exist_ok=True)
        regions = intelligence.ocr.text_regions if intelligence.ocr else []
        results = self.typography_processor.process(image, regions)

        assets: list[TypographyAsset] = []
        boxes_sidecar: list[dict[str, Any]] = []

        for item in results:
            idx = item["text_region_index"]
            crop_path = family_dir / f"text_region_{idx+1:02d}.png"
            self.asset_writer.write_image(item["crop"], crop_path)
            ref = self._create_file_ref("typography_crop", crop_path, item["crop"], source="derived")

            asset = TypographyAsset(
                text_region_index=idx,
                crop=ref,
                text=item["text"],
                bbox=item["bbox"],
                estimated_font_family_guess=item["estimated_font_family_guess"],
                estimated_font_size_px=item["estimated_font_size_px"],
                alignment=item["alignment"],
                dominant_text_color=item["dominant_text_color"],
                has_stroke_or_outline=item["has_stroke_or_outline"],
                source_text_region_index=item["source_text_region_index"],
            )
            assets.append(asset)
            boxes_sidecar.append(
                {
                    "text_region_index": idx,
                    "text": item["text"],
                    "bbox": item["bbox"].model_dump(),
                    "alignment": item["alignment"],
                    "dominant_text_color": item["dominant_text_color"],
                }
            )

        self.asset_writer.write_json_sidecar(
            boxes_sidecar, family_dir / "text_boxes.json"
        )
        return assets

    def _extract_visual_properties(
        self, image: np.ndarray, intelligence: ThumbnailIntelligence, family_dir: Path
    ) -> VisualPropertiesAsset:
        family_dir.mkdir(parents=True, exist_ok=True)
        colors_in = intelligence.colors
        res = self.visual_properties_processor.process(image, colors_in)

        asset = VisualPropertiesAsset(
            dominant_colors=res["dominant_colors"],
            palette_extended=res["palette_extended"],
            gradients_detected=res["gradients_detected"],
            lighting_direction=res["lighting_direction"],
            shadow_regions=res["shadow_regions"],
            highlight_regions=res["highlight_regions"],
            blur_map_summary=res["blur_map_summary"],
            focus_bbox=res["focus_bbox"],
        )

        self.asset_writer.write_json_sidecar(
            {"dominant_colors": res["dominant_colors"], "palette_extended": res["palette_extended"]},
            family_dir / "colors.json",
        )
        self.asset_writer.write_json_sidecar(
            {"gradients_detected": res["gradients_detected"]}, family_dir / "gradients.json"
        )
        self.asset_writer.write_json_sidecar(
            {
                "lighting_direction": res["lighting_direction"],
                "blur_map_summary": res["blur_map_summary"],
            },
            family_dir / "lighting.json",
        )

        return asset

    def _extract_composition(
        self, image: np.ndarray, intelligence: ThumbnailIntelligence, family_dir: Path
    ) -> CompositionAsset:
        family_dir.mkdir(parents=True, exist_ok=True)
        comp_in = intelligence.composition
        res = self.composition_processor.process(image, comp_in)

        eye_ref: Optional[AssetFileRef] = None
        neg_ref: Optional[AssetFileRef] = None
        hier_ref: Optional[AssetFileRef] = None

        if res.get("eye_flow_map") is not None:
            p = family_dir / "eye_flow.png"
            self.asset_writer.write_image(res["eye_flow_map"], p)
            eye_ref = self._create_file_ref("eye_flow_map", p, res["eye_flow_map"], source="derived")

        if res.get("negative_space_mask") is not None:
            p = family_dir / "negative_space_mask.png"
            self.asset_writer.write_image(res["negative_space_mask"], p)
            neg_ref = self._create_file_ref(
                "negative_space_mask", p, res["negative_space_mask"], source="derived"
            )

        if res.get("visual_hierarchy_overlay") is not None:
            p = family_dir / "visual_hierarchy_overlay.png"
            self.asset_writer.write_image(res["visual_hierarchy_overlay"], p)
            hier_ref = self._create_file_ref(
                "visual_hierarchy_overlay", p, res["visual_hierarchy_overlay"], source="derived"
            )

        asset = CompositionAsset(
            eye_flow_map=eye_ref,
            negative_space_mask=neg_ref,
            visual_hierarchy_overlay=hier_ref,
            source_composition_analysis=comp_in,
        )

        self.asset_writer.write_json_sidecar(
            comp_in.model_dump(), family_dir / "composition.json"
        )
        return asset

    def _extract_objects(
        self, image: np.ndarray, intelligence: ThumbnailIntelligence, family_dir: Path
    ) -> list[ObjectAsset]:
        family_dir.mkdir(parents=True, exist_ok=True)
        objs = intelligence.objects if intelligence.objects else []
        results = self.object_processor.process(image, objs)

        assets: list[ObjectAsset] = []
        masks_sidecar: list[dict[str, Any]] = []

        for item in results:
            idx = item["object_index"]
            crop_path = family_dir / f"object_{idx+1:02d}.png"
            mask_path = family_dir / f"object_{idx+1:02d}_mask.png"

            self.asset_writer.write_image(item["crop"], crop_path)
            self.asset_writer.write_image(item["mask"], mask_path)

            crop_ref = self._create_file_ref(
                "object_crop", crop_path, item["crop"], confidence=item["confidence"], source="extracted"
            )
            mask_ref = self._create_file_ref(
                "object_mask", mask_path, item["mask"], confidence=item["confidence"], source="extracted"
            )

            asset = ObjectAsset(
                object_index=idx,
                label=item["label"],
                crop=crop_ref,
                mask=mask_ref,
                bbox=item["bbox"],
                confidence=item["confidence"],
                parent_object_index=item["parent_object_index"],
                child_object_indices=item["child_object_indices"],
                source_detected_object_index=item["source_detected_object_index"],
            )
            assets.append(asset)
            masks_sidecar.append(
                {
                    "object_index": idx,
                    "label": item["label"],
                    "parent_object_index": item["parent_object_index"],
                    "child_object_indices": item["child_object_indices"],
                }
            )

        self.asset_writer.write_json_sidecar(
            masks_sidecar, family_dir / "object_masks.json"
        )
        return assets

    def _extract_people(
        self, image: np.ndarray, intelligence: ThumbnailIntelligence, family_dir: Path
    ) -> list[PersonAsset]:
        family_dir.mkdir(parents=True, exist_ok=True)
        faces_in = intelligence.faces if intelligence.faces else None
        results = self.person_processor.process(image, faces_in)

        assets: list[PersonAsset] = []

        for item in results:
            idx = item["person_index"]
            face_path = family_dir / f"face_{idx+1:02d}.png"
            mask_path = family_dir / f"face_{idx+1:02d}_mask.png"

            self.asset_writer.write_image(item["face"], face_path)
            self.asset_writer.write_image(item["face_mask"], mask_path)

            face_ref = self._create_file_ref("face_crop", face_path, item["face"], source="extracted")
            mask_ref = self._create_file_ref("face_mask", mask_path, item["face_mask"], source="extracted")

            body_ref: Optional[AssetFileRef] = None
            if item.get("body_mask") is not None:
                body_path = family_dir / f"body_mask_{idx+1:02d}.png"
                self.asset_writer.write_image(item["body_mask"], body_path)
                body_ref = self._create_file_ref("body_mask", body_path, item["body_mask"], source="extracted")

            hair_ref: Optional[AssetFileRef] = None
            if item.get("hair_mask") is not None:
                hair_path = family_dir / f"hair_mask_{idx+1:02d}.png"
                self.asset_writer.write_image(item["hair_mask"], hair_path)
                hair_ref = self._create_file_ref("hair_mask", hair_path, item["hair_mask"], source="extracted")

            clothing_ref: Optional[AssetFileRef] = None
            if item.get("clothing_mask") is not None:
                cloth_path = family_dir / f"clothing_mask_{idx+1:02d}.png"
                self.asset_writer.write_image(item["clothing_mask"], cloth_path)
                clothing_ref = self._create_file_ref(
                    "clothing_mask", cloth_path, item["clothing_mask"], source="extracted"
                )

            asset = PersonAsset(
                person_index=idx,
                face=face_ref,
                face_mask=mask_ref,
                face_embedding=item.get("face_embedding"),
                facial_landmarks=item.get("facial_landmarks"),
                body_mask=body_ref,
                pose_keypoints=item.get("pose_keypoints"),
                clothing_mask=clothing_ref,
                hair_mask=hair_ref,
                source_face_detail_index=item["source_face_detail_index"],
                extraction_status=item["extraction_status"],
                extraction_notes=item["extraction_notes"],
            )
            assets.append(asset)

            if item.get("facial_landmarks"):
                self.asset_writer.write_json_sidecar(
                    {"landmarks": item["facial_landmarks"]}, family_dir / "landmarks.json"
                )

        return assets

    def _extract_scene(self, image: np.ndarray, family_dir: Path) -> SceneAsset:
        family_dir.mkdir(parents=True, exist_ok=True)
        res = self.scene_processor.process(image)

        fg_ref = self._save_scene_ref("foreground", family_dir / "foreground.png", res.get("foreground"))
        bg_ref = self._save_scene_ref("background", family_dir / "background.png", res.get("background"))
        depth_ref = self._save_scene_ref("depth_map", family_dir / "depth.png", res.get("depth_map"))
        seg_ref = self._save_scene_ref(
            "segmentation_map", family_dir / "segmentation.png", res.get("segmentation_map")
        )
        sky_ref = self._save_scene_ref("sky_mask", family_dir / "sky_mask.png", res.get("sky_mask"))
        ground_ref = self._save_scene_ref(
            "ground_mask", family_dir / "ground_mask.png", res.get("ground_mask")
        )

        return SceneAsset(
            background=bg_ref,
            foreground=fg_ref,
            depth_map=depth_ref,
            segmentation_map=seg_ref,
            sky_mask=sky_ref,
            ground_mask=ground_ref,
            extraction_status="success",
            extraction_notes=[],
        )

    def _save_scene_ref(
        self, asset_type: str, path: Path, arr: Optional[np.ndarray]
    ) -> Optional[AssetFileRef]:
        if arr is None or arr.size == 0:
            return None
        self.asset_writer.write_image(arr, path)
        return self._create_file_ref(asset_type, path, arr, source="extracted")

    def _extract_effects(self, image: np.ndarray, family_dir: Path) -> EffectsAsset:
        family_dir.mkdir(parents=True, exist_ok=True)
        res = self.effects_processor.process(image)

        asset = EffectsAsset(
            glow_detected=res["glow_detected"],
            outline_detected=res["outline_detected"],
            drop_shadow_detected=res["drop_shadow_detected"],
            motion_blur_detected=res["motion_blur_detected"],
            particles_detected=res["particles_detected"],
            confidence=res["confidence"],
            notes=res["notes"],
        )

        self.asset_writer.write_json_sidecar(res, family_dir / "effects.json")
        return asset

    def _create_file_ref(
        self,
        asset_type: str,
        file_path: Path,
        array: np.ndarray,
        confidence: Optional[float] = None,
        source: str = "extracted",
    ) -> AssetFileRef:
        """Create a validated AssetFileRef instance."""
        data = file_path.read_bytes() if file_path.exists() else array.tobytes()
        checksum = hashlib.sha256(data).hexdigest()
        h, w = array.shape[:2]
        return AssetFileRef(
            asset_type=asset_type,
            file_path=str(file_path),
            checksum=checksum,
            resolution=(w, h),
            confidence_score=confidence,
            source=source,
        )

    @staticmethod
    def _downscale_if_needed(image: np.ndarray) -> np.ndarray:
        """Downscale image if maximum dimension exceeds pixel limit."""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX:
            return image

        scale = ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX / float(max_dim)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _validate_video_id(video_id: str) -> str:
        if not video_id or not video_id.strip():
            raise ValueError("video_id must not be empty")
        return video_id.strip()


def extract_assets(
    video_id: str,
    source_image_path: str,
    intelligence: ThumbnailIntelligence,
    *,
    storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
    options: Optional[dict] = None,
) -> AssetExtractionManifest:
    """Extract every asset family for one thumbnail. Cache-aware, resumable."""
    engine = AssetExtractionEngine(storage_root=storage_root)
    return engine.extract(video_id, source_image_path, intelligence, options=options)


def save_asset_manifest(
    manifest: AssetExtractionManifest,
    *,
    storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
) -> Path:
    """Atomically persist a manifest."""
    builder = ManifestBuilder()
    target_dir = Path(storage_root) / manifest.video_id
    path = target_dir / ASSET_MANIFEST_FILENAME
    builder.serialize_to_disk(manifest, path)
    return path


def load_asset_manifest(
    video_id: str,
    *,
    storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
) -> AssetExtractionManifest:
    """Load and validate a persisted manifest for a video_id."""
    path = Path(storage_root) / video_id / ASSET_MANIFEST_FILENAME
    if not path.exists():
        raise ManifestNotFoundError(f"No asset manifest found for video_id {video_id} at {path}")

    try:
        content = path.read_text(encoding="utf-8")
        manifest = AssetExtractionManifest.model_validate_json(content)
        return manifest
    except Exception as exc:
        raise ManifestValidationError(
            f"Could not load or validate asset manifest at {path}: {exc}"
        ) from exc

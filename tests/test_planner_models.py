"""
test_planner_models.py
======================

Unit tests for Module 10.5 data models and exceptions (Phase 1).
"""

import pytest
from pydantic import ValidationError

from models import (
    BackgroundStrategy,
    BoundingBox,
    FaceStrategy,
    GenerationPlan,
    HeadlineSource,
    PlanConditioningAsset,
)
from thumbnail_planner_exceptions import (
    PlanCacheError,
    PlanPersistError,
    PlanValidationError,
    ThumbnailPlannerError,
    UpstreamArtifactMissingError,
)

VALID_HASH_1 = "a" * 64
VALID_HASH_2 = "b" * 64


def test_planner_enums():
    assert HeadlineSource.PRESERVED_OCR == "preserved_ocr"
    assert HeadlineSource.NONE == "none"
    assert HeadlineSource.GENERATED == "generated"

    assert FaceStrategy.NONE == "none"
    assert FaceStrategy.PRESERVE_AS_IS_IDENTITY_LOCKED == "preserve_as_is_identity_locked"

    assert BackgroundStrategy.STRUCTURE_GUIDED_REPLACE == "structure_guided_replace"


def test_plan_conditioning_asset_valid():
    asset = PlanConditioningAsset(
        role="person_0",
        asset_id="creator_face_0",
        path="data/visual_references/vid123/face.png",
        kind="reference_image",
        source_module="module8",
    )
    assert asset.role == "person_0"
    assert asset.kind == "reference_image"


def test_plan_conditioning_asset_empty_field_raises():
    with pytest.raises(ValidationError):
        PlanConditioningAsset(
            role="",
            asset_id="creator_face_0",
            path="data/face.png",
            kind="reference_image",
            source_module="module8",
        )


def test_generation_plan_valid():
    plan = GenerationPlan(
        video_id="vid123",
        headline="BEST TIPS",
        headline_source=HeadlineSource.PRESERVED_OCR,
        headline_placement_zone=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50),
        face_strategy=FaceStrategy.ENHANCE_EXISTING_IDENTITY_LOCKED,
        background_strategy=BackgroundStrategy.STRUCTURE_GUIDED_REPLACE,
        preserve_objects=["object_0_mic"],
        composition_strategy="rule_of_thirds",
        camera_distance="medium",
        lighting="warm",
        color_palette=["#ff0000", "#00ff00"],
        negative_constraints=["no blurry", "no watermark"],
        conditioning_assets=[
            PlanConditioningAsset(
                role="background",
                asset_id="bg_0",
                path="data/bg.png",
                kind="reference_image",
                source_module="vre",
            )
        ],
        decision_manifest_hash=VALID_HASH_1,
        asset_extraction_manifest_hash=None,
        prompt_package_hash=VALID_HASH_1,
        workspace_hash=VALID_HASH_2,
        status="success",
        engine_version="1.0.0",
        generated_at="2026-08-01T00:00:00Z",
    )
    assert plan.video_id == "vid123"
    assert plan.prompt_package_hash == VALID_HASH_1
    assert plan.decision_manifest_hash == VALID_HASH_1
    assert plan.asset_extraction_manifest_hash is None


def test_generation_plan_invalid_hash_raises():
    with pytest.raises(ValidationError):
        GenerationPlan(
            video_id="vid123",
            headline="",
            headline_source=HeadlineSource.NONE,
            face_strategy=FaceStrategy.NONE,
            background_strategy=BackgroundStrategy.UNGUIDED_REPLACE,
            composition_strategy="centered",
            camera_distance="close",
            lighting="neutral",
            prompt_package_hash="invalid_short_hash",
            workspace_hash=VALID_HASH_2,
            engine_version="1.0.0",
            generated_at="2026-08-01T00:00:00Z",
        )


def test_generation_plan_empty_video_id_raises():
    with pytest.raises(ValidationError):
        GenerationPlan(
            video_id="   ",
            headline="",
            headline_source=HeadlineSource.NONE,
            face_strategy=FaceStrategy.NONE,
            background_strategy=BackgroundStrategy.UNGUIDED_REPLACE,
            composition_strategy="centered",
            camera_distance="close",
            lighting="neutral",
            prompt_package_hash=VALID_HASH_1,
            workspace_hash=VALID_HASH_2,
            engine_version="1.0.0",
            generated_at="2026-08-01T00:00:00Z",
        )


def test_exceptions_hierarchy():
    err = UpstreamArtifactMissingError("missing file")
    assert isinstance(err, ThumbnailPlannerError)

    val_err = PlanValidationError("invalid plan")
    assert isinstance(val_err, ThumbnailPlannerError)

    cache_err = PlanCacheError("cache failure")
    assert isinstance(cache_err, ThumbnailPlannerError)

    persist_err = PlanPersistError("persist failure")
    assert isinstance(persist_err, ThumbnailPlannerError)

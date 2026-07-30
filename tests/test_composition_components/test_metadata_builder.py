"""
test_metadata_builder.py
========================

Unit tests for MetadataBuilder component in Module 10 Asset Composer.
"""

from __future__ import annotations

import pytest

from composition_components.metadata_builder import MetadataBuilder
from models import (
    AssetPlacement,
    BoundingBox,
    CanvasTransform,
    ColorDirection,
    CompositionLayer,
    GenerationParameters,
    LayoutDirection,
    LayerDecision,
    LayerRole,
    LayerTransform,
    ModelSettings,
    PromptPackage,
    QualityParameters,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    VisualReferenceManifest,
)


@pytest.fixture
def manifest():
    return VisualReferenceManifest(
        video_id="test_vid",
        source_image_path="thumb.jpg",
        source_hash="a" * 64,
        created_at="2026-07-30T00:00:00Z",
        assets={},
    )


@pytest.fixture
def spec():
    return RedesignSpecification(
        video_id="test_vid",
        source_thumbnail_path="thumb.jpg",
        color_direction=ColorDirection(),
        subject_treatment=SubjectTreatment(has_subject=True),
        text_overlay=TextOverlaySpec(),
        layout_direction=LayoutDirection(),
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.6,
        source_content_mismatch_detected=False,
        generated_at="2026-07-30T00:00:00Z",
    )


@pytest.fixture
def package():
    return PromptPackage(
        video_id="test_vid",
        positive_prompt="prompt",
        negative_prompt="neg",
        subject_instructions="sub",
        background_instructions="bg",
        typography_instructions="text",
        composition_instructions="comp",
        lighting_instructions="light",
        color_instructions="color",
        generation_parameters=GenerationParameters(),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        generated_at="2026-07-30T00:00:00Z",
    )


def test_metadata_builder_build_and_statistics(manifest, spec, package):
    builder = MetadataBuilder()
    metadata = builder.build("test_vid", manifest, spec, package)

    assert metadata.video_id == "test_vid"
    assert metadata.vre_source_hash == "a" * 64
    assert len(metadata.redesign_spec_hash) == 64
    assert len(metadata.prompt_package_hash) == 64

    # Statistics
    l1 = CompositionLayer(
        layer_id="l1",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    l2 = CompositionLayer(
        layer_id="l2",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            transform=LayerTransform(),
            z_index=10,
        ),
    )

    stats = builder.statistics([l1, l2])
    assert stats.total_layers == 2
    assert stats.replaced == 1
    assert stats.kept == 1
    assert stats.removed == 0

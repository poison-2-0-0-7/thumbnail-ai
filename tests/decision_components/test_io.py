"""
test_io.py
=======

Unit tests for decision_components/io.py (Phase 1 Ingestion and I/O).
"""

from pathlib import Path
import json
import pytest

from modules.decision_components.io import (
    DecisionCache,
    load_cached_decision_manifest,
    load_input_bundle,
    save_decision_manifest,
)
from modules.decision_exceptions import ArtifactValidationError, MissingArtifactError
from modules.models import (
    BoundingBox,
    ColorProfile,
    CompositionAnalysis,
    DecisionAction,
    DecisionManifest,
    DecisionManifestStatus,
    DecisionSource,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    GenerationParameters,
    OCRResult,
    PromptPackage,
    RedesignSpecification,
    ResolvedDecision,
    TargetElement,
    TextOverlaySpec,
    TextRegion,
    ThumbnailIntelligence,
)


@pytest.fixture
def sample_artifacts(tmp_path: Path):
    video_id = "v_test_999"

    # Setup directories
    analysis_dir = tmp_path / "analysis"
    spec_dir = tmp_path / "specs"
    prompt_dir = tmp_path / "prompts"
    asset_dir = tmp_path / "assets"

    analysis_dir.mkdir()
    spec_dir.mkdir()
    prompt_dir.mkdir()
    asset_dir.mkdir()

    # Module 4 Intelligence
    intel = ThumbnailIntelligence(
        video_id=video_id,
        thumbnail_path="thumb.jpg",
        ocr=OCRResult(
            text_regions=[
                TextRegion(
                    text="HOT TOPIC",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.3),
                )
            ]
        ),
        faces=FaceAnalysis(
            face_count=1,
            faces=[
                FaceDetail(
                    bbox=BoundingBox(x_min=0.3, y_min=0.2, x_max=0.7, y_max=0.8),
                    detection_confidence=0.92,
                    is_largest=True,
                )
            ],
            has_face=True,
        ),
        objects=[
            DetectedObject(
                label="car",
                confidence=0.85,
                bbox=BoundingBox(x_min=0.4, y_min=0.4, x_max=0.9, y_max=0.9),
            )
        ],
        colors=ColorProfile(
            dominant_colors=["#ff0000"],
            palette_hex=["#ff0000"],
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
        ),
        composition=CompositionAnalysis(),
        status="success",
        analyzed_at="2026-07-30T00:00:00Z",
    )
    (analysis_dir / f"{video_id}.json").write_text(intel.model_dump_json(), encoding="utf-8")

    # Module 5 Redesign Spec
    from modules.models import ColorDirection, LayoutDirection, SubjectTreatment
    spec = RedesignSpecification(
        video_id=video_id,
        source_thumbnail_path="thumb.jpg",
        elements_to_preserve=["creator face"],
        text_overlay=TextOverlaySpec(include_text=True, target_text="HOT TOPIC"),
        color_direction=ColorDirection(),
        subject_treatment=SubjectTreatment(),
        layout_direction=LayoutDirection(),
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.6,
        source_content_mismatch_detected=False,
        status="success",
        generated_at="2026-07-30T00:00:00Z",
    )
    (spec_dir / f"{video_id}.json").write_text(spec.model_dump_json(), encoding="utf-8")

    # Module 6 Prompt Package
    from modules.models import ModelSettings, QualityParameters
    prompt_pkg = PromptPackage(
        video_id=video_id,
        positive_prompt="A high contrast thumbnail",
        negative_prompt="blurry",
        subject_instructions="keep creator face",
        background_instructions="dark blue gradient",
        typography_instructions="HOT TOPIC text",
        composition_instructions="rule of thirds",
        lighting_instructions="dramatic backlight",
        color_instructions="warm color palette",
        generation_parameters=GenerationParameters(seed=12345),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        status="success",
        generated_at="2026-07-30T00:00:00Z",
    )
    (prompt_dir / f"{video_id}.json").write_text(prompt_pkg.model_dump_json(), encoding="utf-8")

    return {
        "video_id": video_id,
        "analysis_dir": analysis_dir,
        "spec_dir": spec_dir,
        "prompt_dir": prompt_dir,
        "asset_dir": asset_dir,
    }


def test_load_input_bundle_success(sample_artifacts):
    bundle = load_input_bundle(
        video_id=sample_artifacts["video_id"],
        analysis_dir=sample_artifacts["analysis_dir"],
        redesign_spec_dir=sample_artifacts["spec_dir"],
        prompt_package_dir=sample_artifacts["prompt_dir"],
        asset_extraction_dir=sample_artifacts["asset_dir"],
    )

    assert bundle.video_id == sample_artifacts["video_id"]
    assert bundle.intelligence.ocr.text_regions[0].text == "HOT TOPIC"
    assert bundle.redesign_spec.elements_to_preserve == ["creator face"]
    assert bundle.asset_extraction is None  # Missing M8 degraded gracefully
    assert len(bundle.cross_reference_index) >= 3  # M4 obj, text, face indexed


def test_load_input_bundle_missing_m4(sample_artifacts):
    with pytest.raises(MissingArtifactError):
        load_input_bundle(
            video_id="v_non_existent",
            analysis_dir=sample_artifacts["analysis_dir"],
            redesign_spec_dir=sample_artifacts["spec_dir"],
            prompt_package_dir=sample_artifacts["prompt_dir"],
            asset_extraction_dir=sample_artifacts["asset_dir"],
        )


def test_save_and_load_manifest(tmp_path: Path):
    video_id = "v_save_test"
    target_dir = tmp_path / "decisions"

    resolved = ResolvedDecision(
        decision_id="d1",
        target=TargetElement(element_id="m4_face_0", element_type="face", label="creator face"),
        action=DecisionAction.KEEP,
        confidence=0.95,
        source=DecisionSource.RULE,
        rationale="Preserve face",
        priority_rank=1,
    )

    manifest = DecisionManifest(
        video_id=video_id,
        source_generated_image_path="gen.png",
        source_generated_image_hash="a" * 64,
        decisions=[resolved],
        keep_count=1,
        overall_confidence=0.95,
        status=DecisionManifestStatus.SUCCESS,
        decided_at="2026-07-30T00:00:00Z",
    )

    manifest_path = save_decision_manifest(manifest, target_dir=target_dir)
    assert manifest_path.exists()

    # Check 5 per-action files exist
    video_dir = target_dir / video_id
    assert (video_dir / "keep.json").exists()
    assert (video_dir / "remove.json").exists()
    assert (video_dir / "replace.json").exists()
    assert (video_dir / "enhance.json").exists()
    assert (video_dir / "add.json").exists()

    loaded = load_cached_decision_manifest(video_id, decision_dir=target_dir)
    assert loaded is not None
    assert loaded.video_id == video_id
    assert loaded.keep_count == 1

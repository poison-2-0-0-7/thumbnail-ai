"""
test_decision_engine.py
========================

End-to-end integration unit tests for DecisionEngine (Module 9).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from modules.decision_engine import (
    DecisionEngine,
    load_cached_decision_manifest,
    run_decision_engine,
    run_decision_engine_batch,
)
from modules.models import (
    BoundingBox,
    ColorDirection,
    ColorProfile,
    CompositionAnalysis,
    DecisionManifestStatus,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    GenerationParameters,
    LayoutDirection,
    ModelSettings,
    OCRResult,
    PromptPackage,
    QualityParameters,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    TextRegion,
    ThumbnailIntelligence,
)


@pytest.fixture
def mock_upstream_artifacts(tmp_path: Path):
    video_id = "v_engine_test_1"

    analysis_dir = tmp_path / "analysis"
    spec_dir = tmp_path / "redesign_specs"
    prompt_dir = tmp_path / "prompt_packages"
    decision_dir = tmp_path / "decisions"

    analysis_dir.mkdir(parents=True)
    spec_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)
    decision_dir.mkdir(parents=True)

    intel = ThumbnailIntelligence(
        video_id=video_id,
        thumbnail_path="data/thumb.jpg",
        ocr=OCRResult(
            text_regions=[
                TextRegion(
                    text="OLD TEXT",
                    confidence=0.85,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.3, y_max=0.2),
                )
            ]
        ),
        faces=FaceAnalysis(
            face_count=1,
            faces=[
                FaceDetail(
                    bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
                    detection_confidence=0.95,
                    is_largest=True,
                )
            ],
            has_face=True,
        ),
        objects=[
            DetectedObject(
                label="car",
                confidence=0.90,
                bbox=BoundingBox(x_min=0.5, y_min=0.5, x_max=0.8, y_max=0.8),
            )
        ],
        colors=ColorProfile(
            dominant_colors=["#ff0000"],
            palette_hex=["#ff0000"],
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
        ),
        composition=CompositionAnalysis(clutter_score=0.7),
        status="success",
        analyzed_at="2026-07-30T00:00:00Z",
    )
    (analysis_dir / f"{video_id}.json").write_text(intel.model_dump_json(), encoding="utf-8")

    spec = RedesignSpecification(
        video_id=video_id,
        source_thumbnail_path="data/thumb.jpg",
        elements_to_preserve=["creator face"],
        text_overlay=TextOverlaySpec(include_text=True),
        color_direction=ColorDirection(warm_or_cool="warm"),
        subject_treatment=SubjectTreatment(),
        layout_direction=LayoutDirection(),
        source_ctr_potential_score=0.4,
        source_curiosity_gap_score=0.5,
        source_content_mismatch_detected=False,
        status="success",
        generated_at="2026-07-30T00:00:00Z",
    )
    (spec_dir / f"{video_id}.json").write_text(spec.model_dump_json(), encoding="utf-8")

    prompt_pkg = PromptPackage(
        video_id=video_id,
        positive_prompt="High CTR thumbnail",
        negative_prompt="blurry",
        subject_instructions="creator face",
        background_instructions="warm background",
        typography_instructions="text",
        composition_instructions="comp",
        lighting_instructions="dramatic",
        color_instructions="warm",
        generation_parameters=GenerationParameters(),
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
        "decision_dir": decision_dir,
    }


@pytest.fixture
def mock_llm_reasoner():
    reasoner = MagicMock()
    reasoner.adjudicate.side_effect = lambda cands, bundle: cands
    return reasoner


def test_run_decision_engine_end_to_end(mock_upstream_artifacts, mock_llm_reasoner):
    video_id = mock_upstream_artifacts["video_id"]
    decision_dir = mock_upstream_artifacts["decision_dir"]

    engine = DecisionEngine(
        decision_dir=decision_dir,
        analysis_dir=mock_upstream_artifacts["analysis_dir"],
        redesign_spec_dir=mock_upstream_artifacts["spec_dir"],
        prompt_package_dir=mock_upstream_artifacts["prompt_dir"],
        llm_reasoner=mock_llm_reasoner,
    )
    manifest = engine.run(video_id)

    assert manifest.video_id == video_id
    assert manifest.status in [DecisionManifestStatus.SUCCESS, DecisionManifestStatus.PARTIAL]
    assert len(manifest.decisions) > 0

    # Check disk persistence
    v_dir = decision_dir / video_id
    assert (v_dir / "decision_manifest.json").exists()
    assert (v_dir / "keep.json").exists()
    assert (v_dir / "remove.json").exists()
    assert (v_dir / "replace.json").exists()
    assert (v_dir / "enhance.json").exists()
    assert (v_dir / "add.json").exists()


def test_run_decision_engine_caching(mock_upstream_artifacts, mock_llm_reasoner):
    video_id = mock_upstream_artifacts["video_id"]
    decision_dir = mock_upstream_artifacts["decision_dir"]

    engine = DecisionEngine(
        decision_dir=decision_dir,
        analysis_dir=mock_upstream_artifacts["analysis_dir"],
        redesign_spec_dir=mock_upstream_artifacts["spec_dir"],
        prompt_package_dir=mock_upstream_artifacts["prompt_dir"],
        llm_reasoner=mock_llm_reasoner,
    )
    m1 = engine.run(video_id)
    m2 = engine.run(video_id)

    assert m2.decided_at == m1.decided_at


def test_run_decision_engine_batch(mock_upstream_artifacts, mock_llm_reasoner):
    video_id = mock_upstream_artifacts["video_id"]
    decision_dir = mock_upstream_artifacts["decision_dir"]

    engine = DecisionEngine(
        decision_dir=decision_dir,
        analysis_dir=mock_upstream_artifacts["analysis_dir"],
        redesign_spec_dir=mock_upstream_artifacts["spec_dir"],
        prompt_package_dir=mock_upstream_artifacts["prompt_dir"],
        llm_reasoner=mock_llm_reasoner,
    )
    results = [
        engine.run(video_id),
        engine.run("v_missing_99"),
    ]
    assert len(results) == 2
    assert results[0].video_id == video_id
    assert results[0].status != DecisionManifestStatus.ERROR
    assert results[1].video_id == "v_missing_99"
    assert results[1].status == DecisionManifestStatus.ERROR

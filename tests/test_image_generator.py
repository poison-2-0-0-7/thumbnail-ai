from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from PIL import Image, ImageDraw

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import MODULE7_GENERATION_PROFILES  # noqa: E402
from image_generator import (  # noqa: E402
    ArtifactWriter, BackgroundCompositor, CandidateRanker,
    FaceRestorationStage, IdentityPreservationStage, ImageGeneratorPipeline,
    MetricsCollector, ProfileSelector, PromptPackageInvalidError,
    PromptPackageLoader, QualityAssuranceStage, ReferenceAssetResolver,
    ReferenceAssets, UpscaleStage, WorkflowBuilder, cosine_similarity,
    generation_hash, prompt_package_hash, run_image_generation_pipeline,
    _calculate_text_safe_zone_score, _calculate_object_preservation_score,
    _calculate_color_compliance_score, _calculate_composition_score,
)
from models import (  # noqa: E402
    CandidateScore, FaceMatchResult, GeneratedAsset, GenerationMetrics,
    ImageGenerationResult, PromptPackage, QualityAssuranceReport,
)
from module7_exceptions import NoEligibleCandidateError  # noqa: E402
from workflow_library import WorkflowLibrary  # noqa: E402


VIDEO_ID = "abcdEFGH123"


def _package() -> PromptPackage:
    return PromptPackage(
        video_id=VIDEO_ID, positive_prompt="positive", negative_prompt="negative",
        subject_instructions="subject", background_instructions="background",
        typography_instructions="typography", composition_instructions="composition",
        lighting_instructions="lighting", color_instructions="colour",
        generation_parameters={"seed": 123}, quality_parameters={}, model_settings={},
        generated_at="2026-01-01T00:00:00+00:00",
    )


def _create_test_image(path: Path, width: int = 1280, height: int = 720, color: str = "blue") -> Path:
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill="red")
    draw.ellipse([100, 100, 200, 200], fill="yellow")
    img.save(path)
    return path


def test_prompt_package_loader_validates_status_and_identity(tmp_path: Path) -> None:
    (tmp_path / f"{VIDEO_ID}.json").write_text(_package().model_dump_json(), encoding="utf-8")
    assert PromptPackageLoader(tmp_path).load(VIDEO_ID) == _package()
    invalid = _package().model_copy(update={"status": "error", "error_message": "bad upstream"})
    (tmp_path / f"{VIDEO_ID}.json").write_text(invalid.model_dump_json(), encoding="utf-8")
    with pytest.raises(PromptPackageInvalidError):
        PromptPackageLoader(tmp_path).load(VIDEO_ID)


def test_reference_resolver_finds_thumbnail_and_optional_analysis(tmp_path: Path) -> None:
    thumbnails, analysis = tmp_path / "thumbnails", tmp_path / "analysis"
    thumbnails.mkdir()
    analysis.mkdir()
    (thumbnails / f"{VIDEO_ID}.jpg").write_bytes(b"reference")
    (analysis / f"{VIDEO_ID}.json").write_text("{}", encoding="utf-8")
    resolved = ReferenceAssetResolver(thumbnails, analysis).resolve(_package())
    assert resolved.source_thumbnail_path.suffix == ".jpg"
    assert resolved.analysis_path is not None


@pytest.mark.parametrize(("vram", "expected"), [(8.4, "PROFILE_PREMIUM"), (8.0, "PROFILE_STANDARD"), (7.5, "PROFILE_FAST"), (5.5, "PROFILE_LOW_VRAM")])
def test_profile_selector_uses_documented_fallback_ladder(vram: float, expected: str) -> None:
    assert ProfileSelector().select(vram).name == expected


def test_explicit_profile_is_honored_only_when_it_fits() -> None:
    selector = ProfileSelector()
    assert selector.select(8.4, "PROFILE_FAST").name == "PROFILE_FAST"
    assert selector.select(6.0, "PROFILE_STANDARD").name == "PROFILE_LOW_VRAM"


def test_workflow_builder_is_pure_and_hashes_resolved_graph() -> None:
    package = _package()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    library = WorkflowLibrary()
    ref = library.resolve("gaming", profile)
    first = WorkflowBuilder().build(package, profile, ref, library=library)
    second = WorkflowBuilder().build(package, profile, ref, library=library)
    assert first.graph == second.graph
    assert first.workflow_hash == second.workflow_hash
    assert first.graph["5"]["inputs"]["seed"] == 123
    assert first.graph["1"]["inputs"]["ckpt_name"] == profile.checkpoint


def test_general_workflow_uses_comfyui_safe_output_prefix() -> None:
    package = _package()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    library = WorkflowLibrary()
    ref = library.resolve("unknown-niche", profile)

    workflow = WorkflowBuilder().build(package, profile, ref, library=library)

    assert workflow.graph["7"]["inputs"]["filename_prefix"] == f"module7_{VIDEO_ID}"
    assert "/" not in workflow.graph["7"]["inputs"]["filename_prefix"]
    assert "\\" not in workflow.graph["7"]["inputs"]["filename_prefix"]


def test_output_filename_prefix_sanitizes_path_characters() -> None:
    package = _package().model_copy(update={"video_id": "../bad\\id"})

    assert WorkflowBuilder._output_filename_prefix(package) == "module7_bad_id"


def test_hashes_are_stable_and_manifest_and_metrics_are_persisted(tmp_path: Path) -> None:
    package = _package()
    package_digest = prompt_package_hash(package)
    assert package_digest == prompt_package_hash(package)
    digest = generation_hash("workflow", package_digest, None, [], [], 123, "PROFILE_STANDARD")
    result = ImageGenerationResult(video_id=VIDEO_ID, workflow_version="workflow_v1",
                                   prompt_package_hash=package_digest, generation_hash=digest,
                                   generated_at="2026-01-01T00:00:00+00:00")
    manifest = ArtifactWriter(tmp_path / "output").write_manifest(result)
    assert json.loads(manifest.read_text(encoding="utf-8"))["generation_hash"] == digest
    metrics_path = tmp_path / "logs" / "metrics.jsonl"
    metrics = GenerationMetrics(video_id=VIDEO_ID, niche="gaming", workflow_version="workflow_v1",
                                recorded_at="2026-01-01T00:00:00+00:00")
    MetricsCollector(metrics_path).append(metrics)
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["video_id"] == VIDEO_ID


def test_cosine_similarity_basic() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_identity_preservation_stage_skipped_when_no_reference_face(tmp_path: Path) -> None:
    ref_img = _create_test_image(tmp_path / "ref.jpg")
    gen_img = _create_test_image(tmp_path / "gen.png")
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps({"face_analysis": {"has_face": False, "face_count": 0}}), encoding="utf-8")
    ref_assets = ReferenceAssets(source_thumbnail_path=ref_img, analysis_path=analysis_path)

    stage = IdentityPreservationStage(threshold=0.5)
    result = stage.verify(gen_img, ref_assets)

    assert result.skipped is True
    assert result.passed is True


def test_identity_preservation_stage_calculates_similarity(tmp_path: Path) -> None:
    ref_img = _create_test_image(tmp_path / "ref.jpg")
    gen_img = _create_test_image(tmp_path / "gen.png")
    ref_assets = ReferenceAssets(source_thumbnail_path=ref_img)

    stage = IdentityPreservationStage(threshold=0.5)
    result = stage.verify(gen_img, ref_assets)

    assert result.face_detected is True
    assert result.skipped is False
    assert result.similarity >= 0.0


def test_face_restoration_stage_applies_enhancement(tmp_path: Path) -> None:
    gen_img = _create_test_image(tmp_path / "gen.png")
    out_img = tmp_path / "restored.png"
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]

    stage = FaceRestorationStage()
    result = stage.restore(gen_img, profile, output_path=out_img)

    assert result.is_file()
    assert result.stat().st_size > 0


def test_upscale_stage_resizes_to_exact_target(tmp_path: Path) -> None:
    small_img = _create_test_image(tmp_path / "small.png", width=640, height=360)
    out_img = tmp_path / "upscaled.png"
    profile = MODULE7_GENERATION_PROFILES["PROFILE_FAST"]

    stage = UpscaleStage()
    result = stage.upscale(small_img, profile, target_width=1280, target_height=720, output_path=out_img)

    with Image.open(result) as img:
        assert img.size == (1280, 720)


def test_quality_assurance_stage_evaluates_scores_and_hard_gates(tmp_path: Path) -> None:
    gen_img = _create_test_image(tmp_path / "gen.png", width=1280, height=720)
    package = _package()
    face_match = FaceMatchResult(similarity=0.85, threshold=0.5, passed=True, face_detected=True)

    stage = QualityAssuranceStage()
    report = stage.evaluate(gen_img, package, face_match)

    assert report.resolution_passed is True
    assert report.file_integrity_passed is True
    assert report.safety_passed is True
    assert report.hard_gate_passed is True
    assert report.overall_score > 0.0


def test_candidate_ranker_orders_by_overall_score_and_tie_breaks(tmp_path: Path) -> None:
    p1 = tmp_path / "c1.png"
    p2 = tmp_path / "c2.png"

    qa1 = QualityAssuranceReport(
        resolution_passed=True, file_integrity_passed=True, safety_passed=True,
        overall_score=0.80, hard_gate_passed=True
    )
    qa2 = QualityAssuranceReport(
        resolution_passed=True, file_integrity_passed=True, safety_passed=True,
        overall_score=0.92, hard_gate_passed=True
    )

    fm1 = FaceMatchResult(similarity=0.70, threshold=0.5, passed=True)
    fm2 = FaceMatchResult(similarity=0.88, threshold=0.5, passed=True)

    candidates = [(0, p1, qa1, fm1), (1, p2, qa2, fm2)]

    ranker = CandidateRanker()
    winner, scores = ranker.rank(candidates)

    assert winner[0] == 1
    assert len(scores) == 2
    assert scores[1].selected is True
    assert scores[0].selected is False


def test_candidate_ranker_raises_no_eligible_candidate_error_when_all_fail(tmp_path: Path) -> None:
    p1 = tmp_path / "c1.png"
    qa1 = QualityAssuranceReport(
        resolution_passed=False, file_integrity_passed=True, safety_passed=True,
        overall_score=0.50, hard_gate_passed=False
    )
    fm1 = FaceMatchResult(similarity=0.30, threshold=0.5, passed=False)

    candidates = [(0, p1, qa1, fm1)]

    ranker = CandidateRanker()
    with pytest.raises(NoEligibleCandidateError):
        ranker.rank(candidates)


def test_image_generator_pipeline_runs_end_to_end_with_mock_client(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "prompt_packages"
    thumb_dir = tmp_path / "thumbnails"
    analysis_dir = tmp_path / "analysis"
    out_dir = tmp_path / "generated_thumbnails"
    pkg_dir.mkdir()
    thumb_dir.mkdir()
    analysis_dir.mkdir()

    pkg = _package()
    (pkg_dir / f"{VIDEO_ID}.json").write_text(pkg.model_dump_json(), encoding="utf-8")
    ref_img = _create_test_image(thumb_dir / f"{VIDEO_ID}.jpg")

    mock_png_bytes = ref_img.read_bytes()

    class MockClient:
        def generate(self, built_wf, video_id, num_candidates_requested=1, **kwargs):
            from comfyui_client import _OutputResult
            return _OutputResult(
                prompt_id="mock-prompt-id", output_node_id="7", filename="output.png",
                subfolder="", image_type="output", format="png", content=mock_png_bytes,
                width=1280, height=720,
            )

    pipeline = ImageGeneratorPipeline(
        client=MockClient(),
        package_loader=PromptPackageLoader(pkg_dir),
        asset_resolver=ReferenceAssetResolver(thumb_dir, analysis_dir),
        artifact_writer=ArtifactWriter(out_dir),
    )

    result = pipeline.run(VIDEO_ID, niche="gaming", prompt_package=pkg)

    assert result.status == "success"
    assert result.generated_asset is not None
    assert Path(result.generated_asset.path).is_file()
    assert (out_dir / VIDEO_ID / f"{VIDEO_ID}_manifest.json").is_file()


def test_golden_regression_workflow_hash_unchanged_when_conditioning_none(tmp_path: Path):
    """Verify building workflows with conditioning=None produces deterministic hashes for all shipped templates."""
    builder = WorkflowBuilder()
    library = WorkflowLibrary()
    pkg = _package()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    ref_img = _create_test_image(tmp_path / "thumb.jpg")
    ref_assets = ReferenceAssets(source_thumbnail_path=ref_img)

    for niche in ("general", "gaming", "finance", "education", "podcast", "tech", "lifestyle", "vlog", "fitness", "reaction", "documentary"):
        wf_ref = library.resolve(niche, profile)
        built_wf_1 = builder.build(pkg, profile, wf_ref, reference_assets=ref_assets, library=library, conditioning=None)
        built_wf_2 = builder.build(pkg, profile, wf_ref, reference_assets=ref_assets, library=library)

        assert built_wf_1.workflow_hash == built_wf_2.workflow_hash
        assert isinstance(built_wf_1.graph, dict)
        assert len(built_wf_1.graph) > 0


def test_select_fragments_multi_object(tmp_path: Path):
    from generation_components.conditioning_asset_resolver import GenerationConditioningContext
    builder = WorkflowBuilder()
    profile = ProfileSelector().select(8.0, "PROFILE_STANDARD")
    f_obj1 = tmp_path / "mic.png"
    f_obj2 = tmp_path / "laptop.png"

    ctx = GenerationConditioningContext(
        role_image_paths={
            "object_0_mic": f_obj1,
            "object_1_laptop": f_obj2,
        }
    )

    fragments = builder._select_fragments(profile, ctx)
    assert "multi_object_reference" in fragments


def test_multi_candidate_generation_with_strategy_pack(tmp_path: Path, monkeypatch):
    from models import DesignBlueprint, TextPlacement, CandidateManifest, GenerationRunMetadata

    pkg_dir = tmp_path / "prompt_packages"
    thumb_dir = tmp_path / "thumbnails"
    analysis_dir = tmp_path / "analysis"
    out_dir = tmp_path / "generated_thumbnails"
    pkg_dir.mkdir()
    thumb_dir.mkdir()
    analysis_dir.mkdir()

    monkeypatch.setattr("modules.image_generator.MODULE7_MAX_CANDIDATES", 3)
    monkeypatch.setattr("modules.image_generator.MODULE7_STRATEGY_PACK", "default_five")

    from models import GenerationParameters
    pkg = _package().model_copy(
        update={
            "generation_parameters": GenerationParameters(
                seed=123,
                num_candidates=3,
                strategy_pack="default_five",
            )
        }
    )

    (pkg_dir / f"{VIDEO_ID}.json").write_text(pkg.model_dump_json(), encoding="utf-8")
    ref_img = _create_test_image(thumb_dir / f"{VIDEO_ID}.jpg")
    mock_png_bytes = ref_img.read_bytes()

    class MockClient:
        def generate(self, built_wf, video_id, num_candidates_requested=1, **kwargs):
            from comfyui_client import _OutputResult
            return _OutputResult(
                prompt_id="mock-prompt-id", output_node_id="7", filename="output.png",
                subfolder="", image_type="output", format="png", content=mock_png_bytes,
                width=1280, height=720,
            )

    blueprint = DesignBlueprint(
        video_id=VIDEO_ID,
        headline="TEST HEADLINE",
        headline_score=0.95,
        hook_type="curiosity",
        emotion="excited",
        face_strategy="smile",
        background_strategy="blur",
        text_position=TextPlacement(include_text=True),
        camera_distance="medium",
        lighting="bright",
        duration_seconds=0.1,
        generated_at="2026-08-01T00:00:00Z",
    )

    pipeline = ImageGeneratorPipeline(
        client=MockClient(),
        package_loader=PromptPackageLoader(pkg_dir),
        asset_resolver=ReferenceAssetResolver(thumb_dir, analysis_dir),
        artifact_writer=ArtifactWriter(out_dir),
    )

    result = pipeline.run(
        VIDEO_ID,
        niche="gaming",
        prompt_package=pkg,
        design_blueprint=blueprint,
    )

    assert result.status == "success"
    assert result.generated_asset is not None

    cand_manifest_path = out_dir / VIDEO_ID / "candidate_manifest.json"
    gen_meta_path = out_dir / VIDEO_ID / "generation_metadata.json"

    assert cand_manifest_path.is_file()
    assert gen_meta_path.is_file()

    cand_manifest = CandidateManifest.model_validate_json(cand_manifest_path.read_text(encoding="utf-8"))
    assert len(cand_manifest.entries) == 3
    assert cand_manifest.entries[0].strategy_name == "faithful"
    assert cand_manifest.entries[1].strategy_name == "higher_emotion"
    assert cand_manifest.entries[2].strategy_name == "cleaner_composition"

    # Verify seeds differ sequentially
    seeds = [e.seed for e in cand_manifest.entries]
    assert len(set(seeds)) == 3

    gen_meta = GenerationRunMetadata.model_validate_json(gen_meta_path.read_text(encoding="utf-8"))
    assert gen_meta.num_candidates_requested == 3
    assert gen_meta.num_candidates_completed == 3


def test_text_safe_zone_score_evaluates_clutter(tmp_path: Path) -> None:
    pkg = _package()
    clean_img = _create_test_image(tmp_path / "clean.png", color="blue")
    score_clean = _calculate_text_safe_zone_score(clean_img, pkg)
    assert 0.0 <= score_clean <= 1.0
    assert score_clean > 0.8

    # Create image with heavy noise/clutter in bottom-right safe zone
    cluttered_path = tmp_path / "cluttered.png"
    img = Image.new("RGB", (1280, 720), color="blue")
    draw = ImageDraw.Draw(img)
    for x in range(1080, 1280, 10):
        for y in range(576, 720, 10):
            draw.point((x, y), fill="white" if (x + y) % 20 == 0 else "black")
    img.save(cluttered_path)

    score_cluttered = _calculate_text_safe_zone_score(cluttered_path, pkg)
    assert 0.0 <= score_cluttered <= 1.0
    assert score_cluttered < score_clean


def test_object_preservation_score_evaluates_objects(tmp_path: Path) -> None:
    pkg = _package()
    src_img = _create_test_image(tmp_path / "src.jpg", color="green")
    cand_img = _create_test_image(tmp_path / "cand.png", color="green")
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps({"objects": [{"label": "cup", "confidence": 0.9}]}), encoding="utf-8")

    ref_assets = ReferenceAssets(source_thumbnail_path=src_img, analysis_path=analysis_path)
    score = _calculate_object_preservation_score(cand_img, pkg, ref_assets)
    assert 0.0 <= score <= 1.0


def test_color_compliance_score_calculates_lab_distance(tmp_path: Path) -> None:
    pkg = _package().model_copy(update={"color_instructions": "vibrant blue"})
    blue_src = _create_test_image(tmp_path / "blue_src.jpg", color="blue")
    blue_cand = _create_test_image(tmp_path / "blue_cand.png", color="blue")
    red_cand = _create_test_image(tmp_path / "red_cand.png", color="red")

    ref_assets = ReferenceAssets(source_thumbnail_path=blue_src)

    score_match = _calculate_color_compliance_score(blue_cand, pkg, ref_assets)
    score_mismatch = _calculate_color_compliance_score(red_cand, pkg, ref_assets)

    assert 0.0 <= score_match <= 1.0
    assert 0.0 <= score_mismatch <= 1.0
    assert score_match > score_mismatch


def test_composition_score_evaluates_grid_balance(tmp_path: Path) -> None:
    pkg = _package()
    src_img = _create_test_image(tmp_path / "src.jpg", color="navy")
    cand_img = _create_test_image(tmp_path / "cand.png", color="navy")
    ref_assets = ReferenceAssets(source_thumbnail_path=src_img)

    score = _calculate_composition_score(cand_img, pkg, ref_assets)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_backward_compatibility_legacy_txt2img_default(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "prompt_packages"
    thumb_dir = tmp_path / "thumbnails"
    analysis_dir = tmp_path / "analysis"
    out_dir = tmp_path / "generated_thumbnails"
    pkg_dir.mkdir()
    thumb_dir.mkdir()
    analysis_dir.mkdir()

    pkg = _package()
    (pkg_dir / f"{VIDEO_ID}.json").write_text(pkg.model_dump_json(), encoding="utf-8")
    ref_img = _create_test_image(thumb_dir / f"{VIDEO_ID}.jpg")
    mock_png_bytes = ref_img.read_bytes()

    class MockClient:
        def generate(self, built_wf, video_id, num_candidates_requested=1, **kwargs):
            from comfyui_client import _OutputResult
            return _OutputResult(
                prompt_id="mock-prompt-id", output_node_id="7", filename="output.png",
                subfolder="", image_type="output", format="png", content=mock_png_bytes,
                width=1280, height=720,
            )

    pipeline = ImageGeneratorPipeline(
        client=MockClient(),
        package_loader=PromptPackageLoader(pkg_dir),
        asset_resolver=ReferenceAssetResolver(thumb_dir, analysis_dir),
        artifact_writer=ArtifactWriter(out_dir),
    )

    result_default = pipeline.run(VIDEO_ID, niche="gaming", prompt_package=pkg)
    result_legacy = pipeline.run(VIDEO_ID, niche="gaming", prompt_package=pkg, edit_mode="legacy_txt2img")

    assert result_default.status == "success"
    assert result_legacy.status == "success"
    assert result_default.workflow_hash == result_legacy.workflow_hash






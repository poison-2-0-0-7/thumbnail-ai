from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import MODULE7_GENERATION_PROFILES  # noqa: E402
from generation_components.conditioning_asset_resolver import GenerationConditioningContext  # noqa: E402
from image_generator import (  # noqa: E402
    ArtifactWriter, ImageGeneratorPipeline, PromptPackageLoader,
    ReferenceAssetResolver, ReferenceAssets, VRAMExhaustedError,
)
from models import (  # noqa: E402
    CandidateStrategy, DesignBlueprint,
    PromptPackage, QualityAssuranceReport, FaceMatchResult, WorkflowTemplateRef,
)
from workflow_library import WorkflowLibrary  # noqa: E402


def _package(video_id: str = "test_vid_123") -> PromptPackage:
    return PromptPackage(
        video_id=video_id,
        positive_prompt="positive",
        negative_prompt="negative",
        subject_instructions="subject",
        background_instructions="background",
        typography_instructions="typography",
        composition_instructions="composition",
        lighting_instructions="lighting",
        color_instructions="colour",
        generation_parameters={"seed": 123},
        quality_parameters={},
        model_settings={},
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_process_single_candidate_forwards_effective_edit_mode_to_workflow_library(tmp_path: Path) -> None:
    """_process_single_candidate must forward effective_edit_mode to WorkflowLibrary.resolve."""
    mock_library = MagicMock(spec=WorkflowLibrary)
    dummy_ref = WorkflowTemplateRef(
        niche="general",
        profile_name="PROFILE_STANDARD_EDIT",
        template_path=str(tmp_path / "general_edit.json"),
        workflow_version="2.0.0",
        template_name="General Edit Workflow",
    )
    mock_library.resolve.return_value = dummy_ref

    mock_builder = MagicMock()
    mock_builder.build.return_value = MagicMock(graph={"1": {}}, workflow_hash="abc")

    mock_client = MagicMock()
    mock_client.generate.return_value = MagicMock(content=b"png_data")

    mock_identity = MagicMock()
    mock_identity.verify.return_value = FaceMatchResult(similarity=0.9, threshold=0.6, passed=True, skipped=False)

    mock_restoration = MagicMock()
    mock_restoration.restore.side_effect = lambda src, prof, output_path: output_path.write_bytes(b"restored")

    mock_compositor = MagicMock()
    mock_compositor.composite.side_effect = lambda src, ref, pkg, output_path: output_path.write_bytes(b"comp")

    mock_upscale = MagicMock()
    mock_upscale.upscale.side_effect = lambda src, prof, *args, **kwargs: kwargs.get("output_path", args[0] if args else src).write_bytes(b"upscaled")

    mock_qa = MagicMock()
    mock_qa.evaluate.return_value = QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, hard_gate_passed=True, overall_score=0.9)

    pipeline = ImageGeneratorPipeline(
        client=mock_client,
        workflow_library=mock_library,
        workflow_builder=mock_builder,
        identity_stage=mock_identity,
        restoration_stage=mock_restoration,
        background_compositor=mock_compositor,
        upscale_stage=mock_upscale,
        qa_stage=mock_qa,
        artifact_writer=ArtifactWriter(tmp_path / "out"),
    )

    pkg = _package()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"]
    references = ReferenceAssets(source_thumbnail_path=tmp_path / "source.jpg")
    (tmp_path / "source.jpg").write_bytes(b"source")
    conditioning_ctx = GenerationConditioningContext()
    wf_cache = MagicMock()

    cand_work_dir = tmp_path / "work"
    cand_work_dir.mkdir(parents=True, exist_ok=True)

    # Call _process_single_candidate with effective_edit_mode="staged_edit"
    pipeline._process_single_candidate(
        cand_idx=0,
        strategy=CandidateStrategy.faithful_default(),
        package=pkg,
        design_blueprint=None,
        profile=profile,
        niche="gaming",
        video_id="test_vid_123",
        num_candidates=1,
        references=references,
        conditioning_ctx=conditioning_ctx,
        generation_plan=None,
        client_obj=mock_client,
        cand_work_dir=cand_work_dir,
        wf_cache=wf_cache,
        effective_edit_mode="staged_edit",
    )

    # Assert WorkflowLibrary.resolve was called with edit_mode="staged_edit"
    mock_library.resolve.assert_called_once_with("gaming", profile, edit_mode="staged_edit")


def test_process_single_candidate_fallback_forwards_effective_edit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """VRAMExhaustedError retry path in _process_single_candidate must also forward effective_edit_mode."""
    # Monkeypatch MODULE7_PROFILE_PREFERENCE to bypass Phase 1 startup check in ProfileSelector
    monkeypatch.setattr("config.MODULE7_PROFILE_PREFERENCE", ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM"))
    monkeypatch.setattr("image_generator.MODULE7_PROFILE_PREFERENCE", ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM"))

    mock_library = MagicMock(spec=WorkflowLibrary)
    dummy_ref = WorkflowTemplateRef(
        niche="general",
        profile_name="PROFILE_STANDARD_EDIT",
        template_path=str(tmp_path / "general_edit.json"),
        workflow_version="2.0.0",
        template_name="General Edit Workflow",
    )
    mock_library.resolve.return_value = dummy_ref

    mock_builder = MagicMock()
    mock_builder.build.return_value = MagicMock(graph={"1": {}}, workflow_hash="abc")

    mock_client = MagicMock()
    # First generate call raises VRAMExhaustedError, second succeeds
    mock_client.generate.side_effect = [VRAMExhaustedError("OOM"), MagicMock(content=b"png_data")]

    mock_identity = MagicMock()
    mock_identity.verify.return_value = FaceMatchResult(similarity=0.9, threshold=0.6, passed=True, skipped=False)

    mock_restoration = MagicMock()
    mock_restoration.restore.side_effect = lambda src, prof, output_path: output_path.write_bytes(b"restored")

    mock_compositor = MagicMock()
    mock_compositor.composite.side_effect = lambda src, ref, pkg, output_path: output_path.write_bytes(b"comp")

    mock_upscale = MagicMock()
    mock_upscale.upscale.side_effect = lambda src, prof, *args, **kwargs: kwargs.get("output_path", args[0] if args else src).write_bytes(b"upscaled")

    mock_qa = MagicMock()
    mock_qa.evaluate.return_value = QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, hard_gate_passed=True, overall_score=0.9)

    pipeline = ImageGeneratorPipeline(
        client=mock_client,
        workflow_library=mock_library,
        workflow_builder=mock_builder,
        identity_stage=mock_identity,
        restoration_stage=mock_restoration,
        background_compositor=mock_compositor,
        upscale_stage=mock_upscale,
        qa_stage=mock_qa,
        artifact_writer=ArtifactWriter(tmp_path / "out"),
    )

    pkg = _package()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"]
    references = ReferenceAssets(source_thumbnail_path=tmp_path / "source.jpg")
    (tmp_path / "source.jpg").write_bytes(b"source")
    conditioning_ctx = GenerationConditioningContext()
    wf_cache = MagicMock()

    cand_work_dir = tmp_path / "work"
    cand_work_dir.mkdir(parents=True, exist_ok=True)

    pipeline._process_single_candidate(
        cand_idx=0,
        strategy=CandidateStrategy.faithful_default(),
        package=pkg,
        design_blueprint=None,
        profile=profile,
        niche="tech",
        video_id="test_vid_123",
        num_candidates=1,
        references=references,
        conditioning_ctx=conditioning_ctx,
        generation_plan=None,
        client_obj=mock_client,
        cand_work_dir=cand_work_dir,
        wf_cache=wf_cache,
        effective_edit_mode="staged_edit",
    )

    # Assert WorkflowLibrary.resolve was called twice, both with edit_mode="staged_edit"
    assert mock_library.resolve.call_count == 2
    for call in mock_library.resolve.call_args_list:
        assert call.kwargs.get("edit_mode") == "staged_edit"


def test_process_single_candidate_default_parameter_backward_compatibility(tmp_path: Path) -> None:
    """_process_single_candidate without effective_edit_mode defaults to legacy_txt2img."""
    mock_library = MagicMock(spec=WorkflowLibrary)
    dummy_ref = WorkflowTemplateRef(
        niche="general",
        profile_name="PROFILE_STANDARD",
        template_path=str(tmp_path / "general.json"),
        workflow_version="2.0.0",
        template_name="General Legacy Workflow",
    )
    mock_library.resolve.return_value = dummy_ref

    mock_builder = MagicMock()
    mock_builder.build.return_value = MagicMock(graph={"1": {}}, workflow_hash="abc")

    mock_client = MagicMock()
    mock_client.generate.return_value = MagicMock(content=b"png_data")

    mock_identity = MagicMock()
    mock_identity.verify.return_value = FaceMatchResult(similarity=0.9, threshold=0.6, passed=True, skipped=False)

    mock_restoration = MagicMock()
    mock_restoration.restore.side_effect = lambda src, prof, output_path: output_path.write_bytes(b"restored")

    mock_compositor = MagicMock()
    mock_compositor.composite.side_effect = lambda src, ref, pkg, output_path: output_path.write_bytes(b"comp")

    mock_upscale = MagicMock()
    mock_upscale.upscale.side_effect = lambda src, prof, *args, **kwargs: kwargs.get("output_path", args[0] if args else src).write_bytes(b"upscaled")

    mock_qa = MagicMock()
    mock_qa.evaluate.return_value = QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, hard_gate_passed=True, overall_score=0.9)

    pipeline = ImageGeneratorPipeline(
        client=mock_client,
        workflow_library=mock_library,
        workflow_builder=mock_builder,
        identity_stage=mock_identity,
        restoration_stage=mock_restoration,
        background_compositor=mock_compositor,
        upscale_stage=mock_upscale,
        qa_stage=mock_qa,
        artifact_writer=ArtifactWriter(tmp_path / "out"),
    )

    pkg = _package()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    references = ReferenceAssets(source_thumbnail_path=tmp_path / "source.jpg")
    (tmp_path / "source.jpg").write_bytes(b"source")
    conditioning_ctx = GenerationConditioningContext()
    wf_cache = MagicMock()

    cand_work_dir = tmp_path / "work"
    cand_work_dir.mkdir(parents=True, exist_ok=True)

    # Call without effective_edit_mode argument
    pipeline._process_single_candidate(
        cand_idx=0,
        strategy=CandidateStrategy.faithful_default(),
        package=pkg,
        design_blueprint=None,
        profile=profile,
        niche="gaming",
        video_id="test_vid_123",
        num_candidates=1,
        references=references,
        conditioning_ctx=conditioning_ctx,
        generation_plan=None,
        client_obj=mock_client,
        cand_work_dir=cand_work_dir,
        wf_cache=wf_cache,
    )

    mock_library.resolve.assert_called_once_with("gaming", profile, edit_mode="legacy_txt2img")


def test_pipeline_run_threads_explicit_staged_edit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ImageGeneratorPipeline.run with edit_mode='staged_edit' forwards staged_edit to WorkflowLibrary.resolve."""
    monkeypatch.setattr("config.MODULE7_PROFILE_PREFERENCE", ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM"))
    monkeypatch.setattr("image_generator.MODULE7_PROFILE_PREFERENCE", ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM"))

    pkg_dir = tmp_path / "pkgs"
    thumb_dir = tmp_path / "thumbs"
    analysis_dir = tmp_path / "analysis"
    out_dir = tmp_path / "out"
    pkg_dir.mkdir()
    thumb_dir.mkdir()
    analysis_dir.mkdir()
    out_dir.mkdir()

    pkg = _package("vid_staged_1")
    (pkg_dir / "vid_staged_1.json").write_text(pkg.model_dump_json(), encoding="utf-8")
    (thumb_dir / "vid_staged_1.jpg").write_bytes(b"thumb")

    mock_library = MagicMock(spec=WorkflowLibrary)
    dummy_ref = WorkflowTemplateRef(
        niche="gaming",
        profile_name="PROFILE_STANDARD_EDIT",
        template_path=str(tmp_path / "gaming_edit.json"),
        workflow_version="2.0.0",
        template_name="Gaming Edit Workflow",
    )
    mock_library.resolve.return_value = dummy_ref

    mock_builder = MagicMock()
    mock_builder.build.return_value = MagicMock(
        graph={"1": {}},
        workflow_hash="abc",
        template_name="Gaming Edit Workflow",
        workflow_version="2.0.0",
        workflow_template="gaming_edit.json",
    )

    mock_client = MagicMock()
    mock_client.generate.return_value = MagicMock(content=b"png_data")

    mock_identity = MagicMock()
    mock_identity.verify.return_value = FaceMatchResult(similarity=0.9, threshold=0.6, passed=True, skipped=False)

    def _write_valid_png(src: Path, prof: Any, *args: Any, **kwargs: Any) -> None:
        from PIL import Image
        out = kwargs.get("output_path", args[0] if args else src)
        img = Image.new("RGB", (1280, 720), color="blue")
        img.save(out, format="PNG")

    mock_restoration = MagicMock()
    mock_restoration.restore.side_effect = lambda src, prof, output_path: _write_valid_png(src, prof, output_path=output_path)

    mock_compositor = MagicMock()
    mock_compositor.composite.side_effect = lambda src, ref, pkg, output_path: _write_valid_png(src, ref, output_path=output_path)

    mock_upscale = MagicMock()
    mock_upscale.upscale.side_effect = _write_valid_png

    mock_qa = MagicMock()
    mock_qa.evaluate.return_value = QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, hard_gate_passed=True, overall_score=0.9)

    pipeline = ImageGeneratorPipeline(
        client=mock_client,
        workflow_library=mock_library,
        workflow_builder=mock_builder,
        package_loader=PromptPackageLoader(pkg_dir),
        asset_resolver=ReferenceAssetResolver(thumb_dir, analysis_dir),
        identity_stage=mock_identity,
        restoration_stage=mock_restoration,
        background_compositor=mock_compositor,
        upscale_stage=mock_upscale,
        qa_stage=mock_qa,
        artifact_writer=ArtifactWriter(out_dir),
    )

    # Force explicit profile override for testing
    monkeypatch.setattr("image_generator.MODULE7_PROFILE", "PROFILE_STANDARD_EDIT")

    result = pipeline.run("vid_staged_1", niche="gaming", available_vram_gb=8.0, edit_mode="staged_edit")

    assert result.status == "success"
    mock_library.resolve.assert_called_with("gaming", MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"], edit_mode="staged_edit")

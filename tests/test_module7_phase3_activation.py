from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
from PIL import Image

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import MODULE7_GENERATION_PROFILES, MODULE7_PROFILE_PREFERENCE  # noqa: E402
from generation_components.conditioning_asset_resolver import GenerationConditioningContext  # noqa: E402
from image_generator import (  # noqa: E402
    ArtifactWriter, ImageGeneratorPipeline, ProfileSelector,
    PromptPackageLoader, ReferenceAssetResolver, ReferenceAssets,
)
from models import (  # noqa: E402
    CandidateStrategy, FaceMatchResult, PromptPackage, QualityAssuranceReport, WorkflowTemplateRef,
)
from workflow_library import WorkflowLibrary  # noqa: E402


def _package(video_id: str = "test_vid_phase3") -> PromptPackage:
    return PromptPackage(
        video_id=video_id,
        positive_prompt="positive prompt",
        negative_prompt="negative prompt",
        subject_instructions="subject",
        background_instructions="background",
        typography_instructions="typography",
        composition_instructions="composition",
        lighting_instructions="lighting",
        color_instructions="colour",
        generation_parameters={"seed": 456},
        quality_parameters={},
        model_settings={},
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_auto_profile_selection_selects_profile_standard_edit() -> None:
    """When available_vram_gb >= 8.0 GB (usable 7.5) and requested_profile='auto', ProfileSelector must return PROFILE_STANDARD_EDIT."""
    selector = ProfileSelector()
    profile = selector.select(available_vram_gb=8.0, requested_profile="auto")
    assert profile.name == "PROFILE_STANDARD_EDIT"
    assert profile.edit_mode_default == "staged_edit"


def test_auto_profile_selection_preference_order() -> None:
    """Verify preference tuple order has PROFILE_STANDARD_EDIT placed immediately after PROFILE_PREMIUM."""
    assert MODULE7_PROFILE_PREFERENCE[0] == "PROFILE_PREMIUM"
    assert MODULE7_PROFILE_PREFERENCE[1] == "PROFILE_STANDARD_EDIT"
    assert MODULE7_PROFILE_PREFERENCE[2] == "PROFILE_STANDARD"


def test_pipeline_run_auto_mode_activates_staged_edit(tmp_path: Path) -> None:
    """ImageGeneratorPipeline.run with edit_mode='auto' and available_vram_gb=7.7 selects PROFILE_STANDARD_EDIT and edit_mode='staged_edit'."""
    pkg_dir = tmp_path / "pkgs"
    thumb_dir = tmp_path / "thumbs"
    analysis_dir = tmp_path / "analysis"
    out_dir = tmp_path / "out"
    pkg_dir.mkdir()
    thumb_dir.mkdir()
    analysis_dir.mkdir()
    out_dir.mkdir()

    vid = "vid_auto_activation"
    pkg = _package(vid)
    (pkg_dir / f"{vid}.json").write_text(pkg.model_dump_json(), encoding="utf-8")
    (thumb_dir / f"{vid}.jpg").write_bytes(b"thumb")

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
        workflow_hash="hash123",
        template_name="Gaming Edit Workflow",
        workflow_version="2.0.0",
        workflow_template="gaming_edit.json",
    )

    mock_client = MagicMock()
    mock_client.generate.return_value = MagicMock(content=b"png")

    mock_identity = MagicMock()
    mock_identity.verify.return_value = FaceMatchResult(similarity=0.95, threshold=0.6, passed=True, skipped=False)

    def _write_valid_png(src: Path, prof: Any, *args: Any, **kwargs: Any) -> None:
        out = kwargs.get("output_path", args[0] if args else src)
        img = Image.new("RGB", (1280, 720), color="purple")
        img.save(out, format="PNG")

    mock_restoration = MagicMock()
    mock_restoration.restore.side_effect = lambda src, prof, output_path: _write_valid_png(src, prof, output_path=output_path)

    mock_compositor = MagicMock()
    mock_compositor.composite.side_effect = lambda src, ref, pkg, output_path: _write_valid_png(src, ref, output_path=output_path)

    mock_upscale = MagicMock()
    mock_upscale.upscale.side_effect = _write_valid_png

    mock_qa = MagicMock()
    mock_qa.evaluate.return_value = QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, hard_gate_passed=True, overall_score=0.92)

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

    result = pipeline.run(vid, niche="gaming", available_vram_gb=8.0, edit_mode="auto")

    assert result.status == "success"
    # WorkflowLibrary.resolve MUST be called with edit_mode="staged_edit" because PROFILE_STANDARD_EDIT was auto-selected
    mock_library.resolve.assert_called_with(
        "gaming",
        MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"],
        edit_mode="staged_edit",
    )


def test_legacy_fallback_preserved_when_vram_is_lower(tmp_path: Path) -> None:
    """When available_vram_gb is 7.6 GB (usable 7.1), ProfileSelector selects PROFILE_FAST (edit_mode_default=None) and effective_edit_mode is legacy_txt2img."""
    selector = ProfileSelector()
    profile = selector.select(available_vram_gb=7.6, requested_profile="auto")
    assert profile.name == "PROFILE_FAST"
    assert profile.edit_mode_default is None

    pkg_dir = tmp_path / "pkgs"
    thumb_dir = tmp_path / "thumbs"
    analysis_dir = tmp_path / "analysis"
    out_dir = tmp_path / "out"
    pkg_dir.mkdir()
    thumb_dir.mkdir()
    analysis_dir.mkdir()
    out_dir.mkdir()

    vid = "vid_fallback_legacy"
    pkg = _package(vid)
    (pkg_dir / f"{vid}.json").write_text(pkg.model_dump_json(), encoding="utf-8")
    (thumb_dir / f"{vid}.jpg").write_bytes(b"thumb")

    mock_library = MagicMock(spec=WorkflowLibrary)
    dummy_ref = WorkflowTemplateRef(
        niche="tech",
        profile_name="PROFILE_FAST",
        template_path=str(tmp_path / "tech.json"),
        workflow_version="2.0.0",
        template_name="Tech Legacy Workflow",
    )
    mock_library.resolve.return_value = dummy_ref

    mock_builder = MagicMock()
    mock_builder.build.return_value = MagicMock(
        graph={"1": {}},
        workflow_hash="hash_legacy",
        template_name="Tech Legacy Workflow",
        workflow_version="2.0.0",
        workflow_template="tech.json",
    )

    mock_client = MagicMock()
    mock_client.generate.return_value = MagicMock(content=b"png")

    mock_identity = MagicMock()
    mock_identity.verify.return_value = FaceMatchResult(similarity=0.9, threshold=0.6, passed=True, skipped=False)

    def _write_valid_png(src: Path, prof: Any, *args: Any, **kwargs: Any) -> None:
        out = kwargs.get("output_path", args[0] if args else src)
        img = Image.new("RGB", (1280, 720), color="green")
        img.save(out, format="PNG")

    mock_restoration = MagicMock()
    mock_restoration.restore.side_effect = lambda src, prof, output_path: _write_valid_png(src, prof, output_path=output_path)

    mock_compositor = MagicMock()
    mock_compositor.composite.side_effect = lambda src, ref, pkg, output_path: _write_valid_png(src, ref, output_path=output_path)

    mock_upscale = MagicMock()
    mock_upscale.upscale.side_effect = _write_valid_png

    mock_qa = MagicMock()
    mock_qa.evaluate.return_value = QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, hard_gate_passed=True, overall_score=0.88)

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

    result = pipeline.run(vid, niche="tech", available_vram_gb=7.6, edit_mode="auto")

    assert result.status == "success"
    # WorkflowLibrary.resolve MUST be called with edit_mode="legacy_txt2img" because PROFILE_FAST was selected
    mock_library.resolve.assert_called_with(
        "tech",
        MODULE7_GENERATION_PROFILES["PROFILE_FAST"],
        edit_mode="legacy_txt2img",
    )

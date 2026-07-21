from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import MODULE7_GENERATION_PROFILES  # noqa: E402
from image_generator import (  # noqa: E402
    ArtifactWriter, MetricsCollector, ProfileSelector, PromptPackageInvalidError,
    PromptPackageLoader, ReferenceAssetResolver, WorkflowBuilder, generation_hash,
    prompt_package_hash,
)
from models import GenerationMetrics, ImageGenerationResult, PromptPackage  # noqa: E402
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
    assert first.graph["4"]["inputs"]["seed"] == 123
    assert first.graph["1"]["inputs"]["ckpt_name"] == profile.checkpoint


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

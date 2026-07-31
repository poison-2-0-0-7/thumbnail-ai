from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODULES_DIR = _PROJECT_ROOT / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

import main  # noqa: E402
from models import PromptPackage, ThumbnailData, VideoMetadata  # noqa: E402
from module7_exceptions import Module7Error  # noqa: E402


VIDEO_ID = "abcdEFGH123"


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id=VIDEO_ID,
        title="A test video",
        uploader="Creator",
        uploader_id="@creator",
        channel_id="UC123",
        thumbnail_url="https://example.com/thumb.jpg",
        categories=["Gaming"],
    )


def _prompt_package() -> PromptPackage:
    return PromptPackage(
        video_id=VIDEO_ID,
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


def test_run_pipeline_invokes_module7_after_prompt_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    order: list[str] = []
    creator = SimpleNamespace(email="creator@example.com", video_url="https://youtu.be/abcdEFGH123")

    monkeypatch.setattr(main, "load_all_creators", lambda csv_path: [creator])
    monkeypatch.setattr(main, "process_video", lambda creator, enable_oembed_fallback: _metadata())
    monkeypatch.setattr(
        main,
        "process_thumbnail",
        lambda metadata, thumbnail_dir: ThumbnailData(metadata=metadata, thumbnail_path=str(tmp_path / "thumb.jpg")),
    )
    monkeypatch.setattr(main, "analyze_thumbnail", lambda thumbnail: SimpleNamespace(status="success"))
    monkeypatch.setattr(main, "save_intelligence", lambda intelligence, analysis_dir: None)
    monkeypatch.setattr(main, "build_redesign_specification", lambda intelligence: SimpleNamespace())
    monkeypatch.setattr(main, "save_redesign_spec", lambda redesign_spec, spec_dir: None)
    monkeypatch.setattr(
        main,
        "build_design_blueprint",
        lambda intelligence, spec, metadata: SimpleNamespace(headline="Test Headline", text_position=None),
    )
    monkeypatch.setattr(main, "save_design_blueprint", lambda blueprint, blueprint_dir: order.append("module5_5_saved"))
    monkeypatch.setattr(main, "compile_prompt_package", lambda redesign_spec, design_blueprint=None: _prompt_package())

    def save_prompt_package(prompt_package: PromptPackage, package_dir: Path) -> None:
        order.append("module6_saved")

    class FakeAssetComposer:
        def prepare_generation_workspace(self, video_id: str, options: dict | None = None):
            order.append("module10_prepared")
            return SimpleNamespace(video_id=video_id)

    def run_module7_generation(
        prompt_package: PromptPackage,
        *,
        metadata: VideoMetadata,
        thumbnail_dir: Path,
        analysis_dir: Path,
        generation_bundle: object = None,
        design_blueprint: object = None,
        **kwargs: object,
    ) -> Path:
        order.append("module7_generated")
        assert prompt_package.video_id == VIDEO_ID
        assert metadata.video_id == VIDEO_ID
        assert thumbnail_dir == tmp_path / "thumbnails"
        assert analysis_dir == tmp_path / "analysis"
        assert generation_bundle is not None
        return tmp_path / "generated" / f"{VIDEO_ID}.png"

    monkeypatch.setattr(main, "save_prompt_package", save_prompt_package)
    monkeypatch.setattr(main, "AssetComposer", FakeAssetComposer)
    monkeypatch.setattr(main, "_run_module7_generation", run_module7_generation)

    main.run_pipeline(
        csv_path=tmp_path / "creators.csv",
        thumbnail_dir=tmp_path / "thumbnails",
        analysis_dir=tmp_path / "analysis",
        redesign_spec_dir=tmp_path / "specs",
        design_blueprint_dir=tmp_path / "blueprints",
        prompt_package_dir=tmp_path / "packages",
    )

    assert order == ["module5_5_saved", "module6_saved", "module10_prepared", "module7_generated"]


def test_run_pipeline_treats_module7_error_as_per_creator_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    creator = SimpleNamespace(email="creator@example.com", video_url="https://youtu.be/abcdEFGH123")
    monkeypatch.setattr(main, "load_all_creators", lambda csv_path: [creator])
    monkeypatch.setattr(main, "process_video", lambda creator, enable_oembed_fallback: _metadata())
    monkeypatch.setattr(
        main,
        "process_thumbnail",
        lambda metadata, thumbnail_dir: ThumbnailData(metadata=metadata, thumbnail_path=str(tmp_path / "thumb.jpg")),
    )
    monkeypatch.setattr(main, "analyze_thumbnail", lambda thumbnail: SimpleNamespace(status="success"))
    monkeypatch.setattr(main, "save_intelligence", lambda intelligence, analysis_dir: None)
    monkeypatch.setattr(main, "build_redesign_specification", lambda intelligence: SimpleNamespace())
    monkeypatch.setattr(main, "save_redesign_spec", lambda redesign_spec, spec_dir: None)
    monkeypatch.setattr(
        main,
        "build_design_blueprint",
        lambda intelligence, spec, metadata: SimpleNamespace(headline="Test Headline", text_position=None),
    )
    monkeypatch.setattr(main, "save_design_blueprint", lambda blueprint, blueprint_dir: None)
    monkeypatch.setattr(main, "compile_prompt_package", lambda redesign_spec, design_blueprint=None: _prompt_package())
    monkeypatch.setattr(main, "save_prompt_package", lambda prompt_package, package_dir: None)
    monkeypatch.setattr(
        main,
        "AssetComposer",
        lambda: SimpleNamespace(prepare_generation_workspace=lambda vid: SimpleNamespace()),
    )
    monkeypatch.setattr(
        main,
        "_run_module7_generation",
        lambda prompt_package, *, metadata, thumbnail_dir, analysis_dir, generation_bundle=None, design_blueprint=None, **kwargs: (
            _ for _ in ()
        ).throw(Module7Error("generation failed")),
    )



    main.run_pipeline(
        csv_path=tmp_path / "creators.csv",
        thumbnail_dir=tmp_path / "thumbnails",
        analysis_dir=tmp_path / "analysis",
        redesign_spec_dir=tmp_path / "specs",
        prompt_package_dir=tmp_path / "packages",
    )


def test_module7_generation_helper_calls_generate_and_persists_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thumbnails = tmp_path / "thumbnails"
    analysis = tmp_path / "analysis"
    thumbnails.mkdir()
    analysis.mkdir()
    (thumbnails / f"{VIDEO_ID}.jpg").write_bytes(b"source-thumbnail")
    monkeypatch.setattr(main, "MODULE7_OUTPUT_DIR", tmp_path / "generated")

    calls: list[dict[str, object]] = []

    from io import BytesIO
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 720), color="blue")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill="red")
    draw.ellipse([100, 100, 200, 200], fill="yellow")
    img_buf = BytesIO()
    img.save(img_buf, format="PNG")
    fake_png_bytes = img_buf.getvalue()

    class FakeComfyUIClient:
        def generate(self, built_workflow, **kwargs):
            calls.append({"built_workflow": built_workflow, **kwargs})
            return SimpleNamespace(
                prompt_id="prompt-123",
                output_node_id="9",
                filename="result.png",
                subfolder="",
                image_type="output",
                format="png",
                content=fake_png_bytes,
                width=1280,
                height=720,
            )

    monkeypatch.setattr(main, "ComfyUIClient", FakeComfyUIClient)

    output_path = main._run_module7_generation(
        _prompt_package(),
        metadata=_metadata(),
        thumbnail_dir=thumbnails,
        analysis_dir=analysis,
    )

    assert output_path == tmp_path / "generated" / VIDEO_ID / f"{VIDEO_ID}.png"
    assert output_path.is_file() and output_path.stat().st_size > 0
    assert calls[0]["video_id"] == VIDEO_ID
    assert calls[0]["num_candidates_requested"] == 1
    assert calls[0]["built_workflow"].workflow_ref.template_name == "gaming"
    manifest = json.loads(
        (tmp_path / "generated" / VIDEO_ID / f"{VIDEO_ID}_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "success"
    assert manifest["generated_asset"]["path"] == str(output_path)
    assert manifest["generated_asset"]["width"] == 1280
    assert manifest["profile_name"] == "PROFILE_LOW_VRAM"

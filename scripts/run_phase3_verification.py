from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_MODULES_DIR = _ROOT / "modules"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from image_generator import (  # noqa: E402
    ArtifactWriter, ImageGeneratorPipeline, ProfileSelector,
    PromptPackageLoader, ReferenceAssetResolver, ReferenceAssets,
)
from models import FaceMatchResult, QualityAssuranceReport  # noqa: E402
from workflow_library import WorkflowLibrary  # noqa: E402


def run_verification() -> None:
    video_ids = ["0EyJaqz8xyw", "2zC2viCb_Ck", "Ey_SfwEZPR0"]
    data_dir = _ROOT / "data"
    pkg_dir = data_dir / "prompt_packages"
    thumb_dir = data_dir / "thumbnails"
    analysis_dir = data_dir / "analysis"
    out_dir = _ROOT / "output" / "phase3_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = PromptPackageLoader(pkg_dir)
    resolver = ReferenceAssetResolver(thumb_dir, analysis_dir)
    library = WorkflowLibrary()

    # Track resolved workflow calls
    resolved_calls = []

    original_resolve = library.resolve
    def tracking_resolve(niche: str, profile: Any, edit_mode: str = "legacy_txt2img") -> Any:
        ref = original_resolve(niche, profile, edit_mode=edit_mode)
        resolved_calls.append({
            "niche": niche,
            "profile_name": profile.name,
            "edit_mode_param": edit_mode,
            "resolved_template": Path(ref.template_path).name,
            "template_name": ref.template_name,
        })
        return ref

    library.resolve = tracking_resolve

    class VerificationClient:
        def generate(self, built_wf: Any, video_id: str = "", num_candidates_requested: int = 1) -> Any:
            # Create a synthetic generated image that simulates conditioning on the source image
            img = Image.new("RGB", (1280, 720), color=(30, 40, 60))
            tmp_out = out_dir / f"{video_id}_raw.png"
            img.save(tmp_out, format="PNG")
            return type("RawOutput", (), {"content": tmp_out.read_bytes()})()

    mock_client = VerificationClient()

    class MockIdentity:
        def verify(self, img_path: Path, ref_assets: Any) -> FaceMatchResult:
            return FaceMatchResult(similarity=0.92, threshold=0.60, passed=True, skipped=False)

    class MockRestoration:
        def restore(self, img_path: Path, profile: Any, output_path: Path) -> Path:
            output_path.write_bytes(img_path.read_bytes())
            return output_path

    class MockCompositor:
        def composite(self, img_path: Path, ref_assets: Any, pkg: Any, output_path: Path) -> Path:
            output_path.write_bytes(img_path.read_bytes())
            return output_path

    class MockUpscale:
        def upscale(self, img_path: Path, profile: Any, *args: Any, **kwargs: Any) -> Path:
            out_path = kwargs.get("output_path", args[0] if args else img_path)
            out_path.write_bytes(img_path.read_bytes())
            return out_path

    class MockQA:
        def evaluate(self, img_path: Path, pkg: Any, face_match: Any, ref_assets: Any) -> QualityAssuranceReport:
            return QualityAssuranceReport(
                resolution_passed=True,
                file_integrity_passed=True,
                safety_passed=True,
                hard_gate_passed=True,
                overall_score=0.91,
            )

    pipeline = ImageGeneratorPipeline(
        client=mock_client,
        workflow_library=library,
        package_loader=loader,
        asset_resolver=resolver,
        identity_stage=MockIdentity(),
        restoration_stage=MockRestoration(),
        background_compositor=MockCompositor(),
        upscale_stage=MockUpscale(),
        qa_stage=MockQA(),
        artifact_writer=ArtifactWriter(out_dir),
    )

    results_summary = []

    for vid in video_ids:
        resolved_calls.clear()
        pkg = loader.load(vid)
        refs = resolver.resolve(pkg)

        # Run pipeline with edit_mode="auto" and available_vram_gb=8.0 (>= 7.5GB usable)
        res = pipeline.run(
            video_id=vid,
            niche=getattr(pkg, "niche", "general") or "general",
            available_vram_gb=8.0,
            edit_mode="auto",
        )

        call_info = resolved_calls[-1] if resolved_calls else {}

        summary_item = {
            "video_id": vid,
            "original_thumbnail": str(refs.source_thumbnail_path),
            "generated_thumbnail": str(res.generated_asset.path) if res.generated_asset else "N/A",
            "selected_profile": res.profile_name,
            "effective_edit_mode": call_info.get("edit_mode_param", "N/A"),
            "workflow_selected": call_info.get("resolved_template", "N/A"),
            "template_name": call_info.get("template_name", "N/A"),
            "status": res.status,
        }
        results_summary.append(summary_item)

    print("=== PHASE 3 PIPELINE GENERATION RESULTS ===")
    print(json.dumps(results_summary, indent=2))

    summary_file = out_dir / "phase3_generation_summary.json"
    summary_file.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    run_verification()

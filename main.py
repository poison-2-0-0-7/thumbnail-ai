"""
main.py
========

Entry point for the AI-powered YouTube Thumbnail Outreach Automation
system.

Pipeline
--------
Module 1  CSV Reader
          Loads the ``creators.csv`` lead list.
          ↓
Module 2  YouTube Metadata Extractor
          For each creator's video URL, fetches video metadata and
          transcript via yt-dlp and youtube-transcript-api.
          ↓
Module 3  Thumbnail Downloader
          Downloads and validates the video thumbnail, caching it to
          ``data/thumbnails/{video_id}.jpg``.
          ↓
Module 4  Thumbnail Intelligence Engine
          Analyzes the downloaded thumbnail together with its video
          metadata (title, description, transcript) via OCR, face
          analysis, object detection, color analysis, composition
          analysis, and local Ollama-based reasoning, saving a structured
          report to ``data/analysis/{video_id}.json``.
          ↓
Future modules ...

Running
-------
From the project root::

    python main.py

The script processes every creator in ``data/creators.csv``.
Successful results are printed to stdout; errors are logged to
``logs/`` and reported in the summary without terminating the run.
"""

from __future__ import annotations

import hashlib
import sys
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure modules/ is importable regardless of working directory

# ---------------------------------------------------------------------------
# Ensure modules/ is importable regardless of working directory
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent
_MODULES_DIR: Path = _PROJECT_ROOT / "modules"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger  # noqa: E402

from config import (  # noqa: E402
    ASSET_EXTRACTION_ENABLED,
    ASSET_EXTRACTION_REQUIRED,
    COMFYUI_HOST,
    COMFYUI_PORT,
    DECISION_ENGINE_ENABLED,
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ASSET_EXTRACTION_DIR,
    DEFAULT_CSV_PATH,
    DEFAULT_DECISION_DIR,
    DEFAULT_DESIGN_BLUEPRINT_DIR,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    DEFAULT_THUMBNAIL_DIR,
    MODULE7_GENERATION_PROFILES,
    MODULE7_OUTPUT_DIR,
    MODULE7_PROFILE,
    THUMBNAIL_PLANNER_ENABLED,
)
from csv_reader import load_all_creators  # noqa: E402
from comfyui_client import ComfyUIClient  # noqa: E402
from comfyui_manager import ComfyUIProcessManager  # noqa: E402
from image_generator import (  # noqa: E402
    ArtifactWriteError,
    ArtifactWriter,
    ImageGeneratorPipeline,
    ProfileSelector,
    ReferenceAssetResolver,
    WorkflowBuilder,
    generation_hash,
    prompt_package_hash,
    run_image_generation_pipeline,
    utc_now,
)
from models import (  # noqa: E402
    DecisionManifest,
    GeneratedAsset,
    GenerationBundle,
    GenerationPlan,
    ImageGenerationResult,
    PromptPackage,
    ThumbnailData,
    VideoMetadata,
)

from module7_exceptions import ComfyUIStartupError, Module7Error  # noqa: E402
from asset_extraction_engine import extract_assets  # noqa: E402
from asset_extraction_exceptions import AssetExtractionError  # noqa: E402
from decision_engine import run_decision_engine  # noqa: E402
from decision_exceptions import DecisionEngineError  # noqa: E402
from thumbnail_downloader import (  # noqa: E402
    ThumbnailDownloaderError,
    process_thumbnail,
)
from thumbnail_intelligence import (  # noqa: E402
    IntelligenceCacheError,
    InvalidMetadataError,
    analyze_thumbnail,
    save_intelligence,
)
from redesign_spec_engine import (  # noqa: E402
    InvalidIntelligenceError,
    RedesignSpecCacheError,
    build_redesign_specification,
    save_redesign_spec,
)
from design_blueprint_engine import (  # noqa: E402
    DesignBlueprintCacheError,
    DesignBlueprintError,
    build_design_blueprint,
    save_design_blueprint,
)
from prompt_compiler import (  # noqa: E402
    InvalidRedesignSpecError,
    PromptPackageCacheError,
    compile_prompt_package,
    save_prompt_package,
)
from composition_engine import AssetComposer  # noqa: E402
from composition_exceptions import CompositionBaseError  # noqa: E402
from thumbnail_planner import ThumbnailPlanner  # noqa: E402
from thumbnail_planner_exceptions import ThumbnailPlannerError  # noqa: E402
from workflow_library import WorkflowLibrary  # noqa: E402
from youtube_metadata import process_video  # noqa: E402



def run_pipeline(
    csv_path: Path = DEFAULT_CSV_PATH,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    redesign_spec_dir: Path = DEFAULT_REDESIGN_SPEC_DIR,
    design_blueprint_dir: Path = DEFAULT_DESIGN_BLUEPRINT_DIR,
    prompt_package_dir: Path = DEFAULT_PROMPT_PACKAGE_DIR,
    asset_extraction_dir: Path = DEFAULT_ASSET_EXTRACTION_DIR,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    comfyui_manager: ComfyUIProcessManager | None = None,
) -> None:
    """
    Execute the full pipeline for every creator in ``csv_path``.

    Processing is best-effort: a failure on one creator is logged and
    counted, but never prevents the remaining creators from being
    processed.

    Args:
        csv_path:      Path to the creators CSV file.
        thumbnail_dir: Directory where thumbnails are saved.
        analysis_dir:  Directory where thumbnail intelligence reports
                       are saved as JSON.
        redesign_spec_dir: Directory where deterministic Module 5
                           redesign specifications are saved.
        design_blueprint_dir: Directory where deterministic Module 5.5
                               design blueprints are saved.
        prompt_package_dir: Directory where deterministic Module 6
                            prompt packages are saved.
        asset_extraction_dir: Directory where Module 8 asset manifests
                              are saved.
        decision_dir:   Directory where Module 9 decision manifests
                        are saved.
        comfyui_manager: Optional ComfyUI Process Manager instance.
    """
    logger.info("Pipeline starting — CSV: {csv}", csv=csv_path)

    # ── ComfyUI Process Management ───────────────────────────────────────
    manager = comfyui_manager if comfyui_manager is not None else ComfyUIProcessManager()
    logger.info("Checking ComfyUI service status...")
    try:
        manager.ensure_started()
    except ComfyUIStartupError as exc:
        logger.error("Pipeline initialization failed — ComfyUI startup error: {exc}", exc=exc)
        print(f"\nPipeline execution aborted: ComfyUI startup failed: {exc}")
        return

    try:
        _run_pipeline_creators(
            csv_path=csv_path,
            thumbnail_dir=thumbnail_dir,
            analysis_dir=analysis_dir,
            redesign_spec_dir=redesign_spec_dir,
            design_blueprint_dir=design_blueprint_dir,
            prompt_package_dir=prompt_package_dir,
            asset_extraction_dir=asset_extraction_dir,
            decision_dir=decision_dir,
            comfyui_manager=manager,
        )
    finally:
        if manager.shutdown_on_exit:
            manager.stop()


def _run_pipeline_creators(
    csv_path: Path,
    thumbnail_dir: Path,
    analysis_dir: Path,
    redesign_spec_dir: Path,
    design_blueprint_dir: Path,
    prompt_package_dir: Path,
    asset_extraction_dir: Path = DEFAULT_ASSET_EXTRACTION_DIR,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    comfyui_manager: ComfyUIProcessManager | None = None,
) -> None:
    # ── Module 1: load creators ──────────────────────────────────────────
    creators = load_all_creators(csv_path)
    total = len(creators)
    logger.info("Loaded {n} creator(s) from CSV", n=total)

    if not creators:
        logger.warning("No creators found in {csv} — nothing to do.", csv=csv_path)
        return

    succeeded = 0
    skipped = 0

    for creator in creators:
        logger.info(
            "Processing creator_email={email} url={url}",
            email=creator.email,
            url=creator.video_url,
        )

        # ── Module 2: extract metadata ───────────────────────────────────
        # enable_oembed_fallback=True: if yt-dlp itself fails (bot-check or
        # a transient network error), fall back to YouTube's public oEmbed
        # endpoint for partial-but-genuine metadata rather than failing the
        # creator outright. Disabled by default in process_video() so that
        # the test suite stays fully offline and deterministic.
        metadata: VideoMetadata = process_video(creator, enable_oembed_fallback=True)

        if metadata.status == "error":
            logger.error(
                "Metadata extraction failed for creator_email={email}: {reason}",
                email=creator.email,
                reason=metadata.error_message,
            )
            skipped += 1
            continue

        logger.info(
            "Metadata OK for creator_email={email}: {title!r}",
            email=creator.email,
            title=metadata.title,
        )

        # ── Module 3: download thumbnail ─────────────────────────────────
        try:
            thumbnail: ThumbnailData = process_thumbnail(
                metadata,
                thumbnail_dir=thumbnail_dir,
            )
        except ThumbnailDownloaderError as exc:
            logger.error(
                "Thumbnail download failed for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        logger.info(
            "Thumbnail saved for creator_email={email}: {path}",
            email=creator.email,
            path=thumbnail.thumbnail_path,
        )

        # ── Module 4: analyze thumbnail intelligence ─────────────────────
        try:
            intelligence = analyze_thumbnail(thumbnail)
        except InvalidMetadataError as exc:
            logger.error(
                "Thumbnail intelligence skipped for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        try:
            save_intelligence(intelligence, analysis_dir=analysis_dir)
        except IntelligenceCacheError as exc:
            logger.error(
                "Failed to save thumbnail intelligence for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        if intelligence.status == "error":
            logger.error(
                "Thumbnail intelligence failed for creator_email={email} "
                "video_id={vid}: {reason}",
                email=creator.email,
                vid=metadata.video_id,
                reason=intelligence.error_message,
            )
            skipped += 1
            continue

        if intelligence.status == "partial":
            logger.warning(
                "Thumbnail intelligence partially degraded for creator_email={email} "
                "video_id={vid}: {reasons}",
                email=creator.email,
                vid=metadata.video_id,
                reasons=intelligence.partial_failure_reasons,
            )

        logger.info(
            "Thumbnail intelligence saved for creator_email={email} "
            "video_id={vid}: status={status}",
            email=creator.email,
            vid=metadata.video_id,
            status=intelligence.status,
        )

        # ── Module 8: extract visual asset manifest ───────────────────────
        if ASSET_EXTRACTION_ENABLED:
            try:
                extract_assets(
                    metadata.video_id,
                    source_image_path=str(thumbnail.thumbnail_path),
                    intelligence=intelligence,
                    storage_root=asset_extraction_dir,
                )
                logger.info(
                    "Asset extraction saved for creator_email={email} video_id={vid}",
                    email=creator.email,
                    vid=metadata.video_id,
                )
            except AssetExtractionError as exc:
                logger.error(
                    "Asset extraction failed for creator_email={email} "
                    "video_id={vid}: {exc}",
                    email=creator.email,
                    vid=metadata.video_id,
                    exc=exc,
                )
                if ASSET_EXTRACTION_REQUIRED:
                    skipped += 1
                    continue

        # ── Module 5: derive deterministic redesign specification ──
        try:
            redesign_spec = build_redesign_specification(intelligence)
            save_redesign_spec(redesign_spec, spec_dir=redesign_spec_dir)
        except (InvalidIntelligenceError, RedesignSpecCacheError) as exc:
            logger.error(
                "Redesign specification failed for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        # ── Module 5.5: build deterministic design blueprint ──────────────
        try:
            design_blueprint = build_design_blueprint(
                intelligence, redesign_spec, metadata
            )
            save_design_blueprint(design_blueprint, blueprint_dir=design_blueprint_dir)
        except (DesignBlueprintError, DesignBlueprintCacheError) as exc:
            logger.error(
                "Design blueprint failed for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        logger.info(
            "Design blueprint saved for creator_email={email} video_id={vid}",
            email=creator.email,
            vid=metadata.video_id,
        )

        # ── Module 6: compile deterministic image-generation package ──────
        try:
            prompt_package = compile_prompt_package(
                redesign_spec, design_blueprint=design_blueprint
            )
            save_prompt_package(prompt_package, package_dir=prompt_package_dir)
        except (InvalidRedesignSpecError, PromptPackageCacheError) as exc:
            logger.error(
                "Prompt compilation failed for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        logger.info(
            "Prompt package saved for creator_email={email} video_id={vid}",
            email=creator.email,
            vid=metadata.video_id,
        )

        # ── Module 9: AI decision engine ──────────────────────────────────
        decision_manifest: DecisionManifest | None = None
        if DECISION_ENGINE_ENABLED:
            try:
                decision_manifest = run_decision_engine(
                    metadata.video_id,
                    decision_dir=decision_dir,
                    analysis_dir=analysis_dir,
                    redesign_spec_dir=redesign_spec_dir,
                    prompt_package_dir=prompt_package_dir,
                    asset_extraction_dir=asset_extraction_dir,
                )
                logger.info(
                    "Decision manifest saved for creator_email={email} video_id={vid}",
                    email=creator.email,
                    vid=metadata.video_id,
                )
            except DecisionEngineError as exc:
                logger.error(
                    "Decision engine failed for creator_email={email} "
                    "video_id={vid}: {exc}",
                    email=creator.email,
                    vid=metadata.video_id,
                    exc=exc,
                )
                skipped += 1
                continue

        # ── Module 10: prepare composition workspace ─────────────────────
        try:
            generation_bundle = AssetComposer().prepare_generation_workspace(
                prompt_package.video_id,
                decision_manifest=decision_manifest if DECISION_ENGINE_ENABLED else None,
            )
        except CompositionBaseError as exc:
            logger.error(
                "Composition workspace preparation failed for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        logger.info(
            "Composition workspace prepared for creator_email={email} video_id={vid}",
            email=creator.email,
            vid=metadata.video_id,
        )

        # ── Module 10.5: plan generation ────────────────────────────────
        generation_plan: GenerationPlan | None = None
        if THUMBNAIL_PLANNER_ENABLED:
            try:
                generation_plan = ThumbnailPlanner().plan(prompt_package.video_id)
            except ThumbnailPlannerError as exc:
                logger.error(
                    "Thumbnail planner failed for creator_email={email} "
                    "video_id={vid}: {exc}",
                    email=creator.email,
                    vid=metadata.video_id,
                    exc=exc,
                )
                skipped += 1
                continue

            logger.info(
                "Thumbnail plan created for creator_email={email} video_id={vid}",
                email=creator.email,
                vid=metadata.video_id,
            )

        # ── Module 7: generate final thumbnail ─────────────────────────────
        try:
            m7_kwargs = {
                "metadata": metadata,
                "thumbnail_dir": thumbnail_dir,
                "analysis_dir": analysis_dir,
                "generation_bundle": generation_bundle,
                "design_blueprint": design_blueprint,
            }
            if generation_plan is not None:
                m7_kwargs["generation_plan"] = generation_plan

            generated_path = _run_module7_generation(
                prompt_package,
                **m7_kwargs,
            )

        except Module7Error as exc:
            logger.error(
                "Image generation failed for creator_email={email} "
                "video_id={vid}: {exc}",
                email=creator.email,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        logger.info(
            "Generated thumbnail saved for creator_email={email} "
            "video_id={vid}: {path}",
            email=creator.email,
            vid=metadata.video_id,
            path=generated_path,
        )
        succeeded += 1

        # ── Automatic PORCE Observability ────────────────────────────────────
        try:
            from observability.runner import PORCEPipelineObserver
            PORCEPipelineObserver().observe(metadata.video_id)
        except Exception as exc:
            logger.warning(
                "Automatic PORCE observer encountered an error for video_id={vid}: {exc}",
                vid=metadata.video_id,
                exc=exc,
            )


    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(
        "Pipeline complete — {ok}/{total} succeeded, {skip} skipped/failed.",
        ok=succeeded,
        total=total,
        skip=skipped,
    )
    print(
        f"\nPipeline complete: {succeeded}/{total} creators processed "
        f"({skipped} skipped/failed). See logs/ for details."
    )


def _probe_available_vram_gb() -> float:
    """Probe hardware/ComfyUI to determine actual available GPU VRAM in GB."""
    # 1. Probe ComfyUI /system_stats API endpoint if running
    try:
        url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/system_stats"
        resp = requests.get(url, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            devices = data.get("devices", [])
            if isinstance(devices, list) and len(devices) > 0 and isinstance(devices[0], dict):
                dev = devices[0]
                vram_total = dev.get("vram_total") or dev.get("torch_vram_total")
                if vram_total is not None and float(vram_total) > 0:
                    vram_gb = round(float(vram_total) / (1024 ** 3), 1)
                    logger.info(
                        "VRAM probed via ComfyUI /system_stats: device='{device}', vram={vram_gb:.2f} GB",
                        device=dev.get("name", "GPU"),
                        vram_gb=vram_gb,
                    )
                    return vram_gb
    except Exception as exc:
        logger.debug("ComfyUI /system_stats VRAM probe failed: {exc}", exc=exc)

    # 2. Probe PyTorch CUDA if available
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(float(vram_bytes) / (1024 ** 3), 1)
            logger.info(
                "VRAM probed via torch.cuda: device='{device}', vram={vram_gb:.2f} GB",
                device=torch.cuda.get_device_name(0),
                vram_gb=vram_gb,
            )
            return vram_gb
    except Exception as exc:
        logger.debug("torch.cuda VRAM probe failed: {exc}", exc=exc)

    # Fallback to conservative LOW_VRAM if no GPU/ComfyUI detected
    selector = ProfileSelector()
    low_vram = MODULE7_GENERATION_PROFILES["PROFILE_LOW_VRAM"]
    fallback_vram = low_vram.expected_vram_gb + selector.headroom_gb
    logger.warning(
        "Could not detect live GPU VRAM; using conservative VRAM={vram:.1f} GB ({profile})",
        vram=fallback_vram,
        profile=low_vram.name,
    )
    return fallback_vram


def _select_module7_profile(vram_gb: float | None = None) -> GenerationProfile:
    """Select the configured Module 7 profile using real VRAM detection when auto."""
    selector = ProfileSelector()
    if MODULE7_PROFILE != "auto":
        return selector.select(float("inf"), MODULE7_PROFILE)

    if vram_gb is None:
        vram_gb = _probe_available_vram_gb()
    selected = selector.select(vram_gb, "auto")
    logger.info(
        "Module 7 auto profile selection resolved profile={profile} for detected_vram={vram_gb:.2f} GB",
        profile=selected.name,
        vram_gb=vram_gb,
    )
    return selected


def _run_module7_generation(
    prompt_package: PromptPackage,
    *,
    metadata: VideoMetadata,
    thumbnail_dir: Path,
    analysis_dir: Path,
    generation_bundle: GenerationBundle | None = None,
    generation_plan: GenerationPlan | None = None,
    design_blueprint: DesignBlueprint | None = None,
) -> Path:
    """Build the existing Module 7 inputs, generate image candidates, and persist output."""
    vram_gb = _probe_available_vram_gb()
    profile = _select_module7_profile(vram_gb)
    niche = _module7_niche(metadata)
    client = ComfyUIClient()
    return run_image_generation_pipeline(
        video_id=prompt_package.video_id,
        niche=niche,
        available_vram_gb=vram_gb,
        prompt_package=prompt_package,
        generation_bundle=generation_bundle,
        generation_plan=generation_plan,
        design_blueprint=design_blueprint,
        edit_mode="auto",
        client=client,
        thumbnail_dir=thumbnail_dir,
        analysis_dir=analysis_dir,
        output_dir=MODULE7_OUTPUT_DIR,
    )


def _module7_niche(metadata: VideoMetadata) -> str:
    """Use existing metadata categories as the workflow-library niche hint."""
    if metadata.categories:
        return metadata.categories[0].strip().lower()
    return "general"


def _persist_generated_thumbnail(
    prompt_package: PromptPackage,
    profile,
    built_workflow,
    output,
) -> Path:
    """Persist the generated bytes and existing Module 7 manifest model."""
    output_format = str(output.format).strip().lower().lstrip(".") or "png"
    target_dir = MODULE7_OUTPUT_DIR / prompt_package.video_id
    target = target_dir / f"{prompt_package.video_id}.{output_format}"
    temporary = target.with_suffix(".tmp")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(output.content)
        temporary.replace(target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ArtifactWriteError(f"Could not write generated thumbnail to {target}: {exc}") from exc

    width = _required_dimension(output.width, "width")
    height = _required_dimension(output.height, "height")
    content_hash = hashlib.sha256(output.content).hexdigest()
    package_hash = prompt_package_hash(prompt_package)
    result = ImageGenerationResult(
        video_id=prompt_package.video_id,
        generated_asset=GeneratedAsset(
            path=str(target),
            width=width,
            height=height,
            sha256=content_hash,
            candidate_index=0,
        ),
        workflow_version=built_workflow.workflow_ref.workflow_version,
        workflow_hash=built_workflow.workflow_hash,
        prompt_package_hash=package_hash,
        generation_hash=generation_hash(
            built_workflow.workflow_hash,
            package_hash,
            None,
            [],
            [],
            prompt_package.generation_parameters.seed,
            profile.name,
        ),
        profile_name=profile.name,
        seed=prompt_package.generation_parameters.seed,
        selected_candidate_index=0,
        generated_at=utc_now(),
    )
    ArtifactWriter(MODULE7_OUTPUT_DIR).write_manifest(result)
    return target


def _required_dimension(value: int | None, field_name: str) -> int:
    if value is None:
        raise ArtifactWriteError(f"Generated output is missing {field_name}")
    return int(value)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

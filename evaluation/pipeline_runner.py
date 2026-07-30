"""
pipeline_runner.py
==================

Drives the production pipeline stage-by-stage and validates persisted artifacts.
Observer pattern — zero modification to main.py or modules 1–10.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Callable, Sequence

from loguru import logger

from modules.config import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_CSV_PATH,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    DEFAULT_THUMBNAIL_DIR,
    MODULE7_OUTPUT_DIR,
    PROJECT_ROOT,
)
from modules.models import (
    ModuleValidationResult,
    PipelineRunReport,
)
from .config import EVAL_LOG_PATH, EVAL_RUNS_DIR
from .evaluation_exceptions import PipelineStageInvocationError, PVQEFError
from .module_validators import (
    AssetComposerValidator,
    CSVReaderValidator,
    IModuleValidator,
    Module7Validator,
    PromptCompilerValidator,
    RedesignSpecValidator,
    ThumbnailDownloaderValidator,
    ThumbnailIntelligenceValidator,
    YouTubeMetadataValidator,
)

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(EVAL_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()

ALL_STAGES: tuple[str, ...] = (
    "module1_csv_reader",
    "module2_youtube_metadata",
    "module3_thumbnail_downloader",
    "module4_thumbnail_intelligence",
    "module5_redesign_spec",
    "module6_prompt_compiler",
    "module10_asset_composer",
    "module7_image_generator",
)


class PipelineRunner:
    """Orchestrates pipeline execution observation and artifact validation."""

    def __init__(
        self,
        validators: Sequence[IModuleValidator] | None = None,
    ) -> None:
        default_validators = [
            CSVReaderValidator(),
            YouTubeMetadataValidator(),
            ThumbnailDownloaderValidator(),
            ThumbnailIntelligenceValidator(),
            RedesignSpecValidator(),
            PromptCompilerValidator(),
            AssetComposerValidator(),
            Module7Validator(),
        ]
        chosen = validators if validators is not None else default_validators
        self.validators: dict[str, IModuleValidator] = {v.module_name: v for v in chosen}

    def run(
        self,
        csv_path: Path = DEFAULT_CSV_PATH,
        *,
        run_id: str | None = None,
        golden_only: bool = False,
        stages: tuple[str, ...] | None = None,
    ) -> PipelineRunReport:
        """Run validation harness across requested pipeline stages."""
        start_mono = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        effective_run_id = run_id or self._generate_run_id()
        selected_stages = stages or ALL_STAGES

        logger.info(
            "Starting pipeline evaluation run_id={run_id} csv={csv} stages={stages}",
            run_id=effective_run_id,
            csv=str(csv_path),
            stages=selected_stages,
        )

        from csv_reader import load_all_creators

        try:
            creators = load_all_creators(csv_path)
        except Exception as exc:
            logger.error("Failed loading CSV {csv}: {exc}", csv=str(csv_path), exc=exc)
            creators = []

        total_creators = len(creators)
        succeeded = 0
        skipped = 0
        module_results: dict[str, list[ModuleValidationResult]] = {}
        stage_failure_counts: dict[str, int] = {s: 0 for s in selected_stages}
        stage_durations: dict[str, list[float]] = {s: [] for s in selected_stages}

        for creator in creators:
            video_id = getattr(creator, "video_id", None) or hashlib.md5(creator.video_url.encode()).hexdigest()[:10]
            creator_results: list[ModuleValidationResult] = []
            creator_failed = False

            # Module 1
            if "module1_csv_reader" in selected_stages:
                val = self._validate_stage("module1_csv_reader", video_id, csv_path, stage_durations)
                creator_results.append(val)
                if val.status == "error":
                    creator_failed = True

            # Import module execution helpers
            from youtube_metadata import process_video
            from thumbnail_downloader import process_thumbnail
            from thumbnail_intelligence import analyze_thumbnail, save_intelligence
            from redesign_spec_engine import build_redesign_specification, save_redesign_spec
            from prompt_compiler import compile_prompt_package, save_prompt_package
            from composition_engine import AssetComposer
            from main import _run_module7_generation

            metadata = None
            thumbnail = None
            intelligence = None
            redesign_spec = None
            prompt_package = None
            generation_bundle = None

            # Module 2
            if "module2_youtube_metadata" in selected_stages and not creator_failed:
                t0 = time.monotonic()
                try:
                    metadata = process_video(creator, enable_oembed_fallback=True)
                    dur = time.monotonic() - t0
                    stage_durations["module2_youtube_metadata"].append(dur)
                    artifact_p = DEFAULT_ANALYSIS_DIR / f"{video_id}.json"
                    val = self._validate_stage("module2_youtube_metadata", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if metadata.status == "error" or val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module2_youtube_metadata"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module2_youtube_metadata", video_id, str(exc), time.monotonic() - t0))

            # Module 3
            if "module3_thumbnail_downloader" in selected_stages and not creator_failed and metadata:
                t0 = time.monotonic()
                try:
                    thumbnail = process_thumbnail(metadata, thumbnail_dir=DEFAULT_THUMBNAIL_DIR)
                    dur = time.monotonic() - t0
                    stage_durations["module3_thumbnail_downloader"].append(dur)
                    artifact_p = DEFAULT_ANALYSIS_DIR / f"{video_id}.json"
                    val = self._validate_stage("module3_thumbnail_downloader", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module3_thumbnail_downloader"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module3_thumbnail_downloader", video_id, str(exc), time.monotonic() - t0))

            # Module 4
            if "module4_thumbnail_intelligence" in selected_stages and not creator_failed and thumbnail:
                t0 = time.monotonic()
                try:
                    intelligence = analyze_thumbnail(thumbnail)
                    save_intelligence(intelligence, analysis_dir=DEFAULT_ANALYSIS_DIR)
                    dur = time.monotonic() - t0
                    stage_durations["module4_thumbnail_intelligence"].append(dur)
                    artifact_p = DEFAULT_ANALYSIS_DIR / f"{video_id}.json"
                    val = self._validate_stage("module4_thumbnail_intelligence", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if intelligence.status == "error" or val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module4_thumbnail_intelligence"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module4_thumbnail_intelligence", video_id, str(exc), time.monotonic() - t0))

            # Module 5
            if "module5_redesign_spec" in selected_stages and not creator_failed and intelligence:
                t0 = time.monotonic()
                try:
                    redesign_spec = build_redesign_specification(intelligence)
                    save_redesign_spec(redesign_spec, spec_dir=DEFAULT_REDESIGN_SPEC_DIR)
                    dur = time.monotonic() - t0
                    stage_durations["module5_redesign_spec"].append(dur)
                    artifact_p = DEFAULT_REDESIGN_SPEC_DIR / f"{video_id}.json"
                    val = self._validate_stage("module5_redesign_spec", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module5_redesign_spec"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module5_redesign_spec", video_id, str(exc), time.monotonic() - t0))

            # Module 6
            if "module6_prompt_compiler" in selected_stages and not creator_failed and redesign_spec:
                t0 = time.monotonic()
                try:
                    prompt_package = compile_prompt_package(redesign_spec)
                    save_prompt_package(prompt_package, package_dir=DEFAULT_PROMPT_PACKAGE_DIR)
                    dur = time.monotonic() - t0
                    stage_durations["module6_prompt_compiler"].append(dur)
                    artifact_p = DEFAULT_PROMPT_PACKAGE_DIR / f"{video_id}.json"
                    val = self._validate_stage("module6_prompt_compiler", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module6_prompt_compiler"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module6_prompt_compiler", video_id, str(exc), time.monotonic() - t0))

            # Module 10
            if "module10_asset_composer" in selected_stages and not creator_failed and prompt_package:
                t0 = time.monotonic()
                try:
                    generation_bundle = AssetComposer().prepare_generation_workspace(video_id)
                    dur = time.monotonic() - t0
                    stage_durations["module10_asset_composer"].append(dur)
                    artifact_p = PROJECT_ROOT / "data" / "composition_workspaces" / video_id / "workspace_manifest.json"
                    val = self._validate_stage("module10_asset_composer", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module10_asset_composer"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module10_asset_composer", video_id, str(exc), time.monotonic() - t0))

            # Module 7
            if "module7_image_generator" in selected_stages and not creator_failed and prompt_package and metadata:
                t0 = time.monotonic()
                try:
                    _run_module7_generation(
                        prompt_package,
                        metadata=metadata,
                        thumbnail_dir=DEFAULT_THUMBNAIL_DIR,
                        analysis_dir=DEFAULT_ANALYSIS_DIR,
                        generation_bundle=generation_bundle,
                    )
                    dur = time.monotonic() - t0
                    stage_durations["module7_image_generator"].append(dur)
                    artifact_p = MODULE7_OUTPUT_DIR / video_id / "result_manifest.json"
                    val = self._validate_stage("module7_image_generator", video_id, artifact_p, stage_durations)
                    creator_results.append(val)
                    if val.status == "error":
                        creator_failed = True
                except Exception as exc:
                    stage_failure_counts["module7_image_generator"] += 1
                    creator_failed = True
                    creator_results.append(self._make_error_result("module7_image_generator", video_id, str(exc), time.monotonic() - t0))

            module_results[video_id] = creator_results
            if creator_failed:
                skipped += 1
            else:
                succeeded += 1

        aggregate_perf = {
            s: (sum(stage_durations[s]) / len(stage_durations[s])) if stage_durations[s] else 0.0
            for s in selected_stages
        }
        total_duration = time.monotonic() - start_mono
        completed_at = datetime.now(timezone.utc).isoformat()

        report = PipelineRunReport(
            run_id=effective_run_id,
            csv_path=str(csv_path),
            golden_only=golden_only,
            total_creators=total_creators,
            succeeded=succeeded,
            skipped=skipped,
            module_results=module_results,
            quality_reports={},
            regressions=[],
            stage_failure_counts=stage_failure_counts,
            aggregate_performance=aggregate_perf,
            status="success" if skipped == 0 else ("partial" if succeeded > 0 else "error"),
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
        )

        return report

    def _validate_stage(
        self,
        module_name: str,
        video_id: str,
        artifact_path: Path,
        stage_durations: dict[str, list[float]],
    ) -> ModuleValidationResult:
        validator = self.validators.get(module_name)
        if not validator:
            return self._make_error_result(module_name, video_id, f"No validator for {module_name}", 0.0)
        t0 = time.monotonic()
        result = validator.validate(video_id, artifact_path)
        stage_durations[module_name].append(time.monotonic() - t0)
        return result

    def _make_error_result(
        self,
        module_name: str,
        video_id: str,
        error_msg: str,
        duration: float,
    ) -> ModuleValidationResult:
        return ModuleValidationResult(
            video_id=video_id,
            module_name=module_name,
            artifact_path=None,
            schema_valid=False,
            invariants_checked=[],
            invariants_failed=["execution_success"],
            status="error",
            error_message=error_msg,
            duration_seconds=duration,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_run_id(self) -> str:
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.md5(now_str.encode()).hexdigest()[:6]
        return f"run_{now_str}_{short_hash}"


def run_full_evaluation(
    csv_path: Path = DEFAULT_CSV_PATH,
    *,
    run_id: str | None = None,
    golden_only: bool = False,
    stages: tuple[str, ...] | None = None,
    validators: Sequence[IModuleValidator] | None = None,
) -> PipelineRunReport:
    """Public entry point for executing full pipeline artifact validation."""
    runner = PipelineRunner(validators=validators)
    return runner.run(
        csv_path=csv_path,
        run_id=run_id,
        golden_only=golden_only,
        stages=stages,
    )

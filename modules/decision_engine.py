"""
decision_engine.py
===================

Module 9 Orchestrator (AI Decision Engine).
Arbitrates visual element decisions (keep, remove, replace, enhance, add)
from upstream Modules 4, 5, 6, and 8 artifacts.
Cache-aware, resumable, deterministic-first, LLM-assisted.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from loguru import logger

from modules.config import (
    DEFAULT_DECISION_DIR,
    DECISION_CACHE_ENABLED,
    MODULE9_LOG_PATH,
)
from modules.decision_components.ambiguity_router import AmbiguityRouter
from modules.decision_components.conflict_resolver import ConflictResolver
from modules.decision_components.io import DecisionCache, load_cached_decision_manifest, load_input_bundle, save_decision_manifest
from modules.decision_components.interfaces import (
    IAmbiguityRouter,
    IConflictResolver,
    IDecisionCache,
    IDecisionValidator,
    ILLMReasoner,
    IManifestAssembler,
    IRuleEngine,
)
from modules.decision_components.llm_reasoner import LLMReasoner
from modules.decision_components.manifest_assembler import ManifestAssembler
from modules.decision_components.metrics import MetricsCollector
from modules.decision_components.rule_engine import RuleEngine
from modules.decision_components.validator import DecisionValidator
from modules.decision_exceptions import DecisionEngineError
from modules.models import (
    DecisionManifest,
    DecisionManifestStatus,
)

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    MODULE9_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE9_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


class DecisionEngine:
    """Orchestrator class for Module 9 AI Decision Engine."""

    def __init__(
        self,
        decision_dir: Path = DEFAULT_DECISION_DIR,
        analysis_dir: Path | None = None,
        redesign_spec_dir: Path | None = None,
        prompt_package_dir: Path | None = None,
        asset_extraction_dir: Path | None = None,
        rule_engine: IRuleEngine | None = None,
        ambiguity_router: IAmbiguityRouter | None = None,
        llm_reasoner: ILLMReasoner | None = None,
        conflict_resolver: IConflictResolver | None = None,
        validator: IDecisionValidator | None = None,
        manifest_assembler: IManifestAssembler | None = None,
        cache: IDecisionCache | None = None,
        metrics: MetricsCollector | None = None,
        cache_enabled: bool = DECISION_CACHE_ENABLED,
    ) -> None:
        self.decision_dir = Path(decision_dir)
        self.analysis_dir = Path(analysis_dir) if analysis_dir else None
        self.redesign_spec_dir = Path(redesign_spec_dir) if redesign_spec_dir else None
        self.prompt_package_dir = Path(prompt_package_dir) if prompt_package_dir else None
        self.asset_extraction_dir = Path(asset_extraction_dir) if asset_extraction_dir else None
        self.cache_enabled = cache_enabled

        self.rule_engine = rule_engine or RuleEngine()
        self.ambiguity_router = ambiguity_router or AmbiguityRouter()
        self.llm_reasoner = llm_reasoner or LLMReasoner()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.validator = validator or DecisionValidator()
        self.manifest_assembler = manifest_assembler or ManifestAssembler()
        self.cache = cache or DecisionCache(decision_dir=self.decision_dir)
        self.metrics = metrics or MetricsCollector()

    def run(self, video_id: str, force_recompute: bool = False) -> DecisionManifest:
        """Run full decision pipeline for one video_id."""
        started_time = time.monotonic()
        video_id = self._validate_video_id(video_id)

        # 1. Cache hit check
        if self.cache_enabled and not force_recompute:
            cached = self.cache.load(video_id)
            if cached is not None:
                logger.info("Decision manifest cache hit for video_id={id}", id=video_id)
                return cached

        logger.info("Executing AI Decision Engine for video_id={id}", id=video_id)

        try:
            # 2. Ingestion
            bundle = load_input_bundle(
                video_id,
                analysis_dir=self.analysis_dir,
                redesign_spec_dir=self.redesign_spec_dir,
                prompt_package_dir=self.prompt_package_dir,
                asset_extraction_dir=self.asset_extraction_dir,
            )

            # 3. Rule Evaluation
            candidates = self.rule_engine.evaluate(bundle)
            candidate_count = len(candidates)

            # 4. Ambiguity Routing
            confident_cands, needs_llm_cands = self.ambiguity_router.select(candidates)

            # 5. LLM Adjudication
            llm_adjudications_count = len(needs_llm_cands)
            if needs_llm_cands:
                logger.info(
                    "Routing {count} ambiguous candidates to LLM adjudication video_id={id}",
                    count=llm_adjudications_count,
                    id=video_id,
                )
                adjudicated_cands = self.llm_reasoner.adjudicate(needs_llm_cands, bundle)
                all_candidates = confident_cands + adjudicated_cands
            else:
                all_candidates = confident_cands

            # 6. Conflict Resolution
            resolved_decisions = self.conflict_resolver.resolve(all_candidates)
            conflicts_resolved_count = sum(
                len(d.superseded_candidate_ids) for d in resolved_decisions
            )

            # 7. Validation
            validation_report = self.validator.validate(resolved_decisions, bundle=bundle)

            # 8. Manifest Assembly
            duration_seconds = time.monotonic() - started_time
            source_img_path = (
                bundle.asset_extraction.source_thumbnail_path
                if bundle.asset_extraction
                else bundle.intelligence.thumbnail_path
            )
            source_img_hash = (
                bundle.asset_extraction.source_hash
                if bundle.asset_extraction
                else "0" * 64
            )

            manifest = self.manifest_assembler.build(
                video_id=video_id,
                source_image_path=source_img_path,
                source_image_hash=source_img_hash,
                decisions=resolved_decisions,
                validation_report=validation_report,
                duration_seconds=duration_seconds,
            )

            # 9. Persistence
            self.cache.save(manifest)

            # 10. Metrics Record
            self.metrics.record_run(
                video_id=video_id,
                duration_seconds=duration_seconds,
                candidate_count=candidate_count,
                resolved_count=len(resolved_decisions),
                llm_adjudications_count=llm_adjudications_count,
                conflicts_resolved_count=conflicts_resolved_count,
                status=manifest.status.value,
            )

            logger.info(
                "Decision Engine complete video_id={id} status={status} duration={duration:.2f}s",
                id=video_id,
                status=manifest.status.value,
                duration=duration_seconds,
            )
            return manifest

        except Exception as exc:
            duration_seconds = time.monotonic() - started_time
            logger.error(
                "Decision Engine execution failed for video_id={id}: {exc}",
                id=video_id,
                exc=str(exc),
            )
            error_manifest = DecisionManifest(
                video_id=video_id,
                source_generated_image_path="",
                source_generated_image_hash="0" * 64,
                decisions=[],
                status=DecisionManifestStatus.ERROR,
                error_message=str(exc),
                total_duration_seconds=round(duration_seconds, 4),
                decided_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return error_manifest

    @staticmethod
    def _validate_video_id(video_id: str) -> str:
        if not video_id or not video_id.strip():
            raise ValueError("video_id must not be empty")
        return video_id.strip()


def run_decision_engine(
    video_id: str,
    *,
    force_recompute: bool = False,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    analysis_dir: Path | None = None,
    redesign_spec_dir: Path | None = None,
    prompt_package_dir: Path | None = None,
    asset_extraction_dir: Path | None = None,
) -> DecisionManifest:
    """Public top-level API for running Decision Engine on one video_id."""
    engine = DecisionEngine(
        decision_dir=decision_dir,
        analysis_dir=analysis_dir,
        redesign_spec_dir=redesign_spec_dir,
        prompt_package_dir=prompt_package_dir,
        asset_extraction_dir=asset_extraction_dir,
    )
    return engine.run(video_id, force_recompute=force_recompute)


def run_decision_engine_batch(
    video_ids: list[str],
    *,
    force_recompute: bool = False,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    analysis_dir: Path | None = None,
    redesign_spec_dir: Path | None = None,
    prompt_package_dir: Path | None = None,
    asset_extraction_dir: Path | None = None,
) -> list[DecisionManifest]:
    """Public batch entry point. Processes video_ids with per-video isolation."""
    engine = DecisionEngine(
        decision_dir=decision_dir,
        analysis_dir=analysis_dir,
        redesign_spec_dir=redesign_spec_dir,
        prompt_package_dir=prompt_package_dir,
        asset_extraction_dir=asset_extraction_dir,
    )
    results: list[DecisionManifest] = []
    for vid in video_ids:
        try:
            manifest = engine.run(vid, force_recompute=force_recompute)
            results.append(manifest)
        except Exception as exc:
            logger.error("Unexpected error in batch for video_id={id}: {exc}", id=vid, exc=str(exc))
            results.append(
                DecisionManifest(
                    video_id=vid,
                    source_generated_image_path="",
                    source_generated_image_hash="0" * 64,
                    status=DecisionManifestStatus.ERROR,
                    error_message=str(exc),
                    decided_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            )
    return results

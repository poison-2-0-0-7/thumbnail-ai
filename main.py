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

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure modules/ is importable regardless of working directory
# ---------------------------------------------------------------------------

_MODULES_DIR: Path = Path(__file__).resolve().parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_CSV_PATH,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    DEFAULT_THUMBNAIL_DIR,
)
from csv_reader import load_all_creators  # noqa: E402
from models import ThumbnailData, VideoMetadata  # noqa: E402
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
from prompt_compiler import (  # noqa: E402
    InvalidRedesignSpecError,
    PromptPackageCacheError,
    compile_prompt_package,
    save_prompt_package,
)
from youtube_metadata import process_video  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    csv_path: Path = DEFAULT_CSV_PATH,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    redesign_spec_dir: Path = DEFAULT_REDESIGN_SPEC_DIR,
    prompt_package_dir: Path = DEFAULT_PROMPT_PACKAGE_DIR,
) -> None:
    """
    Execute the full four-module pipeline for every creator in ``csv_path``.

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
        prompt_package_dir: Directory where deterministic Module 6
                            prompt packages are saved.
    """
    logger.info("Pipeline starting — CSV: {csv}", csv=csv_path)

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

        # â”€â”€ Module 5: derive deterministic redesign specification â”€â”€
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

        # â”€â”€ Module 6: compile deterministic image-generation package â”€â”€
        try:
            prompt_package = compile_prompt_package(redesign_spec)
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
        succeeded += 1

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

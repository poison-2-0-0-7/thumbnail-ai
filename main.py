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
          analysis, and Gemini-based reasoning, saving a structured
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

from config import DEFAULT_ANALYSIS_DIR, DEFAULT_CSV_PATH, DEFAULT_THUMBNAIL_DIR  # noqa: E402
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
from youtube_metadata import process_video  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    csv_path: Path = DEFAULT_CSV_PATH,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
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

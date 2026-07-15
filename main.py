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

from config import DEFAULT_CSV_PATH, DEFAULT_THUMBNAIL_DIR  # noqa: E402
from csv_reader import load_all_creators  # noqa: E402
from models import ThumbnailData, VideoMetadata  # noqa: E402
from thumbnail_downloader import (  # noqa: E402
    ThumbnailDownloaderError,
    process_thumbnail,
)
from youtube_metadata import process_video  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    csv_path: Path = DEFAULT_CSV_PATH,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
) -> None:
    """
    Execute the full three-module pipeline for every creator in ``csv_path``.

    Processing is best-effort: a failure on one creator is logged and
    counted, but never prevents the remaining creators from being
    processed.

    Args:
        csv_path:      Path to the creators CSV file.
        thumbnail_dir: Directory where thumbnails are saved.
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
            "Processing creator_id={id} url={url}",
            id=creator.id,
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
                "Metadata extraction failed for creator_id={id}: {reason}",
                id=creator.id,
                reason=metadata.error_message,
            )
            skipped += 1
            continue

        logger.info(
            "Metadata OK for creator_id={id}: {title!r}",
            id=creator.id,
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
                "Thumbnail download failed for creator_id={id} "
                "video_id={vid}: {exc}",
                id=creator.id,
                vid=metadata.video_id,
                exc=exc,
            )
            skipped += 1
            continue

        logger.info(
            "Thumbnail saved for creator_id={id}: {path}",
            id=creator.id,
            path=thumbnail.thumbnail_path,
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

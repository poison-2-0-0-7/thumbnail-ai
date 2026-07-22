"""Verify read-only ComfyUI HTTP connectivity for Module 7 development.

Run from the repository root:
    python scripts/verify_comfyui_http.py
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import requests
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULES_DIR = _PROJECT_ROOT / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from comfyui_client import _ComfyUIHTTPError, _ComfyUIHTTPTransport  # noqa: E402
from config import (  # noqa: E402
    COMFYUI_HOST,
    COMFYUI_PORT,
    COMFYUI_REQUEST_TIMEOUT_SECONDS,
)
from module7_exceptions import ComfyUIConnectionError, Module7Error  # noqa: E402


def verify() -> tuple[dict[str, object], dict[str, object]]:
    """Read system and queue state from the configured local ComfyUI server."""
    base_url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
    with requests.Session() as session:
        transport = _ComfyUIHTTPTransport(
            base_url=base_url,
            session=session,
            timeout_seconds=COMFYUI_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            stats = transport.system_stats()
            queue = transport.queue_status()
        except _ComfyUIHTTPError as exc:
            raise ComfyUIConnectionError(
                f"Could not verify ComfyUI HTTP connectivity at {base_url}: {exc}"
            ) from exc
    return asdict(stats), queue


def main() -> int:
    """Run the read-only verification and return a shell-friendly exit code."""
    try:
        stats, queue = verify()
    except Module7Error as exc:
        logger.error("ComfyUI HTTP verification failed: {error}", error=str(exc))
        return 1
    except Exception as exc:  # Defensive boundary for a developer command.
        logger.exception("Unexpected ComfyUI HTTP verification failure: {error}", error=str(exc))
        return 1

    logger.info("ComfyUI HTTP verification succeeded")
    print("ComfyUI HTTP verification succeeded.")
    print(json.dumps({"system_stats": stats, "queue_status": queue}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

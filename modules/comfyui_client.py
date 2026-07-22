"""Synchronous HTTP transport for the local ComfyUI server.

This Sprint 1 module deliberately owns only ComfyUI's REST endpoints.  Queue
tracking, WebSocket events, output staging, and the public client facade are
implemented in later sprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from loguru import logger

from config import LOG_DIR, MODULE7_LOG_PATH

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """Attach the Module 7 rotating Loguru sink using project conventions."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE7_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


@dataclass(frozen=True)
class SystemStats:
    """Minimal, validated hardware and version data returned by ComfyUI."""

    vram_free_mb: float
    vram_total_mb: float
    device_name: str
    comfyui_version: str


class _ComfyUIHTTPError(RuntimeError):
    """Transport-local failure translated to public Module 7 errors by a later facade."""


class _ComfyUIHTTPTransport:
    """Small session-reusing wrapper around the ComfyUI HTTP API."""

    def __init__(
        self,
        base_url: str,
        session: requests.Session,
        timeout_seconds: float,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if not normalized_url:
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._base_url = normalized_url
        self._session = session
        self._timeout_seconds = timeout_seconds

    def system_stats(self) -> SystemStats:
        """Return validated local GPU capacity and ComfyUI version data."""
        payload = self._request_json("GET", "/system_stats")
        if not isinstance(payload, dict):
            raise _ComfyUIHTTPError("ComfyUI /system_stats response must be a JSON object")

        devices = payload.get("devices")
        system = payload.get("system")
        if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
            raise _ComfyUIHTTPError("ComfyUI /system_stats response has no valid device entry")
        if not isinstance(system, dict):
            raise _ComfyUIHTTPError("ComfyUI /system_stats response has no valid system entry")

        device = devices[0]
        vram_free = self._number(device.get("vram_free"), "devices[0].vram_free")
        vram_total = self._number(device.get("vram_total"), "devices[0].vram_total")
        device_name = self._text(device.get("name"), "devices[0].name")
        version = self._text(system.get("comfyui_version"), "system.comfyui_version")
        return SystemStats(
            vram_free_mb=vram_free,
            vram_total_mb=vram_total,
            device_name=device_name,
            comfyui_version=version,
        )

    def submit_prompt(self, graph: dict[str, Any], client_id: str) -> str:
        """Submit one already-materialized workflow and return its prompt ID."""
        payload = self._request_json(
            "POST",
            "/prompt",
            json_body={"prompt": graph, "client_id": client_id},
        )
        if not isinstance(payload, dict):
            raise _ComfyUIHTTPError("ComfyUI /prompt response must be a JSON object")
        return self._text(payload.get("prompt_id"), "prompt_id")

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        """Return ComfyUI history for one prompt, or ``None`` when it is absent."""
        payload = self._request_json("GET", f"/history/{prompt_id}")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise _ComfyUIHTTPError("ComfyUI history response must be a JSON object")
        return payload or None

    def view_image(self, filename: str, subfolder: str, image_type: str) -> bytes:
        """Fetch one completed ComfyUI image without interpreting its pixels."""
        response = self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": image_type},
        )
        return response.content

    def interrupt(self) -> None:
        """Request best-effort interruption of ComfyUI's currently running prompt."""
        self._request("POST", "/interrupt")

    def delete_from_queue(self, prompt_id: str) -> None:
        """Remove one prompt from ComfyUI's pending queue."""
        self._request("POST", "/queue", json_body={"delete": [prompt_id]})

    def queue_status(self) -> dict[str, Any]:
        """Return the unmodified queue payload for later queue tracking."""
        payload = self._request_json("GET", "/queue")
        if not isinstance(payload, dict):
            raise _ComfyUIHTTPError("ComfyUI /queue response must be a JSON object")
        return payload

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        response = self._request(method, path, params=params, json_body=json_body)
        try:
            return response.json()
        # ``requests`` may expose its JSON decoder exception differently by
        # version; it is always a ``ValueError`` at this boundary.
        except ValueError as exc:
            raise _ComfyUIHTTPError(
                f"ComfyUI returned malformed JSON for {method} {path}"
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = f"{self._base_url}{path}"
        logger.debug("ComfyUI HTTP request: {method} {url}", method=method, url=url)
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise _ComfyUIHTTPError(f"ComfyUI HTTP request failed for {method} {path}: {exc}") from exc
        return response

    @staticmethod
    def _number(value: object, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise _ComfyUIHTTPError(f"ComfyUI /system_stats response has invalid {field}")
        return float(value)

    @staticmethod
    def _text(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise _ComfyUIHTTPError(f"ComfyUI response has invalid {field}")
        return value


__all__ = ["SystemStats"]

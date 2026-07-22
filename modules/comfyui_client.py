"""Synchronous transports for the local ComfyUI server.

This module owns ComfyUI's HTTP REST endpoints and WebSocket event stream.
Queue tracking, output staging, and the public client facade are implemented
in later sprints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Literal
from urllib.parse import urlencode

import requests
import websocket
from loguru import logger

from config import LOG_DIR, MODULE7_LOG_PATH
from module7_exceptions import ComfyUIConnectionError

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


@dataclass(frozen=True)
class ComfyUIEvent:
    """Typed ComfyUI WebSocket event consumed by the future queue tracker."""

    event_type: Literal["status", "executing", "progress", "execution_error", "execution_cached"]
    prompt_id: str | None
    node: str | None
    progress_value: int | None
    progress_max: int | None
    queue_remaining: int | None
    error_payload: dict[str, Any] | None


class _ComfyUIHTTPError(RuntimeError):
    """Transport-local failure translated to public Module 7 errors by a later facade."""


class _ComfyUIWebSocketTransport:
    """Small read-only wrapper around ComfyUI's WebSocket event stream."""

    _STATE_DISCONNECTED = "Disconnected"
    _STATE_CONNECTING = "Connecting"
    _STATE_CONNECTED = "Connected"
    _STATE_RECEIVING = "Receiving"
    _STATE_CLOSED = "Closed"
    _STATE_FAILED = "Failed"
    _TRACKED_EVENT_TYPES = {
        "status",
        "executing",
        "progress",
        "execution_error",
        "execution_cached",
    }

    def __init__(self, ws_url: str, client_id: str, connect_timeout_seconds: float) -> None:
        normalized_url = ws_url.rstrip("/")
        if not normalized_url.startswith(("ws://", "wss://")):
            raise ValueError("ws_url must start with ws:// or wss://")
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be greater than zero")

        self._ws_url = normalized_url
        self._client_id = client_id.strip()
        self._connect_timeout_seconds = connect_timeout_seconds
        self._socket: websocket.WebSocket | None = None
        self._state = self._STATE_DISCONNECTED

    def ensure_connected(self) -> None:
        """Establish the WebSocket connection once, or reconnect after closure."""
        if self.is_connected():
            logger.debug("ComfyUI WebSocket already connected: {url}", url=self._ws_url)
            return

        connect_url = self._connect_url()
        previous_state = self._state
        self._state = self._STATE_CONNECTING
        started = time.monotonic()
        logger.debug("ComfyUI WebSocket connecting: {url}", url=connect_url)
        try:
            self._socket = websocket.create_connection(
                connect_url,
                timeout=self._connect_timeout_seconds,
            )
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            OSError,
            socket.timeout,
            TimeoutError,
        ) as exc:
            self._socket = None
            self._state = self._STATE_FAILED
            elapsed = time.monotonic() - started
            logger.error(
                "ComfyUI WebSocket connection failed after {elapsed:.2f}s: {url} ({error})",
                elapsed=elapsed,
                url=self._ws_url,
                error=str(exc),
            )
            raise ComfyUIConnectionError(
                f"Could not connect to ComfyUI WebSocket at {self._ws_url}: {exc}"
            ) from exc

        self._state = self._STATE_CONNECTED
        elapsed = time.monotonic() - started
        if previous_state == self._STATE_FAILED:
            logger.info(
                "ComfyUI WebSocket reconnected after failure: {url} in {elapsed:.2f}s",
                url=self._ws_url,
                elapsed=elapsed,
            )
        elif previous_state == self._STATE_CLOSED:
            logger.info(
                "ComfyUI WebSocket reconnected after close: {url} in {elapsed:.2f}s",
                url=self._ws_url,
                elapsed=elapsed,
            )
        else:
            logger.info(
                "ComfyUI WebSocket connected: {url} in {elapsed:.2f}s",
                url=self._ws_url,
                elapsed=elapsed,
            )

    def receive(self, timeout_seconds: float) -> str | None:
        """Return one raw text frame, or ``None`` for timeout/binary frames."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self._state == self._STATE_RECEIVING:
            raise RuntimeError("ComfyUI WebSocket receive is already in progress")
        if self._socket is None or not self.is_connected():
            raise ComfyUIConnectionError("ComfyUI WebSocket is not connected")

        socket_obj = self._socket
        self._state = self._STATE_RECEIVING
        try:
            socket_obj.settimeout(timeout_seconds)
            frame = socket_obj.recv()
        except websocket.WebSocketTimeoutException:
            logger.debug("ComfyUI WebSocket receive timed out after {timeout:.2f}s", timeout=timeout_seconds)
            if self._state != self._STATE_CLOSED:
                self._state = self._STATE_CONNECTED
            return None
        except (
            websocket.WebSocketConnectionClosedException,
            ConnectionResetError,
            BrokenPipeError,
            OSError,
        ) as exc:
            self._socket = None
            if self._state != self._STATE_CLOSED:
                self._state = self._STATE_FAILED
            logger.warning("ComfyUI WebSocket connection lost during receive: {error}", error=str(exc))
            raise ComfyUIConnectionError("ComfyUI WebSocket connection was closed") from exc

        if self._state != self._STATE_CLOSED:
            self._state = self._STATE_CONNECTED
        if isinstance(frame, bytes):
            logger.debug("ComfyUI WebSocket binary frame ignored: {size} bytes", size=len(frame))
            return None
        if not isinstance(frame, str):
            logger.debug("ComfyUI WebSocket non-text frame ignored: {type}", type=type(frame).__name__)
            return None

        logger.debug("ComfyUI WebSocket frame received: {frame}", frame=self._preview(frame))
        return frame

    def next_event(self, timeout_seconds: float) -> ComfyUIEvent | None:
        """Return the next actionable ComfyUI event, or ``None`` when skipped."""
        frame = self.receive(timeout_seconds)
        if frame is None:
            return None

        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            logger.warning("ComfyUI WebSocket frame is malformed JSON: {frame}", frame=self._preview(frame))
            return None

        if not isinstance(payload, dict):
            logger.warning("ComfyUI WebSocket frame is not a JSON object")
            return None

        event_type = payload.get("type")
        if not isinstance(event_type, str) or not event_type.strip():
            logger.warning("ComfyUI WebSocket frame has no valid type")
            return None

        if event_type not in self._TRACKED_EVENT_TYPES:
            logger.debug("ComfyUI WebSocket event ignored: {event_type}", event_type=event_type)
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            logger.warning(
                "ComfyUI WebSocket event has invalid data: {event_type}",
                event_type=event_type,
            )
            return None

        event = self._parse_event(event_type, data)
        if event is None:
            return None
        logger.debug(
            "ComfyUI WebSocket event parsed: {event_type} prompt_id={prompt_id}",
            event_type=event.event_type,
            prompt_id=event.prompt_id,
        )
        return event

    def is_connected(self) -> bool:
        """Return whether this instance believes the socket is currently open."""
        return self._state in {self._STATE_CONNECTED, self._STATE_RECEIVING}

    def close(self) -> None:
        """Close the WebSocket connection idempotently."""
        socket_obj = self._socket
        self._socket = None
        self._state = self._STATE_CLOSED
        if socket_obj is None:
            return

        try:
            socket_obj.close(timeout=1)
        except Exception as exc:  # Cleanup must never mask the caller's real outcome.
            logger.debug("ComfyUI WebSocket close failed during cleanup: {error}", error=str(exc))
            return
        logger.info("ComfyUI WebSocket closed: {url}", url=self._ws_url)

    def _connect_url(self) -> str:
        separator = "&" if "?" in self._ws_url else "?"
        return f"{self._ws_url}{separator}{urlencode({'clientId': self._client_id})}"

    def _parse_event(self, event_type: str, data: dict[str, Any]) -> ComfyUIEvent | None:
        if event_type == "status":
            return self._parse_status(data)
        if event_type == "executing":
            return self._parse_executing(data)
        if event_type == "progress":
            return self._parse_progress(data)
        if event_type == "execution_error":
            return self._parse_execution_error(data)
        if event_type == "execution_cached":
            return self._parse_execution_cached(data)
        return None

    def _parse_status(self, data: dict[str, Any]) -> ComfyUIEvent | None:
        status = data.get("status")
        if not isinstance(status, dict):
            logger.warning("ComfyUI status event has invalid status payload")
            return None
        exec_info = status.get("exec_info")
        if not isinstance(exec_info, dict):
            logger.warning("ComfyUI status event has invalid exec_info payload")
            return None
        queue_remaining = exec_info.get("queue_remaining")
        if not self._is_non_negative_int(queue_remaining):
            logger.warning("ComfyUI status event has invalid queue_remaining")
            return None
        return ComfyUIEvent(
            event_type="status",
            prompt_id=None,
            node=None,
            progress_value=None,
            progress_max=None,
            queue_remaining=queue_remaining,
            error_payload=None,
        )

    def _parse_executing(self, data: dict[str, Any]) -> ComfyUIEvent | None:
        prompt_id = self._prompt_id(data)
        if prompt_id is None:
            logger.warning("ComfyUI executing event has invalid prompt_id")
            return None
        node = data.get("node")
        if node is not None and not isinstance(node, str):
            logger.warning("ComfyUI executing event has invalid node")
            return None
        return ComfyUIEvent(
            event_type="executing",
            prompt_id=prompt_id,
            node=node,
            progress_value=None,
            progress_max=None,
            queue_remaining=None,
            error_payload=None,
        )

    def _parse_progress(self, data: dict[str, Any]) -> ComfyUIEvent | None:
        prompt_id = self._prompt_id(data)
        value = data.get("value")
        maximum = data.get("max")
        node = data.get("node")
        if prompt_id is None:
            logger.warning("ComfyUI progress event has invalid prompt_id")
            return None
        if not self._is_non_negative_int(value) or not self._is_non_negative_int(maximum) or value > maximum:
            logger.warning("ComfyUI progress event has invalid progress values")
            return None
        if not isinstance(node, str) or not node.strip():
            logger.warning("ComfyUI progress event has invalid node")
            return None
        return ComfyUIEvent(
            event_type="progress",
            prompt_id=prompt_id,
            node=node,
            progress_value=value,
            progress_max=maximum,
            queue_remaining=None,
            error_payload=None,
        )

    def _parse_execution_error(self, data: dict[str, Any]) -> ComfyUIEvent | None:
        prompt_id = self._prompt_id(data)
        if prompt_id is None:
            logger.warning("ComfyUI execution_error event has invalid prompt_id")
            return None
        return ComfyUIEvent(
            event_type="execution_error",
            prompt_id=prompt_id,
            node=None,
            progress_value=None,
            progress_max=None,
            queue_remaining=None,
            error_payload=data,
        )

    def _parse_execution_cached(self, data: dict[str, Any]) -> ComfyUIEvent | None:
        prompt_id = self._prompt_id(data)
        if prompt_id is None:
            logger.warning("ComfyUI execution_cached event has invalid prompt_id")
            return None
        return ComfyUIEvent(
            event_type="execution_cached",
            prompt_id=prompt_id,
            node=None,
            progress_value=None,
            progress_max=None,
            queue_remaining=None,
            error_payload=None,
        )

    @staticmethod
    def _prompt_id(data: dict[str, Any]) -> str | None:
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            return None
        return prompt_id

    @staticmethod
    def _is_non_negative_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _preview(value: str, limit: int = 200) -> str:
        if len(value) <= limit:
            return value
        return f"{value[:limit]}..."


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
        if not payload:
            return None
        if prompt_id in payload:
            history_entry = payload[prompt_id]
            if not isinstance(history_entry, dict):
                raise _ComfyUIHTTPError(
                    f"ComfyUI history response has invalid entry for prompt_id {prompt_id}"
                )
            return history_entry
        return payload

    def view_image(self, filename: str, subfolder: str, image_type: str) -> bytes:
        """Fetch one completed ComfyUI image without interpreting its pixels."""
        response = self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": image_type},
        )
        if not isinstance(response.content, bytes) or not response.content:
            raise _ComfyUIHTTPError("ComfyUI /view response did not contain image bytes")
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

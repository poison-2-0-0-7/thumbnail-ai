"""Synchronous transports for the local ComfyUI server.

This module owns ComfyUI's HTTP REST endpoints and WebSocket event stream.
Queue tracking, output staging, and the public client facade are implemented
in later sprints.
"""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from enum import Enum
import json
import socket
import time
from uuid import uuid4
from typing import Any, ClassVar, Literal, Sequence
from urllib.parse import urlencode

from PIL import Image, UnidentifiedImageError
import requests
import websocket
from loguru import logger

from config import (
    COMFYUI_HOST,
    COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS,
    COMFYUI_HISTORY_CONFIRMATION_RETRY_DELAY_SECONDS,
    COMFYUI_EXECUTION_TIMEOUT_SECONDS,
    COMFYUI_OUTPUT_DOWNLOAD_MAX_RETRIES,
    COMFYUI_OUTPUT_DOWNLOAD_RETRY_BACKOFF_SECONDS,
    COMFYUI_OUTPUT_DOWNLOAD_TIMEOUT_SECONDS,
    COMFYUI_OUTPUT_PREFERRED_NODES,
    COMFYUI_OUTPUT_SUPPORTED_IMAGE_FORMATS,
    COMFYUI_POLL_INTERVAL_SECONDS,
    COMFYUI_PORT,
    COMFYUI_REQUEST_TIMEOUT_SECONDS,
    COMFYUI_STARTUP_TIMEOUT_SECONDS,
    COMFYUI_WEBSOCKET_TIMEOUT_SECONDS,
    COMFYUI_WS_PATH,
    COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS,
    COMFYUI_WS_RECONNECT_POLL_CYCLES,
    LOG_DIR,
    MODULE7_LOG_PATH,
    MODULE7_METRICS_PATH,
    MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT,
    MODULE7_STILL_QUEUED_WARNING_SECONDS,
)
from image_generator import BuiltWorkflow, MetricsCollector, utc_now
from module7_exceptions import (
    ComfyUIConnectionError,
    ComfyUIQueueError,
    ComfyUITimeoutError,
    CorruptImageError,
    MissingOutputFileError,
    NoOutputImageError,
    OutputDownloadError,
    OutputHistoryError,
    UnsupportedImageFormatError,
    VRAMExhaustedError,
)
from models import GenerationMetrics

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


class _CompletionOutcome(Enum):
    COMPLETED = "completed"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class _CompletionResult:
    outcome: _CompletionOutcome
    history_payload: dict[str, Any] | None
    error_payload: dict[str, Any] | None
    queue_wait_seconds: float
    generation_seconds: float
    used_http_fallback: bool


@dataclass(frozen=True)
class _ImageCandidate:
    output_node_id: str
    filename: str
    subfolder: str
    image_type: str
    format: str


@dataclass(frozen=True)
class _OutputResult:
    prompt_id: str
    output_node_id: str
    filename: str
    subfolder: str
    image_type: str
    format: str
    content: bytes
    width: int | None
    height: int | None


class _PhaseOutcome(Enum):
    COMPLETED = "completed"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"
    RESUME_WEBSOCKET = "resume_websocket"


@dataclass(frozen=True)
class _PhaseResult:
    outcome: _PhaseOutcome
    history_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None


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


class _QueueTracker:
    """Track one submitted ComfyUI prompt through terminal completion."""

    _STATE_QUEUED = "Queued"
    _STATE_EXECUTING = "Executing"

    def __init__(
        self,
        http: "_ComfyUIHTTPTransport",
        ws: "_ComfyUIWebSocketTransport",
        poll_interval_seconds: float = COMFYUI_POLL_INTERVAL_SECONDS,
        execution_timeout_seconds: float = COMFYUI_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        if execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be greater than zero")
        if poll_interval_seconds >= execution_timeout_seconds:
            raise ValueError("poll_interval_seconds must be less than execution_timeout_seconds")

        self._http = http
        self._ws = ws
        self._poll_interval_seconds = poll_interval_seconds
        self._execution_timeout_seconds = execution_timeout_seconds

        self._call_started_at = 0.0
        self._execution_started_at: float | None = None
        self._last_terminal_at = 0.0
        self._state = self._STATE_QUEUED
        self._used_http_fallback = False
        self._queued_warning_logged = False
        self._last_progress_milestone = 0

    def await_completion(self, prompt_id: str, client_id: str) -> _CompletionResult:
        """Block until the prompt completes, errors, or exceeds its execution deadline."""
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            return self._timeout_result(0.0)
        if not isinstance(client_id, str) or not client_id.strip():
            logger.debug("Queue tracker received empty client_id for prompt_id={prompt_id}", prompt_id=prompt_id)

        self._reset_call_state()
        deadline = self._call_started_at + self._execution_timeout_seconds

        phase = self._run_websocket_phase(prompt_id.strip(), deadline)
        while phase.outcome is _PhaseOutcome.FALLBACK:
            self._used_http_fallback = True
            phase = self._run_polling_phase(prompt_id.strip(), deadline)
            if phase.outcome is _PhaseOutcome.RESUME_WEBSOCKET:
                phase = self._run_websocket_phase(prompt_id.strip(), deadline)

        return self._completion_result(phase)

    def _reset_call_state(self) -> None:
        self._call_started_at = time.monotonic()
        self._execution_started_at = None
        self._last_terminal_at = self._call_started_at
        self._state = self._STATE_QUEUED
        self._used_http_fallback = False
        self._queued_warning_logged = False
        self._last_progress_milestone = 0

    def _run_websocket_phase(self, prompt_id: str, deadline: float) -> _PhaseResult:
        while True:
            if self._deadline_reached(deadline):
                return self._phase_timeout()

            timeout_seconds = min(COMFYUI_WEBSOCKET_TIMEOUT_SECONDS, self._remaining(deadline))
            if timeout_seconds <= 0:
                return self._phase_timeout()

            try:
                event = self._ws.next_event(timeout_seconds)
            except ComfyUIConnectionError as exc:
                self._used_http_fallback = True
                logger.warning(
                    "ComfyUI WebSocket disconnected while tracking prompt_id={prompt_id}; "
                    "falling back to HTTP polling ({error})",
                    prompt_id=prompt_id,
                    error=str(exc),
                )
                return _PhaseResult(_PhaseOutcome.FALLBACK)

            if event is None:
                self._log_still_queued_if_needed(prompt_id)
                continue

            result = self._handle_websocket_event(prompt_id, event, deadline)
            if result is not None:
                return result

    def _run_polling_phase(self, prompt_id: str, deadline: float) -> _PhaseResult:
        poll_count = 0
        while True:
            if self._deadline_reached(deadline):
                return self._phase_timeout()

            phase_result = self._poll_history_once(prompt_id)
            if phase_result is not None:
                return phase_result

            poll_count += 1
            if self._should_attempt_reconnect(poll_count, deadline):
                reconnect_result = self._attempt_websocket_reconnect(prompt_id)
                if reconnect_result is not None:
                    return reconnect_result

            if not self._sleep_until_next_poll(deadline):
                return self._phase_timeout()

    def _handle_websocket_event(
        self,
        prompt_id: str,
        event: ComfyUIEvent,
        deadline: float,
    ) -> _PhaseResult | None:
        if event.event_type == "status":
            self._log_still_queued_if_needed(prompt_id)
            logger.debug(
                "ComfyUI queue status observed while tracking prompt_id={prompt_id}: "
                "queue_remaining={queue_remaining}",
                prompt_id=prompt_id,
                queue_remaining=event.queue_remaining,
            )
            return None

        if event.prompt_id != prompt_id:
            logger.debug(
                "Ignored ComfyUI event for foreign prompt_id={foreign_prompt_id} while tracking prompt_id={prompt_id}",
                foreign_prompt_id=event.prompt_id,
                prompt_id=prompt_id,
            )
            return None

        if event.event_type == "executing":
            if event.node is None:
                terminal_at = time.monotonic()
                self._mark_direct_terminal_if_queued(terminal_at)
                history_payload = self._confirm_completion(prompt_id, deadline)
                if history_payload is None:
                    logger.warning(
                        "ComfyUI history confirmation missing for prompt_id={prompt_id}; "
                        "falling back to HTTP polling",
                        prompt_id=prompt_id,
                    )
                    self._used_http_fallback = True
                    return _PhaseResult(_PhaseOutcome.FALLBACK)
                outcome = self._classify_history_status(history_payload)
                if outcome is _CompletionOutcome.EXECUTION_ERROR:
                    logger.warning(
                        "ComfyUI completion signal resolved to execution error in history: prompt_id={prompt_id}",
                        prompt_id=prompt_id,
                    )
                    return self._phase_execution_error(self._history_error_payload(history_payload), terminal_at)
                if outcome is _CompletionOutcome.COMPLETED:
                    return self._phase_completed(history_payload, terminal_at)
                self._used_http_fallback = True
                return _PhaseResult(_PhaseOutcome.FALLBACK)

            self._mark_executing(prompt_id)
            logger.debug(
                "ComfyUI executing node observed: prompt_id={prompt_id} node={node}",
                prompt_id=prompt_id,
                node=event.node,
            )
            return None

        if event.event_type == "progress":
            self._mark_executing(prompt_id)
            self._log_progress(prompt_id, event)
            return None

        if event.event_type == "execution_error":
            terminal_at = time.monotonic()
            self._mark_direct_terminal_if_queued(terminal_at)
            return self._phase_execution_error(event.error_payload or {}, terminal_at)

        if event.event_type == "execution_cached":
            logger.debug("ComfyUI execution cache event observed: prompt_id={prompt_id}", prompt_id=prompt_id)
            return None

        return None

    def _poll_history_once(self, prompt_id: str) -> _PhaseResult | None:
        try:
            history_payload = self._http.history(prompt_id)
        except _ComfyUIHTTPError as exc:
            logger.warning(
                "ComfyUI history poll failed for prompt_id={prompt_id}: {error}",
                prompt_id=prompt_id,
                error=str(exc),
            )
            return None

        if history_payload is None:
            logger.debug("ComfyUI history poll missing for prompt_id={prompt_id}", prompt_id=prompt_id)
            self._log_still_queued_if_needed(prompt_id)
            return None

        outcome = self._classify_history_status(history_payload)
        if outcome is _CompletionOutcome.COMPLETED:
            return self._phase_completed(history_payload, time.monotonic())
        if outcome is _CompletionOutcome.EXECUTION_ERROR:
            return self._phase_execution_error(self._history_error_payload(history_payload), time.monotonic())

        if self._history_indicates_running(history_payload):
            self._mark_executing(prompt_id)
        logger.debug("ComfyUI history poll non-terminal for prompt_id={prompt_id}", prompt_id=prompt_id)
        return None

    def _confirm_completion(self, prompt_id: str, deadline: float) -> dict[str, Any] | None:
        attempts = max(1, COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            if self._deadline_reached(deadline):
                return None
            try:
                history_payload = self._http.history(prompt_id)
            except _ComfyUIHTTPError as exc:
                logger.warning(
                    "ComfyUI history confirmation attempt {attempt} failed for prompt_id={prompt_id}: {error}",
                    attempt=attempt,
                    prompt_id=prompt_id,
                    error=str(exc),
                )
                history_payload = None

            if history_payload is not None and self._classify_history_status(history_payload) is not None:
                return history_payload

            if attempt < attempts:
                delay = min(COMFYUI_HISTORY_CONFIRMATION_RETRY_DELAY_SECONDS, self._remaining(deadline))
                if delay <= 0:
                    return None
                time.sleep(delay)

        return None

    def _classify_history_status(self, history_entry: dict[str, Any]) -> _CompletionOutcome | None:
        status = history_entry.get("status")
        if not isinstance(status, dict):
            return None
        completed = status.get("completed")
        status_str = status.get("status_str")
        if completed is True and status_str == "success":
            return _CompletionOutcome.COMPLETED
        if completed is True and status_str == "error":
            return _CompletionOutcome.EXECUTION_ERROR
        return None

    def _history_indicates_running(self, history_entry: dict[str, Any]) -> bool:
        status = history_entry.get("status")
        if not isinstance(status, dict):
            return False
        return status.get("completed") is False and isinstance(status.get("status_str"), str)

    def _history_error_payload(self, history_entry: dict[str, Any]) -> dict[str, Any]:
        status = history_entry.get("status")
        if not isinstance(status, dict):
            return {}
        messages = status.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if (
                    isinstance(message, (list, tuple))
                    and len(message) >= 2
                    and message[0] == "execution_error"
                    and isinstance(message[1], dict)
                ):
                    return message[1]
        return {"status": status}

    def _should_attempt_reconnect(self, poll_count: int, deadline: float) -> bool:
        if COMFYUI_WS_RECONNECT_POLL_CYCLES <= 0:
            return False
        if poll_count % COMFYUI_WS_RECONNECT_POLL_CYCLES != 0:
            return False
        if self._remaining(deadline) <= COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS:
            logger.debug("Skipping ComfyUI WebSocket reconnect attempt because execution budget is nearly exhausted")
            return False
        return True

    def _attempt_websocket_reconnect(self, prompt_id: str) -> _PhaseResult | None:
        logger.debug("Attempting ComfyUI WebSocket reconnect while polling prompt_id={prompt_id}", prompt_id=prompt_id)
        try:
            self._ws.ensure_connected()
        except ComfyUIConnectionError as exc:
            logger.debug(
                "ComfyUI WebSocket reconnect attempt failed while polling prompt_id={prompt_id}: {error}",
                prompt_id=prompt_id,
                error=str(exc),
            )
            return None

        logger.debug("ComfyUI WebSocket reconnected while polling prompt_id={prompt_id}", prompt_id=prompt_id)
        immediate_result = self._poll_history_once(prompt_id)
        if immediate_result is not None:
            return immediate_result
        logger.debug("Resuming ComfyUI WebSocket tracking for prompt_id={prompt_id}", prompt_id=prompt_id)
        return _PhaseResult(_PhaseOutcome.RESUME_WEBSOCKET)

    def _mark_executing(self, prompt_id: str) -> None:
        if self._state == self._STATE_EXECUTING:
            return
        now = time.monotonic()
        self._execution_started_at = now
        self._state = self._STATE_EXECUTING
        logger.info(
            "ComfyUI prompt transitioned to executing: prompt_id={prompt_id} queue_wait_seconds={seconds:.2f}",
            prompt_id=prompt_id,
            seconds=now - self._call_started_at,
        )

    def _mark_direct_terminal_if_queued(self, terminal_at: float) -> None:
        if self._state == self._STATE_QUEUED:
            self._last_terminal_at = terminal_at
        elif self._execution_started_at is not None:
            self._last_terminal_at = terminal_at

    def _phase_completed(self, history_payload: dict[str, Any], terminal_at: float) -> _PhaseResult:
        self._last_terminal_at = terminal_at
        logger.info(
            "ComfyUI prompt completed in {seconds:.2f}s; used_http_fallback={fallback}",
            seconds=self._generation_seconds(terminal_at),
            fallback=self._used_http_fallback,
        )
        return _PhaseResult(_PhaseOutcome.COMPLETED, history_payload=history_payload)

    def _phase_execution_error(self, error_payload: dict[str, Any], terminal_at: float) -> _PhaseResult:
        self._last_terminal_at = terminal_at
        logger.info(
            "ComfyUI prompt reached execution error in {seconds:.2f}s; used_http_fallback={fallback}",
            seconds=self._generation_seconds(terminal_at),
            fallback=self._used_http_fallback,
        )
        return _PhaseResult(_PhaseOutcome.EXECUTION_ERROR, error_payload=error_payload)

    def _phase_timeout(self) -> _PhaseResult:
        self._last_terminal_at = time.monotonic()
        logger.warning(
            "ComfyUI prompt tracking timed out after {seconds:.2f}s; used_http_fallback={fallback}",
            seconds=self._last_terminal_at - self._call_started_at,
            fallback=self._used_http_fallback,
        )
        return _PhaseResult(_PhaseOutcome.TIMEOUT)

    def _completion_result(self, phase: _PhaseResult) -> _CompletionResult:
        if phase.outcome is _PhaseOutcome.COMPLETED:
            outcome = _CompletionOutcome.COMPLETED
        elif phase.outcome is _PhaseOutcome.EXECUTION_ERROR:
            outcome = _CompletionOutcome.EXECUTION_ERROR
        else:
            outcome = _CompletionOutcome.TIMEOUT

        return _CompletionResult(
            outcome=outcome,
            history_payload=phase.history_payload if outcome is _CompletionOutcome.COMPLETED else None,
            error_payload=phase.error_payload if outcome is _CompletionOutcome.EXECUTION_ERROR else None,
            queue_wait_seconds=self._queue_wait_seconds(self._last_terminal_at),
            generation_seconds=self._generation_seconds(self._last_terminal_at),
            used_http_fallback=self._used_http_fallback,
        )

    def _timeout_result(self, started_at: float) -> _CompletionResult:
        return _CompletionResult(
            outcome=_CompletionOutcome.TIMEOUT,
            history_payload=None,
            error_payload=None,
            queue_wait_seconds=0.0,
            generation_seconds=0.0,
            used_http_fallback=False,
        )

    def _queue_wait_seconds(self, reference_time: float) -> float:
        if self._execution_started_at is None:
            return max(0.0, reference_time - self._call_started_at)
        return max(0.0, self._execution_started_at - self._call_started_at)

    def _generation_seconds(self, reference_time: float) -> float:
        if self._execution_started_at is None:
            return 0.0
        return max(0.0, reference_time - self._execution_started_at)

    def _log_still_queued_if_needed(self, prompt_id: str) -> None:
        if self._state != self._STATE_QUEUED or self._queued_warning_logged:
            return
        elapsed = time.monotonic() - self._call_started_at
        if elapsed >= MODULE7_STILL_QUEUED_WARNING_SECONDS:
            self._queued_warning_logged = True
            logger.warning(
                "ComfyUI prompt still queued after {seconds:.2f}s: prompt_id={prompt_id}",
                seconds=elapsed,
                prompt_id=prompt_id,
            )

    def _log_progress(self, prompt_id: str, event: ComfyUIEvent) -> None:
        if event.progress_value is None or event.progress_max is None or event.progress_max <= 0:
            logger.debug("ComfyUI progress event observed: prompt_id={prompt_id}", prompt_id=prompt_id)
            return

        percent = int((event.progress_value / event.progress_max) * 100)
        granularity = max(1, MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT)
        milestone = (percent // granularity) * granularity
        if milestone > self._last_progress_milestone:
            self._last_progress_milestone = milestone
            logger.info(
                "ComfyUI progress milestone reached: prompt_id={prompt_id} progress={percent}%",
                prompt_id=prompt_id,
                percent=min(100, milestone),
            )
            return
        logger.debug(
            "ComfyUI progress event observed: prompt_id={prompt_id} progress={value}/{maximum}",
            prompt_id=prompt_id,
            value=event.progress_value,
            maximum=event.progress_max,
        )

    def _sleep_until_next_poll(self, deadline: float) -> bool:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            return False
        time.sleep(min(self._poll_interval_seconds, remaining))
        return not self._deadline_reached(deadline)

    @staticmethod
    def _remaining(deadline: float) -> float:
        return deadline - time.monotonic()

    @staticmethod
    def _deadline_reached(deadline: float) -> bool:
        return time.monotonic() >= deadline


class _OutputRetriever:
    """Parse completed ComfyUI history and retrieve one validated output image."""

    _PIL_FORMAT_EXTENSIONS = {
        "JPEG": {"jpg", "jpeg"},
        "PNG": {"png"},
        "WEBP": {"webp"},
        "GIF": {"gif"},
    }

    def __init__(
        self,
        transport: "_ComfyUIHTTPTransport",
        *,
        preferred_output_nodes: Sequence[str] | None = None,
        allowed_image_formats: Sequence[str] | None = None,
        download_timeout: float | None = None,
        download_retries: int | None = None,
        download_retry_backoff: float | None = None,
    ) -> None:
        if transport is None:
            raise ValueError("transport must not be None")
        if download_timeout is not None and download_timeout < 0:
            raise ValueError("download_timeout must not be negative")
        if download_retries is not None and download_retries < 0:
            raise ValueError("download_retries must not be negative")
        if download_retry_backoff is not None and download_retry_backoff < 0:
            raise ValueError("download_retry_backoff must not be negative")

        self._transport = transport
        self._preferred_output_nodes = tuple(str(node) for node in (
            preferred_output_nodes if preferred_output_nodes is not None else COMFYUI_OUTPUT_PREFERRED_NODES
        ))
        raw_formats = (
            allowed_image_formats
            if allowed_image_formats is not None
            else COMFYUI_OUTPUT_SUPPORTED_IMAGE_FORMATS
        )
        self._allowed_image_formats = frozenset(self._normalize_format(value) for value in raw_formats)
        self._download_timeout = (
            COMFYUI_OUTPUT_DOWNLOAD_TIMEOUT_SECONDS if download_timeout is None else download_timeout
        )
        self._download_retries = (
            COMFYUI_OUTPUT_DOWNLOAD_MAX_RETRIES if download_retries is None else download_retries
        )
        self._download_retry_backoff = (
            COMFYUI_OUTPUT_DOWNLOAD_RETRY_BACKOFF_SECONDS
            if download_retry_backoff is None
            else download_retry_backoff
        )

    def retrieve(
        self,
        completion: _CompletionResult,
        *,
        prompt_id: str | None = None,
    ) -> _OutputResult:
        """Return one downloaded and validated image from a successful completion."""
        resolved_prompt_id = self._resolve_prompt_id(completion, prompt_id)
        logger.info("Starting ComfyUI output retrieval: prompt_id={prompt_id}", prompt_id=resolved_prompt_id)

        history_payload = self._extract_history_payload(completion, resolved_prompt_id)
        candidates = self._collect_candidate_images(history_payload, resolved_prompt_id)
        selected = self._select_image(candidates, resolved_prompt_id)
        content = self._download_image(selected, resolved_prompt_id)
        width, height = self._validate_image_bytes(content, selected, resolved_prompt_id)

        logger.info(
            "ComfyUI output retrieval complete: prompt_id={prompt_id} node={node} "
            "filename={filename} width={width} height={height}",
            prompt_id=resolved_prompt_id,
            node=selected.output_node_id,
            filename=selected.filename,
            width=width,
            height=height,
        )
        return _OutputResult(
            prompt_id=resolved_prompt_id,
            output_node_id=selected.output_node_id,
            filename=selected.filename,
            subfolder=selected.subfolder,
            image_type=selected.image_type,
            format=selected.format,
            content=content,
            width=width,
            height=height,
        )

    def _resolve_prompt_id(self, completion: _CompletionResult, prompt_id: str | None) -> str:
        if completion is None:
            raise ValueError("completion must not be None")
        if completion.outcome is not _CompletionOutcome.COMPLETED:
            raise ValueError("completion must be successful before retrieving output")

        candidate = prompt_id
        if candidate is None:
            candidate = getattr(completion, "prompt_id", None)
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("prompt_id must be provided for output retrieval")
        return candidate.strip()

    def _extract_history_payload(
        self,
        completion: _CompletionResult,
        prompt_id: str,
    ) -> dict[str, Any]:
        history_payload = completion.history_payload
        source = "embedded"
        if history_payload is None:
            source = "fetched"
            try:
                history_payload = self._transport.history(prompt_id)
            except _ComfyUIHTTPError as exc:
                raise OutputHistoryError(
                    f"Could not fetch ComfyUI history for prompt_id {prompt_id}",
                    prompt_id=prompt_id,
                ) from exc

        logger.debug("ComfyUI output history payload source: {source}", source=source)
        if not isinstance(history_payload, dict):
            raise OutputHistoryError("ComfyUI history payload is malformed", prompt_id=prompt_id)
        if prompt_id in history_payload:
            entry = history_payload[prompt_id]
            if not isinstance(entry, dict):
                raise OutputHistoryError(
                    "ComfyUI history entry is malformed",
                    prompt_id=prompt_id,
                )
            return entry
        if "outputs" in history_payload:
            return history_payload
        raise OutputHistoryError(
            f"ComfyUI history payload does not contain prompt_id {prompt_id}",
            prompt_id=prompt_id,
        )

    def _collect_candidate_images(
        self,
        history_payload: dict[str, Any],
        prompt_id: str,
    ) -> list[_ImageCandidate]:
        outputs = history_payload.get("outputs")
        if not isinstance(outputs, dict):
            raise OutputHistoryError("ComfyUI history payload missing outputs", prompt_id=prompt_id)
        if not outputs:
            raise NoOutputImageError("ComfyUI history contains no output nodes", prompt_id=prompt_id)

        candidates: list[_ImageCandidate] = []
        for raw_node_id, node_payload in outputs.items():
            output_node_id = str(raw_node_id)
            if not isinstance(node_payload, dict):
                logger.debug(
                    "Skipping malformed ComfyUI output node: prompt_id={prompt_id} node={node}",
                    prompt_id=prompt_id,
                    node=output_node_id,
                )
                continue

            images = node_payload.get("images")
            if images is None:
                logger.debug(
                    "Skipping ComfyUI output node with no images: prompt_id={prompt_id} node={node}",
                    prompt_id=prompt_id,
                    node=output_node_id,
                )
                continue
            if not isinstance(images, list):
                logger.warning(
                    "Skipping ComfyUI output node with invalid images list: prompt_id={prompt_id} node={node}",
                    prompt_id=prompt_id,
                    node=output_node_id,
                )
                continue

            for image in images:
                candidate = self._candidate_from_image(output_node_id, image, prompt_id)
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            raise NoOutputImageError("ComfyUI history contains no output images", prompt_id=prompt_id)
        logger.debug(
            "Collected ComfyUI output candidates: prompt_id={prompt_id} count={count}",
            prompt_id=prompt_id,
            count=len(candidates),
        )
        return candidates

    def _select_image(self, candidates: list[_ImageCandidate], prompt_id: str | None = None) -> _ImageCandidate:
        filtered = [candidate for candidate in candidates if candidate.format in self._allowed_image_formats]
        if not filtered:
            raise NoOutputImageError("ComfyUI output images have no supported formats", prompt_id=prompt_id)

        for preferred_node in self._preferred_output_nodes:
            for candidate in filtered:
                if candidate.output_node_id == preferred_node:
                    self._log_selected(candidate, prompt_id, "preferred")
                    return candidate

        selected = filtered[0]
        self._log_selected(selected, prompt_id, "fallback")
        return selected

    def _download_image(self, candidate: _ImageCandidate, prompt_id: str) -> bytes:
        max_attempts = self._download_retries + 1
        last_error: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "Downloading ComfyUI output image: prompt_id={prompt_id} node={node} "
                "filename={filename} attempt={attempt}/{max_attempts}",
                prompt_id=prompt_id,
                node=candidate.output_node_id,
                filename=candidate.filename,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            try:
                content = self._transport.view_image(
                    candidate.filename,
                    candidate.subfolder,
                    candidate.image_type,
                )
            except _ComfyUIHTTPError as exc:
                if self._is_missing_file_error(exc):
                    logger.error(
                        "ComfyUI output file missing on server: prompt_id={prompt_id} "
                        "node={node} filename={filename}",
                        prompt_id=prompt_id,
                        node=candidate.output_node_id,
                        filename=candidate.filename,
                    )
                    raise MissingOutputFileError(
                        "ComfyUI output file missing on server",
                        prompt_id=prompt_id,
                        output_node_id=candidate.output_node_id,
                        filename=candidate.filename,
                    ) from exc
                last_error = exc
            else:
                if isinstance(content, bytes) and content:
                    logger.info(
                        "ComfyUI output image downloaded: prompt_id={prompt_id} "
                        "filename={filename} bytes={size}",
                        prompt_id=prompt_id,
                        filename=candidate.filename,
                        size=len(content),
                    )
                    return content
                last_error = OutputDownloadError(
                    "ComfyUI /view returned empty image bytes",
                    prompt_id=prompt_id,
                    output_node_id=candidate.output_node_id,
                    filename=candidate.filename,
                )

            if attempt < max_attempts:
                logger.warning(
                    "ComfyUI output download attempt failed; retrying: prompt_id={prompt_id} "
                    "filename={filename} attempt={attempt} error={error}",
                    prompt_id=prompt_id,
                    filename=candidate.filename,
                    attempt=attempt,
                    error=str(last_error),
                )
                time.sleep(self._download_retry_backoff * (2 ** (attempt - 1)))

        logger.error(
            "ComfyUI output download failed after {attempts} attempts: prompt_id={prompt_id} filename={filename}",
            attempts=max_attempts,
            prompt_id=prompt_id,
            filename=candidate.filename,
        )
        raise OutputDownloadError(
            "ComfyUI output download failed after retries",
            prompt_id=prompt_id,
            output_node_id=candidate.output_node_id,
            filename=candidate.filename,
        ) from last_error

    def _validate_image_bytes(
        self,
        content: bytes,
        candidate: _ImageCandidate,
        prompt_id: str | None = None,
    ) -> tuple[int | None, int | None]:
        if not isinstance(content, bytes) or not content:
            logger.error(
                "ComfyUI output image validation failed: prompt_id={prompt_id} filename={filename} reason=empty",
                prompt_id=prompt_id,
                filename=candidate.filename,
            )
            raise CorruptImageError(
                "ComfyUI output image payload is empty",
                prompt_id=prompt_id,
                output_node_id=candidate.output_node_id,
                filename=candidate.filename,
            )

        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                image_format = image.format
                width, height = image.size
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            logger.error(
                "ComfyUI output image validation failed: prompt_id={prompt_id} filename={filename} reason=decode",
                prompt_id=prompt_id,
                filename=candidate.filename,
            )
            raise CorruptImageError(
                "ComfyUI output image failed to decode",
                prompt_id=prompt_id,
                output_node_id=candidate.output_node_id,
                filename=candidate.filename,
            ) from exc

        decoded_formats = self._PIL_FORMAT_EXTENSIONS.get(str(image_format).upper(), set())
        if candidate.format not in decoded_formats or candidate.format not in self._allowed_image_formats:
            logger.error(
                "ComfyUI output image validation failed: prompt_id={prompt_id} "
                "filename={filename} reason=unsupported_format format={format}",
                prompt_id=prompt_id,
                filename=candidate.filename,
                format=image_format,
            )
            raise UnsupportedImageFormatError(
                "ComfyUI output image format is unsupported",
                prompt_id=prompt_id,
                output_node_id=candidate.output_node_id,
                filename=candidate.filename,
            )

        return int(width), int(height)

    def _candidate_from_image(
        self,
        output_node_id: str,
        image: object,
        prompt_id: str,
    ) -> _ImageCandidate | None:
        if not isinstance(image, dict):
            logger.debug(
                "Skipping malformed ComfyUI image entry: prompt_id={prompt_id} node={node}",
                prompt_id=prompt_id,
                node=output_node_id,
            )
            return None

        filename = image.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            logger.debug(
                "Skipping ComfyUI image entry with missing filename: prompt_id={prompt_id} node={node}",
                prompt_id=prompt_id,
                node=output_node_id,
            )
            return None

        subfolder = image.get("subfolder", "")
        image_type = image.get("type", "output")
        if not isinstance(subfolder, str):
            subfolder = ""
        if not isinstance(image_type, str) or not image_type.strip():
            image_type = "output"
        image_format = self._filename_format(filename)
        if image_format is None:
            logger.debug(
                "Skipping ComfyUI image with no extension: prompt_id={prompt_id} node={node} filename={filename}",
                prompt_id=prompt_id,
                node=output_node_id,
                filename=filename,
            )
            return None

        return _ImageCandidate(
            output_node_id=output_node_id,
            filename=filename.strip(),
            subfolder=subfolder,
            image_type=image_type.strip(),
            format=image_format,
        )

    def _log_selected(self, candidate: _ImageCandidate, prompt_id: str | None, via: str) -> None:
        logger.info(
            "Selected ComfyUI output image: prompt_id={prompt_id} node={node} "
            "filename={filename} format={format} via={via}",
            prompt_id=prompt_id,
            node=candidate.output_node_id,
            filename=candidate.filename,
            format=candidate.format,
            via=via,
        )

    def _is_missing_file_error(self, exc: _ComfyUIHTTPError) -> bool:
        cause = exc.__cause__
        response = getattr(cause, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code == 404:
            return True
        return "404" in str(exc)

    @classmethod
    def _filename_format(cls, filename: str) -> str | None:
        if "." not in filename:
            return None
        return cls._normalize_format(filename.rsplit(".", 1)[-1])

    @staticmethod
    def _normalize_format(value: object) -> str:
        return str(value).strip().lower().lstrip(".")


class _ComfyUIMetricsRecorder:
    """Translate one ComfyUI attempt outcome into the existing metrics sink."""

    _FAILURE_REASON_BY_EXCEPTION_TYPE: ClassVar[tuple[tuple[type[BaseException], str], ...]] = (
        (MissingOutputFileError, "missing_output_file"),
        (CorruptImageError, "corrupt_image"),
        (UnsupportedImageFormatError, "unsupported_image_format"),
        (NoOutputImageError, "no_output_image"),
        (OutputHistoryError, "output_history_error"),
        (OutputDownloadError, "output_download_error"),
        (VRAMExhaustedError, "vram_exhausted"),
        (ComfyUIQueueError, "queue_error"),
        (ComfyUIConnectionError, "connection_error"),
    )

    def __init__(self, collector: MetricsCollector) -> None:
        if collector is None:
            raise ValueError("collector must not be None")
        self._collector = collector

    def record_attempt(
        self,
        *,
        video_id: str,
        niche: str,
        workflow_version: str,
        profile_name: str | None = None,
        workflow_hash: str | None = None,
        completions: Sequence[_CompletionResult],
        output: _OutputResult | None = None,
        exception: BaseException | None = None,
        num_candidates_requested: int = 1,
        identity_retry_count: int = 0,
        peak_vram_mb: float | None = None,
        gpu_utilization_percent: float | None = None,
        attempt_started_at: float | None = None,
    ) -> None:
        """Append one GenerationMetrics record without masking generation outcome."""
        clean_video_id = self._required_text(video_id, "video_id")
        clean_niche = self._required_text(niche, "niche")
        clean_workflow_version = self._required_text(workflow_version, "workflow_version")
        if num_candidates_requested < 1:
            raise ValueError("num_candidates_requested must be at least 1")
        if identity_retry_count < 0:
            raise ValueError("identity_retry_count must not be negative")
        completion_list = self._validated_completions(completions)

        queue_time_seconds = sum(completion.queue_wait_seconds for completion in completion_list)
        generation_time_seconds = [completion.generation_seconds for completion in completion_list]
        total_duration_seconds = self._total_duration_seconds(completion_list, attempt_started_at)
        failure_reason = self._classify_failure(completion_list[-1], output, exception)

        try:
            metrics = GenerationMetrics(
                video_id=clean_video_id,
                niche=clean_niche,
                profile_name=profile_name,
                workflow_version=clean_workflow_version,
                workflow_hash=workflow_hash,
                num_candidates_requested=num_candidates_requested,
                queue_time_seconds=queue_time_seconds,
                generation_time_seconds=generation_time_seconds,
                total_duration_seconds=total_duration_seconds,
                identity_retry_count=identity_retry_count,
                generation_retry_count=len(completion_list) - 1,
                failure_reason=failure_reason,
                peak_vram_mb=peak_vram_mb,
                gpu_utilization_percent=gpu_utilization_percent,
                recorded_at=utc_now(),
            )
            self._collector.append(metrics)
            logger.debug(
                "ComfyUI metrics recorded for video_id={video_id} failure_reason={failure_reason}",
                video_id=clean_video_id,
                failure_reason=failure_reason,
            )
        except Exception as exc:
            logger.error(
                "ComfyUI metrics recording failed for video_id={video_id}: {error}",
                video_id=clean_video_id,
                error=str(exc),
            )

    def _classify_failure(
        self,
        completion: _CompletionResult,
        output: _OutputResult | None,
        exception: BaseException | None,
    ) -> str | None:
        if exception is not None:
            for exception_type, reason in self._FAILURE_REASON_BY_EXCEPTION_TYPE:
                if isinstance(exception, exception_type):
                    return reason
            logger.warning(
                "ComfyUI metrics saw unclassified exception type: {exception_type}",
                exception_type=type(exception).__name__,
            )
            return "unclassified_error"

        if completion.outcome is _CompletionOutcome.EXECUTION_ERROR:
            return "execution_error"
        if completion.outcome is _CompletionOutcome.TIMEOUT:
            return "timeout"
        if completion.outcome is _CompletionOutcome.COMPLETED and output is None:
            return "output_missing_uncaptured"
        return None

    @staticmethod
    def _required_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must not be empty")
        return value.strip()

    @staticmethod
    def _validated_completions(completions: Sequence[_CompletionResult]) -> tuple[_CompletionResult, ...]:
        if completions is None:
            raise ValueError("completions must not be empty")
        completion_list = tuple(completions)
        if not completion_list:
            raise ValueError("completions must not be empty")
        for completion in completion_list:
            if not isinstance(completion, _CompletionResult):
                raise ValueError("completions must contain _CompletionResult values")
            if completion.queue_wait_seconds < 0:
                raise ValueError("queue_wait_seconds must not be negative")
            if completion.generation_seconds < 0:
                raise ValueError("generation_seconds must not be negative")
        return completion_list

    @staticmethod
    def _total_duration_seconds(
        completions: Sequence[_CompletionResult],
        attempt_started_at: float | None,
    ) -> float:
        if attempt_started_at is None:
            return sum(
                completion.queue_wait_seconds + completion.generation_seconds
                for completion in completions
            )
        return max(0.0, time.monotonic() - attempt_started_at)


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


class ComfyUIClient:
    """Thin synchronous facade around the completed ComfyUI collaborators."""

    _OOM_SIGNATURES: ClassVar[tuple[str, ...]] = (
        "outofmemoryerror",
        "cuda out of memory",
        "out of memory",
    )

    def __init__(
        self,
        *,
        host: str = COMFYUI_HOST,
        port: int = COMFYUI_PORT,
        client_id: str | None = None,
        session: requests.Session | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        clean_host = str(host).strip()
        if not clean_host:
            raise ValueError("host must not be empty")
        if port <= 0:
            raise ValueError("port must be greater than zero")

        self._client_id = (client_id or str(uuid4())).strip()
        if not self._client_id:
            raise ValueError("client_id must not be empty")

        base_url = f"http://{clean_host}:{port}"
        ws_url = f"ws://{clean_host}:{port}{COMFYUI_WS_PATH}"
        self._session = session or requests.Session()
        self._http = _ComfyUIHTTPTransport(
            base_url,
            self._session,
            COMFYUI_REQUEST_TIMEOUT_SECONDS,
        )
        self._ws = _ComfyUIWebSocketTransport(
            ws_url,
            self._client_id,
            COMFYUI_STARTUP_TIMEOUT_SECONDS,
        )
        self._queue_tracker = _QueueTracker(self._http, self._ws)
        self._output_retriever = _OutputRetriever(self._http)
        self._metrics_recorder = _ComfyUIMetricsRecorder(
            metrics_collector or MetricsCollector(MODULE7_METRICS_PATH)
        )

    def generate(
        self,
        built_workflow: BuiltWorkflow,
        *,
        video_id: str,
        num_candidates_requested: int = 1,
        identity_retry_count: int = 0,
        peak_vram_mb: float | None = None,
        gpu_utilization_percent: float | None = None,
    ) -> _OutputResult:
        """Submit, await, retrieve, and record one ComfyUI generation attempt."""
        clean_video_id = self._required_video_id(video_id)
        attempt_started_at = time.monotonic()
        completion: _CompletionResult | None = None
        output: _OutputResult | None = None
        exception: BaseException | None = None

        logger.info("Starting ComfyUI generation for video_id={video_id}", video_id=clean_video_id)
        try:
            self._ws.ensure_connected()
            try:
                prompt_id = self._http.submit_prompt(built_workflow.graph, self._client_id)
            except _ComfyUIHTTPError as exc:
                raise ComfyUIConnectionError(f"ComfyUI prompt submission failed: {exc}") from exc

            completion = self._queue_tracker.await_completion(prompt_id, self._client_id)
            if completion.outcome is _CompletionOutcome.TIMEOUT:
                self._best_effort_cancel(prompt_id)
                raise ComfyUITimeoutError(
                    f"ComfyUI generation timed out for prompt_id={prompt_id}"
                )

            if completion.outcome is _CompletionOutcome.EXECUTION_ERROR:
                message = self._error_message(completion.error_payload)
                logger.error(
                    "ComfyUI generation failed for prompt_id={prompt_id}: {error}",
                    prompt_id=prompt_id,
                    error=message,
                )
                if self._is_oom(completion.error_payload):
                    raise VRAMExhaustedError(message)
                raise ComfyUIQueueError(message)

            output = self._output_retriever.retrieve(completion, prompt_id=prompt_id)
            logger.info(
                "ComfyUI generation completed for prompt_id={prompt_id}: "
                "queue_wait={queue_wait:.2f}s generation={generation:.2f}s "
                "http_fallback={http_fallback}",
                prompt_id=prompt_id,
                queue_wait=completion.queue_wait_seconds,
                generation=completion.generation_seconds,
                http_fallback=completion.used_http_fallback,
            )
            return output
        except BaseException as exc:
            exception = exc
            raise
        finally:
            self._metrics_recorder.record_attempt(
                video_id=clean_video_id,
                niche=built_workflow.workflow_ref.niche,
                workflow_version=built_workflow.workflow_ref.workflow_version,
                profile_name=built_workflow.workflow_ref.profile_name,
                workflow_hash=built_workflow.workflow_hash,
                completions=(completion if completion is not None else self._synthetic_completion(),),
                output=output,
                exception=exception,
                num_candidates_requested=num_candidates_requested,
                identity_retry_count=identity_retry_count,
                peak_vram_mb=peak_vram_mb,
                gpu_utilization_percent=gpu_utilization_percent,
                attempt_started_at=attempt_started_at,
            )

    def _best_effort_cancel(self, prompt_id: str) -> None:
        for action in (self._http.interrupt, lambda: self._http.delete_from_queue(prompt_id)):
            try:
                action()
            except _ComfyUIHTTPError as exc:
                logger.warning(
                    "ComfyUI best-effort cancellation failed for prompt_id={prompt_id}: {error}",
                    prompt_id=prompt_id,
                    error=str(exc),
                )

    def _is_oom(self, error_payload: dict[str, Any] | None) -> bool:
        if not error_payload:
            return False
        haystack = " ".join(
            str(error_payload.get(field, ""))
            for field in ("exception_type", "exception_message")
        ).lower()
        return any(signature in haystack for signature in self._OOM_SIGNATURES)

    @staticmethod
    def _error_message(error_payload: dict[str, Any] | None) -> str:
        if not isinstance(error_payload, dict):
            return "ComfyUI execution error"
        message = error_payload.get("exception_message")
        if not isinstance(message, str) or not message.strip():
            return "ComfyUI execution error"
        return message.strip()

    @staticmethod
    def _required_video_id(video_id: str) -> str:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must not be empty")
        return video_id.strip()

    @staticmethod
    def _synthetic_completion() -> _CompletionResult:
        return _CompletionResult(
            outcome=_CompletionOutcome.EXECUTION_ERROR,
            history_payload=None,
            error_payload=None,
            queue_wait_seconds=0.0,
            generation_seconds=0.0,
            used_http_fallback=False,
        )


__all__ = ["SystemStats", "ComfyUIClient"]

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import sys
import threading
import time
from unittest.mock import Mock

from PIL import Image
import pytest
import requests
import websocket

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from comfyui_client import (  # noqa: E402
    ComfyUIClient,
    ComfyUIEvent,
    SystemStats,
    _CompletionResult,
    _CompletionOutcome,
    _ComfyUIHTTPError,
    _ComfyUIHTTPTransport,
    _ComfyUIMetricsRecorder,
    _ComfyUIWebSocketTransport,
    _ImageCandidate,
    _OutputRetriever,
    _OutputResult,
    _QueueTracker,
)
from image_generator import BuiltWorkflow  # noqa: E402
from module7_exceptions import (  # noqa: E402
    ComfyUIConnectionError,
    ComfyUIQueueError,
    ComfyUITimeoutError,
    CorruptImageError,
    MetricsWriteError,
    MissingOutputFileError,
    NoOutputImageError,
    OutputDownloadError,
    OutputHistoryError,
    UnsupportedImageFormatError,
    VRAMExhaustedError,
)
from models import WorkflowTemplateRef  # noqa: E402


BASE_URL = "http://127.0.0.1:8188"
TIMEOUT_SECONDS = 12.5
WS_URL = "ws://127.0.0.1:8188/ws"
CLIENT_ID = "client-abc"
CONNECT_TIMEOUT_SECONDS = 0.25


def _response(
    *,
    payload: object | None = None,
    json_error: Exception | None = None,
    request_error: requests.RequestException | None = None,
    content: bytes = b"image-bytes",
) -> Mock:
    response = Mock()
    response.content = content
    if request_error is None:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = request_error
    if json_error is None:
        response.json.return_value = payload
    else:
        response.json.side_effect = json_error
    return response


@pytest.fixture
def session() -> Mock:
    return Mock(spec=requests.Session)


@pytest.fixture
def transport(session: Mock) -> _ComfyUIHTTPTransport:
    return _ComfyUIHTTPTransport(f"{BASE_URL}/", session, TIMEOUT_SECONDS)


def _assert_request(session: Mock, **kwargs: object) -> None:
    session.request.assert_called_once_with(timeout=TIMEOUT_SECONDS, **kwargs)


def test_system_stats_returns_validated_stats_and_reuses_session(
    session: Mock, transport: _ComfyUIHTTPTransport
) -> None:
    session.request.return_value = _response(payload={
        "system": {"comfyui_version": "v0.3.12"},
        "devices": [{"name": "NVIDIA RTX", "vram_free": 6144, "vram_total": 8192}],
    })

    assert transport.system_stats() == SystemStats(6144.0, 8192.0, "NVIDIA RTX", "v0.3.12")
    _assert_request(
        session,
        method="GET",
        url=f"{BASE_URL}/system_stats",
        params=None,
        json=None,
    )

    session.request.reset_mock()
    session.request.return_value = _response(payload={"queue_running": [], "queue_pending": []})
    assert transport.queue_status() == {"queue_running": [], "queue_pending": []}
    _assert_request(session, method="GET", url=f"{BASE_URL}/queue", params=None, json=None)


def test_submit_prompt_posts_exact_comfyui_payload(session: Mock, transport: _ComfyUIHTTPTransport) -> None:
    graph = {"1": {"class_type": "KSampler", "inputs": {}}}
    session.request.return_value = _response(payload={"prompt_id": "prompt-123", "number": 4})

    assert transport.submit_prompt(graph, "client-456") == "prompt-123"
    _assert_request(
        session,
        method="POST",
        url=f"{BASE_URL}/prompt",
        params=None,
        json={"prompt": graph, "client_id": "client-456"},
    )


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"prompt-123": {"outputs": {}}}, {"outputs": {}}),
        ({"outputs": {}, "status": {"completed": True}}, {"outputs": {}, "status": {"completed": True}}),
        ({}, None),
    ],
)
def test_history_returns_payload_or_none(
    session: Mock, transport: _ComfyUIHTTPTransport, payload: dict[str, object], expected: dict[str, object] | None
) -> None:
    session.request.return_value = _response(payload=payload)

    assert transport.history("prompt-123") == expected
    _assert_request(
        session,
        method="GET",
        url=f"{BASE_URL}/history/prompt-123",
        params=None,
        json=None,
    )


def test_view_image_uses_exact_query_parameters(session: Mock, transport: _ComfyUIHTTPTransport) -> None:
    session.request.return_value = _response(content=b"png-bytes")

    assert transport.view_image("result.png", "nested", "output") == b"png-bytes"
    _assert_request(
        session,
        method="GET",
        url=f"{BASE_URL}/view",
        params={"filename": "result.png", "subfolder": "nested", "type": "output"},
        json=None,
    )


@pytest.mark.parametrize("content", [b"", "not-bytes"])
def test_view_image_rejects_missing_image_bytes(
    session: Mock, transport: _ComfyUIHTTPTransport, content: object
) -> None:
    response = _response()
    response.content = content
    session.request.return_value = response

    with pytest.raises(_ComfyUIHTTPError, match="image bytes"):
        transport.view_image("result.png", "", "output")


def test_interrupt_and_delete_from_queue_use_documented_requests(
    session: Mock, transport: _ComfyUIHTTPTransport
) -> None:
    session.request.return_value = _response()

    assert transport.interrupt() is None
    _assert_request(session, method="POST", url=f"{BASE_URL}/interrupt", params=None, json=None)

    session.request.reset_mock()
    session.request.return_value = _response()
    assert transport.delete_from_queue("prompt-123") is None
    _assert_request(
        session,
        method="POST",
        url=f"{BASE_URL}/queue",
        params=None,
        json={"delete": ["prompt-123"]},
    )


@pytest.mark.parametrize(
    "method_call, request_exception",
    [
        (lambda transport: transport.queue_status(), requests.Timeout("slow response")),
        (lambda transport: transport.interrupt(), requests.ConnectionError("connection refused")),
        (lambda transport: transport.submit_prompt({}, "client"), requests.HTTPError("500 Server Error")),
    ],
)
def test_requests_failures_are_translated_to_transport_error(
    session: Mock,
    transport: _ComfyUIHTTPTransport,
    method_call: object,
    request_exception: requests.RequestException,
) -> None:
    session.request.side_effect = request_exception

    with pytest.raises(_ComfyUIHTTPError) as raised:
        method_call(transport)  # type: ignore[operator]
    assert isinstance(raised.value.__cause__, requests.RequestException)


@pytest.mark.parametrize(
    "method_call",
    [
        lambda transport: transport.queue_status(),
        lambda transport: transport.history("prompt-123"),
        lambda transport: transport.submit_prompt({}, "client"),
    ],
)
def test_malformed_json_is_translated_to_transport_error(
    session: Mock, transport: _ComfyUIHTTPTransport, method_call: object
) -> None:
    session.request.return_value = _response(json_error=ValueError("not JSON"))

    with pytest.raises(_ComfyUIHTTPError, match="malformed JSON") as raised:
        method_call(transport)  # type: ignore[operator]
    assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "method_call, payload",
    [
        (lambda transport: transport.system_stats(), {"system": {}, "devices": []}),
        (lambda transport: transport.submit_prompt({}, "client"), {"number": 1}),
        (lambda transport: transport.queue_status(), []),
        (lambda transport: transport.history("prompt-123"), ["unexpected"]),
        (lambda transport: transport.history("prompt-123"), {"prompt-123": []}),
    ],
)
def test_invalid_response_shapes_raise_typed_transport_error(
    session: Mock, transport: _ComfyUIHTTPTransport, method_call: object, payload: object
) -> None:
    session.request.return_value = _response(payload=payload)

    with pytest.raises(_ComfyUIHTTPError):
        method_call(transport)  # type: ignore[operator]


@pytest.mark.parametrize(
    "http_error",
    [
        requests.HTTPError("400 Client Error"),
        requests.HTTPError("500 Server Error"),
    ],
)
def test_http_status_failure_is_translated_with_original_cause(
    session: Mock, transport: _ComfyUIHTTPTransport, http_error: requests.HTTPError
) -> None:
    session.request.return_value = _response(payload={}, request_error=http_error)

    with pytest.raises(_ComfyUIHTTPError) as raised:
        transport.queue_status()
    assert raised.value.__cause__ is http_error


def test_transport_does_not_retry_failed_requests(session: Mock, transport: _ComfyUIHTTPTransport) -> None:
    session.request.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(_ComfyUIHTTPError):
        transport.submit_prompt({}, "client")

    assert session.request.call_count == 1


@pytest.mark.parametrize(
    "base_url, timeout_seconds",
    [("", 1.0), (BASE_URL, 0.0), (BASE_URL, -1.0)],
)
def test_invalid_transport_configuration_is_rejected(
    session: Mock, base_url: str, timeout_seconds: float
) -> None:
    with pytest.raises(ValueError):
        _ComfyUIHTTPTransport(base_url, session, timeout_seconds)


class _FakeWebSocket:
    def __init__(self, frames: list[object] | None = None) -> None:
        self.frames = list(frames or [])
        self.timeouts: list[float] = []
        self.closed = False
        self.close_calls = 0
        self.recv_started = threading.Event()
        self.release_recv = threading.Event()
        self.block_on_recv = False

    def settimeout(self, timeout_seconds: float) -> None:
        self.timeouts.append(timeout_seconds)

    def recv(self) -> object:
        self.recv_started.set()
        if self.block_on_recv:
            self.release_recv.wait(timeout=1.0)
            raise websocket.WebSocketConnectionClosedException("closed during read")
        if not self.frames:
            raise websocket.WebSocketTimeoutException("read timed out")
        frame = self.frames.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    def close(self, timeout: int = 1) -> None:
        self.close_calls += 1
        self.closed = True
        self.release_recv.set()


@pytest.fixture
def ws_transport() -> _ComfyUIWebSocketTransport:
    return _ComfyUIWebSocketTransport(WS_URL, CLIENT_ID, CONNECT_TIMEOUT_SECONDS)


def _connect(monkeypatch: pytest.MonkeyPatch, transport: _ComfyUIWebSocketTransport, socket_obj: _FakeWebSocket) -> None:
    create_connection = Mock(return_value=socket_obj)
    monkeypatch.setattr("comfyui_client.websocket.create_connection", create_connection)

    transport.ensure_connected()

    create_connection.assert_called_once_with(
        f"{WS_URL}?clientId={CLIENT_ID}",
        timeout=CONNECT_TIMEOUT_SECONDS,
    )


def _frame(event_type: str, data: dict[str, object]) -> str:
    return json.dumps({"type": event_type, "data": data})


class TestWebSocketTransport:
    def test_successful_connection_is_lazy_and_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        socket_obj = _FakeWebSocket()
        create_connection = Mock(return_value=socket_obj)
        monkeypatch.setattr("comfyui_client.websocket.create_connection", create_connection)

        assert ws_transport.is_connected() is False
        ws_transport.ensure_connected()
        ws_transport.ensure_connected()

        assert ws_transport.is_connected() is True
        create_connection.assert_called_once_with(
            f"{WS_URL}?clientId={CLIENT_ID}",
            timeout=CONNECT_TIMEOUT_SECONDS,
        )

    @pytest.mark.parametrize(
        "raised",
        [
            websocket.WebSocketException("upgrade failed"),
            ConnectionRefusedError("refused"),
            TimeoutError("timed out"),
        ],
    )
    def test_failed_connection_is_translated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_transport: _ComfyUIWebSocketTransport,
        raised: BaseException,
    ) -> None:
        monkeypatch.setattr("comfyui_client.websocket.create_connection", Mock(side_effect=raised))

        with pytest.raises(ComfyUIConnectionError) as error:
            ws_transport.ensure_connected()

        assert error.value.__cause__ is raised
        assert ws_transport.is_connected() is False

    def test_connection_timeout_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        def fail_slowly(*_args: object, **_kwargs: object) -> None:
            time.sleep(0.02)
            raise TimeoutError("handshake timed out")

        monkeypatch.setattr("comfyui_client.websocket.create_connection", fail_slowly)
        started = time.monotonic()

        with pytest.raises(ComfyUIConnectionError):
            ws_transport.ensure_connected()

        assert time.monotonic() - started < 1.0

    def test_receive_returns_text_frame_and_sets_read_timeout(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        socket_obj = _FakeWebSocket(frames=['{"type": "status"}'])
        _connect(monkeypatch, ws_transport, socket_obj)

        assert ws_transport.receive(0.5) == '{"type": "status"}'
        assert socket_obj.timeouts == [0.5]
        assert ws_transport.is_connected() is True

    def test_receive_timeout_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        socket_obj = _FakeWebSocket()
        _connect(monkeypatch, ws_transport, socket_obj)

        assert ws_transport.receive(0.01) is None
        assert ws_transport.is_connected() is True

    def test_receive_filters_binary_frames(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        socket_obj = _FakeWebSocket(frames=[b"\x89PNG"])
        _connect(monkeypatch, ws_transport, socket_obj)

        assert ws_transport.receive(0.5) is None
        assert ws_transport.next_event(0.5) is None

    @pytest.mark.parametrize(
        "raised",
        [
            websocket.WebSocketConnectionClosedException("closed"),
            ConnectionResetError("reset"),
            BrokenPipeError("broken pipe"),
        ],
    )
    def test_disconnect_during_receive_is_translated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_transport: _ComfyUIWebSocketTransport,
        raised: BaseException,
    ) -> None:
        socket_obj = _FakeWebSocket(frames=[raised])
        _connect(monkeypatch, ws_transport, socket_obj)

        with pytest.raises(ComfyUIConnectionError) as error:
            ws_transport.receive(0.5)

        assert error.value.__cause__ is raised
        assert ws_transport.is_connected() is False

    def test_receive_without_connection_raises_typed_error(self, ws_transport: _ComfyUIWebSocketTransport) -> None:
        with pytest.raises(ComfyUIConnectionError):
            ws_transport.receive(0.5)

    def test_reconnect_after_detected_disconnect(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        first_socket = _FakeWebSocket(frames=[websocket.WebSocketConnectionClosedException("closed")])
        second_socket = _FakeWebSocket()
        create_connection = Mock(side_effect=[first_socket, second_socket])
        monkeypatch.setattr("comfyui_client.websocket.create_connection", create_connection)

        ws_transport.ensure_connected()
        with pytest.raises(ComfyUIConnectionError):
            ws_transport.receive(0.5)
        ws_transport.ensure_connected()

        assert ws_transport.is_connected() is True
        assert create_connection.call_count == 2

    def test_close_is_idempotent_and_supports_reconnect(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        first_socket = _FakeWebSocket()
        second_socket = _FakeWebSocket()
        create_connection = Mock(side_effect=[first_socket, second_socket])
        monkeypatch.setattr("comfyui_client.websocket.create_connection", create_connection)

        ws_transport.close()
        ws_transport.ensure_connected()
        ws_transport.close()
        ws_transport.close()
        ws_transport.ensure_connected()

        assert first_socket.close_calls == 1
        assert second_socket.close_calls == 0
        assert create_connection.call_count == 2
        assert ws_transport.is_connected() is True

    def test_close_swallows_cleanup_failures(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        socket_obj = _FakeWebSocket()
        socket_obj.close = Mock(side_effect=websocket.WebSocketException("close failed"))  # type: ignore[method-assign]
        _connect(monkeypatch, ws_transport, socket_obj)

        ws_transport.close()

        assert ws_transport.is_connected() is False

    def test_close_unblocks_active_receive(
        self, monkeypatch: pytest.MonkeyPatch, ws_transport: _ComfyUIWebSocketTransport
    ) -> None:
        socket_obj = _FakeWebSocket()
        socket_obj.block_on_recv = True
        _connect(monkeypatch, ws_transport, socket_obj)
        result: list[object] = []

        def receive() -> None:
            try:
                result.append(ws_transport.receive(0.5))
            except Exception as exc:  # The design allows connection error or clean None.
                result.append(exc)

        thread = threading.Thread(target=receive)
        thread.start()
        assert socket_obj.recv_started.wait(timeout=1.0)
        ws_transport.close()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert result

    @pytest.mark.parametrize(
        "event_type, data, expected",
        [
            (
                "status",
                {"status": {"exec_info": {"queue_remaining": 3}}},
                ComfyUIEvent("status", None, None, None, None, 3, None),
            ),
            (
                "executing",
                {"node": "12", "display_node": "12", "prompt_id": "prompt-1"},
                ComfyUIEvent("executing", "prompt-1", "12", None, None, None, None),
            ),
            (
                "executing",
                {"node": None, "display_node": "12", "prompt_id": "prompt-1"},
                ComfyUIEvent("executing", "prompt-1", None, None, None, None, None),
            ),
            (
                "progress",
                {"value": 4, "max": 10, "prompt_id": "prompt-1", "node": "12"},
                ComfyUIEvent("progress", "prompt-1", "12", 4, 10, None, None),
            ),
            (
                "execution_error",
                {"prompt_id": "prompt-1", "exception_type": "RuntimeError"},
                ComfyUIEvent(
                    "execution_error",
                    "prompt-1",
                    None,
                    None,
                    None,
                    None,
                    {"prompt_id": "prompt-1", "exception_type": "RuntimeError"},
                ),
            ),
            (
                "execution_cached",
                {"nodes": ["1", "2"], "prompt_id": "prompt-1"},
                ComfyUIEvent("execution_cached", "prompt-1", None, None, None, None, None),
            ),
        ],
    )
    def test_next_event_parses_recognized_events(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_transport: _ComfyUIWebSocketTransport,
        event_type: str,
        data: dict[str, object],
        expected: ComfyUIEvent,
    ) -> None:
        socket_obj = _FakeWebSocket(frames=[_frame(event_type, data)])
        _connect(monkeypatch, ws_transport, socket_obj)

        assert ws_transport.next_event(0.5) == expected

    @pytest.mark.parametrize(
        "frame",
        [
            "{not json",
            json.dumps(["not", "object"]),
            json.dumps({"data": {}}),
            json.dumps({"type": 123, "data": {}}),
            _frame("status", {}),
            _frame("status", {"status": {"exec_info": {"queue_remaining": -1}}}),
            _frame("executing", {"node": "1"}),
            _frame("executing", {"node": 12, "prompt_id": "prompt-1"}),
            _frame("progress", {"value": 11, "max": 10, "prompt_id": "prompt-1", "node": "1"}),
            _frame("progress", {"value": 1, "max": 10, "prompt_id": "prompt-1", "node": ""}),
            _frame("execution_error", {"exception_type": "RuntimeError"}),
            _frame("execution_cached", {"nodes": ["1"]}),
        ],
    )
    def test_next_event_rejects_malformed_or_invalid_frames(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_transport: _ComfyUIWebSocketTransport,
        frame: str,
    ) -> None:
        socket_obj = _FakeWebSocket(frames=[frame])
        _connect(monkeypatch, ws_transport, socket_obj)

        assert ws_transport.next_event(0.5) is None

    @pytest.mark.parametrize(
        "event_type",
        ["execution_start", "executed", "future_event"],
    )
    def test_next_event_filters_unknown_or_untracked_events(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_transport: _ComfyUIWebSocketTransport,
        event_type: str,
    ) -> None:
        socket_obj = _FakeWebSocket(frames=[_frame(event_type, {"prompt_id": "prompt-1"})])
        _connect(monkeypatch, ws_transport, socket_obj)

        assert ws_transport.next_event(0.5) is None

    @pytest.mark.parametrize(
        "ws_url, client_id, timeout_seconds",
        [
            ("http://127.0.0.1:8188/ws", CLIENT_ID, 1.0),
            (WS_URL, "", 1.0),
            (WS_URL, "   ", 1.0),
            (WS_URL, CLIENT_ID, 0.0),
            (WS_URL, CLIENT_ID, -1.0),
        ],
    )
    def test_invalid_websocket_transport_configuration_is_rejected(
        self, ws_url: str, client_id: str, timeout_seconds: float
    ) -> None:
        with pytest.raises(ValueError):
            _ComfyUIWebSocketTransport(ws_url, client_id, timeout_seconds)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeQueueWebSocket:
    def __init__(
        self,
        events: list[object] | None = None,
        clock: _FakeClock | None = None,
        advance_seconds: float = 1.0,
        reconnect_results: list[object] | None = None,
    ) -> None:
        self.events = list(events or [])
        self.clock = clock
        self.advance_seconds = advance_seconds
        self.reconnect_results = list(reconnect_results or [])
        self.next_event_timeouts: list[float] = []
        self.ensure_connected_calls = 0

    def next_event(self, timeout_seconds: float) -> ComfyUIEvent | None:
        self.next_event_timeouts.append(timeout_seconds)
        if self.clock is not None:
            self.clock.advance(self.advance_seconds)
        if not self.events:
            return None
        item = self.events.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    def ensure_connected(self) -> None:
        self.ensure_connected_calls += 1
        if not self.reconnect_results:
            return
        result = self.reconnect_results.pop(0)
        if isinstance(result, BaseException):
            raise result


class _FakeQueueHTTP:
    def __init__(self, histories: list[object] | None = None) -> None:
        self.histories = list(histories or [])
        self.prompt_ids: list[str] = []

    def history(self, prompt_id: str) -> dict[str, object] | None:
        self.prompt_ids.append(prompt_id)
        if not self.histories:
            return None
        item = self.histories.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]


class _FakeLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.kwargs: list[dict[str, object]] = []

    def debug(self, message: str, **kwargs: object) -> None:
        self.entries.append(("DEBUG", message))
        self.kwargs.append(kwargs)

    def info(self, message: str, **kwargs: object) -> None:
        self.entries.append(("INFO", message))
        self.kwargs.append(kwargs)

    def warning(self, message: str, **kwargs: object) -> None:
        self.entries.append(("WARNING", message))
        self.kwargs.append(kwargs)

    def error(self, message: str, **kwargs: object) -> None:
        self.entries.append(("ERROR", message))
        self.kwargs.append(kwargs)


def _status_event(queue_remaining: int) -> ComfyUIEvent:
    return ComfyUIEvent("status", None, None, None, None, queue_remaining, None)


def _executing_event(prompt_id: str, node: str | None) -> ComfyUIEvent:
    return ComfyUIEvent("executing", prompt_id, node, None, None, None, None)


def _progress_event(prompt_id: str, value: int, maximum: int, node: str = "4") -> ComfyUIEvent:
    return ComfyUIEvent("progress", prompt_id, node, value, maximum, None, None)


def _error_event(prompt_id: str, payload: dict[str, object]) -> ComfyUIEvent:
    return ComfyUIEvent("execution_error", prompt_id, None, None, None, None, payload)


def _cached_event(prompt_id: str) -> ComfyUIEvent:
    return ComfyUIEvent("execution_cached", prompt_id, None, None, None, None, None)


def _history(status_str: str = "success", completed: bool = True) -> dict[str, object]:
    return {"status": {"completed": completed, "status_str": status_str}, "outputs": {"9": {"images": []}}}


def _error_history(payload: dict[str, object] | None = None) -> dict[str, object]:
    error_payload = payload or {"prompt_id": "prompt-123", "exception_message": "failed"}
    return {
        "status": {
            "completed": True,
            "status_str": "error",
            "messages": [["execution_error", error_payload]],
        }
    }


def _tracker(
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    http: _FakeQueueHTTP,
    ws: _FakeQueueWebSocket,
    *,
    poll_interval_seconds: float = 3.0,
    execution_timeout_seconds: float = 30.0,
) -> _QueueTracker:
    monkeypatch.setattr("comfyui_client.time.monotonic", clock.monotonic)
    monkeypatch.setattr("comfyui_client.time.sleep", clock.sleep)
    return _QueueTracker(
        http,  # type: ignore[arg-type]
        ws,  # type: ignore[arg-type]
        poll_interval_seconds=poll_interval_seconds,
        execution_timeout_seconds=execution_timeout_seconds,
    )


class TestQueueTracker:
    def test_successful_execution_confirms_history_and_measures_timing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([_history()])
        ws = _FakeQueueWebSocket(
            [
                _status_event(2),
                _status_event(0),
                _executing_event("prompt-123", "4"),
                _progress_event("prompt-123", 5, 10),
                _executing_event("prompt-123", None),
            ],
            clock,
        )
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.history_payload == _history()
        assert result.error_payload is None
        assert result.queue_wait_seconds == 3.0
        assert result.generation_seconds == 2.0
        assert result.used_http_fallback is False
        assert http.prompt_ids == ["prompt-123"]

    def test_direct_cached_completion_has_zero_generation_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([_history()])
        ws = _FakeQueueWebSocket([_status_event(0), _executing_event("prompt-123", None)], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.queue_wait_seconds == 2.0
        assert result.generation_seconds == 0.0

    def test_foreign_prompt_events_do_not_affect_our_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([_history()])
        ws = _FakeQueueWebSocket(
            [
                _executing_event("foreign", "1"),
                _progress_event("foreign", 10, 10),
                _executing_event("prompt-123", "4"),
                _executing_event("prompt-123", None),
            ],
            clock,
        )
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.queue_wait_seconds == 3.0
        assert result.generation_seconds == 1.0

    def test_execution_cached_is_informational_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([_history()])
        ws = _FakeQueueWebSocket(
            [
                _cached_event("prompt-123"),
                _executing_event("prompt-123", "4"),
                _executing_event("prompt-123", None),
            ],
            clock,
        )
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.queue_wait_seconds == 2.0
        assert result.generation_seconds == 1.0

    def test_execution_error_event_returns_error_without_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        payload = {"prompt_id": "prompt-123", "exception_message": "node failed"}
        http = _FakeQueueHTTP([_history()])
        ws = _FakeQueueWebSocket([_executing_event("prompt-123", "4"), _error_event("prompt-123", payload)], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.EXECUTION_ERROR
        assert result.error_payload == payload
        assert result.history_payload is None
        assert http.prompt_ids == []
        assert result.queue_wait_seconds == 1.0
        assert result.generation_seconds == 1.0

    def test_completion_history_error_becomes_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        payload = {"prompt_id": "prompt-123", "exception_type": "RuntimeError"}
        http = _FakeQueueHTTP([_error_history(payload)])
        ws = _FakeQueueWebSocket([_executing_event("prompt-123", "4"), _executing_event("prompt-123", None)], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.EXECUTION_ERROR
        assert result.error_payload == payload
        assert result.history_payload is None

    def test_confirmation_retries_until_history_is_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([None, {"outputs": {}}, _history()])
        ws = _FakeQueueWebSocket([_executing_event("prompt-123", "4"), _executing_event("prompt-123", None)], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert len(http.prompt_ids) == 3
        assert clock.sleeps == [0.5, 0.5]

    def test_confirmation_exhaustion_falls_back_to_polling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([None, None, None, None, _history()])
        ws = _FakeQueueWebSocket([_executing_event("prompt-123", None)], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.used_http_fallback is True
        assert len(http.prompt_ids) == 5

    def test_websocket_disconnect_uses_http_fallback_until_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([None, {"status": {"completed": False, "status_str": "running"}}, _history()])
        ws = _FakeQueueWebSocket([ComfyUIConnectionError("closed")], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.used_http_fallback is True
        assert len(http.prompt_ids) == 3
        assert clock.sleeps == [3.0, 3.0]

    def test_http_poll_error_does_not_abort_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([_ComfyUIHTTPError("temporary"), _history()])
        ws = _FakeQueueWebSocket([ComfyUIConnectionError("closed")], clock)
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.used_http_fallback is True
        assert len(http.prompt_ids) == 2

    def test_polling_reconnect_checks_history_before_resuming_websocket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([None, None, None, _history()])
        ws = _FakeQueueWebSocket([ComfyUIConnectionError("closed")], clock, reconnect_results=[None])
        tracker = _tracker(monkeypatch, clock, http, ws, execution_timeout_seconds=60.0)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert ws.ensure_connected_calls == 1
        assert len(http.prompt_ids) == 4

    def test_polling_reconnect_can_resume_websocket_phase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([None, None, None, {"status": {"completed": False, "status_str": "running"}}, _history()])
        ws = _FakeQueueWebSocket(
            [
                ComfyUIConnectionError("closed"),
                _executing_event("prompt-123", "4"),
                _executing_event("prompt-123", None),
            ],
            clock,
            reconnect_results=[None],
        )
        tracker = _tracker(monkeypatch, clock, http, ws, execution_timeout_seconds=60.0)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        assert result.used_http_fallback is True
        assert ws.ensure_connected_calls == 1
        assert len(ws.next_event_timeouts) == 3

    def test_reconnect_skipped_when_budget_is_nearly_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP([None, None, None, None])
        ws = _FakeQueueWebSocket([ComfyUIConnectionError("closed")], clock)
        tracker = _tracker(
            monkeypatch,
            clock,
            http,
            ws,
            poll_interval_seconds=3.0,
            execution_timeout_seconds=12.0,
        )

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.TIMEOUT
        assert ws.ensure_connected_calls == 0

    def test_timeout_before_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP()
        ws = _FakeQueueWebSocket([None, None, None], clock, advance_seconds=2.0)
        tracker = _tracker(monkeypatch, clock, http, ws, execution_timeout_seconds=5.0)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.TIMEOUT
        assert result.history_payload is None
        assert result.error_payload is None
        assert result.generation_seconds == 0.0

    def test_timeout_after_execution_reports_partial_generation_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        http = _FakeQueueHTTP()
        ws = _FakeQueueWebSocket([_executing_event("prompt-123", "4"), None, None], clock, advance_seconds=2.0)
        tracker = _tracker(monkeypatch, clock, http, ws, execution_timeout_seconds=5.0)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.TIMEOUT
        assert result.queue_wait_seconds == 2.0
        assert result.generation_seconds == 4.0

    def test_still_queued_warning_is_logged_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)
        http = _FakeQueueHTTP()
        ws = _FakeQueueWebSocket([_status_event(3), _status_event(3), None], clock, advance_seconds=15.0)
        tracker = _tracker(monkeypatch, clock, http, ws, execution_timeout_seconds=40.0)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.TIMEOUT
        warnings = [entry for entry in logger_spy.entries if entry[0] == "WARNING" and "still queued" in entry[1]]
        assert len(warnings) == 1

    def test_progress_milestones_are_logged_at_info_granularity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)
        http = _FakeQueueHTTP([_history()])
        ws = _FakeQueueWebSocket(
            [
                _executing_event("prompt-123", "4"),
                _progress_event("prompt-123", 10, 100),
                _progress_event("prompt-123", 25, 100),
                _progress_event("prompt-123", 55, 100),
                _progress_event("prompt-123", 100, 100),
                _executing_event("prompt-123", None),
            ],
            clock,
        )
        tracker = _tracker(monkeypatch, clock, http, ws)

        result = tracker.await_completion("prompt-123", "client-abc")

        assert result.outcome is _CompletionOutcome.COMPLETED
        milestone_logs = [
            entry for entry in logger_spy.entries
            if entry[0] == "INFO" and "progress milestone" in entry[1]
        ]
        assert len(milestone_logs) == 3

    @pytest.mark.parametrize(
        "poll_interval_seconds, execution_timeout_seconds",
        [(0.0, 30.0), (-1.0, 30.0), (3.0, 0.0), (30.0, 30.0), (31.0, 30.0)],
    )
    def test_invalid_tracker_configuration_is_rejected(
        self, poll_interval_seconds: float, execution_timeout_seconds: float
    ) -> None:
        with pytest.raises(ValueError):
            _QueueTracker(
                _FakeQueueHTTP(),  # type: ignore[arg-type]
                _FakeQueueWebSocket(),  # type: ignore[arg-type]
                poll_interval_seconds=poll_interval_seconds,
                execution_timeout_seconds=execution_timeout_seconds,
            )


def _image_bytes(image_format: str = "PNG", size: tuple[int, int] = (3, 2)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color=(20, 40, 60)).save(stream, format=image_format)
    return stream.getvalue()


def _completed(history_payload: dict[str, object] | None = None) -> _CompletionResult:
    return _CompletionResult(
        outcome=_CompletionOutcome.COMPLETED,
        history_payload=history_payload,
        error_payload=None,
        queue_wait_seconds=1.0,
        generation_seconds=2.0,
        used_http_fallback=False,
    )


def _failed_completion() -> _CompletionResult:
    return _CompletionResult(
        outcome=_CompletionOutcome.EXECUTION_ERROR,
        history_payload=None,
        error_payload={"exception_message": "failed"},
        queue_wait_seconds=1.0,
        generation_seconds=2.0,
        used_http_fallback=False,
    )


def _output_history(outputs: dict[str, object]) -> dict[str, object]:
    return {"status": {"completed": True, "status_str": "success"}, "outputs": outputs}


def _top_level_history(prompt_id: str, outputs: dict[str, object]) -> dict[str, object]:
    return {prompt_id: _output_history(outputs)}


def _image_entry(filename: str, subfolder: str = "", image_type: str = "output") -> dict[str, str]:
    return {"filename": filename, "subfolder": subfolder, "type": image_type}


class _FakeOutputHTTP:
    def __init__(
        self,
        *,
        histories: list[object] | None = None,
        images: list[object] | None = None,
    ) -> None:
        self.histories = list(histories or [])
        self.images = list(images or [_image_bytes()])
        self.history_prompt_ids: list[str] = []
        self.view_calls: list[tuple[str, str, str]] = []

    def history(self, prompt_id: str) -> dict[str, object] | None:
        self.history_prompt_ids.append(prompt_id)
        if not self.histories:
            return None
        item = self.histories.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    def view_image(self, filename: str, subfolder: str, image_type: str) -> bytes:
        self.view_calls.append((filename, subfolder, image_type))
        if not self.images:
            return _image_bytes()
        item = self.images.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]


class _FakeMetricsCollector:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.records: list[object] = []

    def append(self, metrics: object) -> None:
        if self.error is not None:
            raise self.error
        self.records.append(metrics)


class TestOutputRetriever:
    def test_single_output_node_single_image_returns_validated_result(self) -> None:
        history_payload = _output_history({"9": {"images": [_image_entry("result.png", "finals")]}})
        http = _FakeOutputHTTP(images=[_image_bytes("PNG", (5, 4))])
        retriever = _OutputRetriever(http)  # type: ignore[arg-type]

        result = retriever.retrieve(_completed(history_payload), prompt_id="prompt-123")

        assert result == _OutputResult(
            prompt_id="prompt-123",
            output_node_id="9",
            filename="result.png",
            subfolder="finals",
            image_type="output",
            format="png",
            content=result.content,
            width=5,
            height=4,
        )
        assert result.content
        assert http.history_prompt_ids == []
        assert http.view_calls == [("result.png", "finals", "output")]

    def test_top_level_embedded_history_is_supported(self) -> None:
        history_payload = _top_level_history("prompt-123", {"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[_image_bytes()])

        result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.output_node_id == "9"
        assert http.history_prompt_ids == []

    def test_missing_embedded_history_is_fetched_from_transport(self) -> None:
        http = _FakeOutputHTTP(
            histories=[_output_history({"9": {"images": [_image_entry("result.png")]}})],
            images=[_image_bytes()],
        )

        result = _OutputRetriever(http).retrieve(_completed(None), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.filename == "result.png"
        assert http.history_prompt_ids == ["prompt-123"]

    def test_single_output_node_is_selected(self) -> None:
        history_payload = _output_history({"22": {"images": [_image_entry("only.png")]}})
        http = _FakeOutputHTTP(images=[_image_bytes()])

        result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.output_node_id == "22"
        assert result.filename == "only.png"

    def test_multiple_output_nodes_fall_back_to_collection_order(self) -> None:
        history_payload = _output_history({
            "4": {"images": [_image_entry("first.png")]},
            "9": {"images": [_image_entry("second.png")]},
        })
        http = _FakeOutputHTTP(images=[_image_bytes()])

        result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.output_node_id == "4"
        assert result.filename == "first.png"

    def test_preferred_output_node_wins_over_collection_order(self) -> None:
        history_payload = _output_history({
            "4": {"images": [_image_entry("first.png")]},
            "9": {"images": [_image_entry("preferred.png")]},
        })
        http = _FakeOutputHTTP(images=[_image_bytes()])
        retriever = _OutputRetriever(http, preferred_output_nodes=("9",))  # type: ignore[arg-type]

        result = retriever.retrieve(_completed(history_payload), prompt_id="prompt-123")

        assert result.output_node_id == "9"
        assert result.filename == "preferred.png"

    def test_unmatched_preferred_node_falls_back_to_first_candidate(self) -> None:
        history_payload = _output_history({
            "4": {"images": [_image_entry("first.png")]},
            "9": {"images": [_image_entry("second.png")]},
        })
        http = _FakeOutputHTTP(images=[_image_bytes()])
        retriever = _OutputRetriever(http, preferred_output_nodes=("99",))  # type: ignore[arg-type]

        result = retriever.retrieve(_completed(history_payload), prompt_id="prompt-123")

        assert result.output_node_id == "4"
        assert result.filename == "first.png"

    def test_multiple_images_on_same_node_selects_first_image(self) -> None:
        history_payload = _output_history({
            "9": {"images": [_image_entry("first.png"), _image_entry("second.png")]},
        })
        http = _FakeOutputHTTP(images=[_image_bytes()])

        result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.filename == "first.png"

    def test_unsupported_extension_is_filtered_before_selection(self) -> None:
        history_payload = _output_history({
            "4": {"images": [_image_entry("first.bmp")]},
            "9": {"images": [_image_entry("second.png")]},
        })
        http = _FakeOutputHTTP(images=[_image_bytes()])

        result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.output_node_id == "9"
        assert result.filename == "second.png"

    def test_select_image_is_deterministic(self) -> None:
        retriever = _OutputRetriever(_FakeOutputHTTP(), preferred_output_nodes=("9",))  # type: ignore[arg-type]
        candidates = [
            _ImageCandidate("4", "first.png", "", "output", "png"),
            _ImageCandidate("9", "preferred.png", "", "output", "png"),
            _ImageCandidate("9", "second.png", "", "output", "png"),
        ]

        assert retriever._select_image(candidates) == retriever._select_image(candidates)
        assert retriever._select_image(candidates).filename == "preferred.png"

    @pytest.mark.parametrize("payload", [None, "garbage", []])
    def test_malformed_history_payload_raises_history_error(self, payload: object) -> None:
        http = _FakeOutputHTTP()
        completion = _completed(payload)  # type: ignore[arg-type]

        with pytest.raises(OutputHistoryError):
            _OutputRetriever(http).retrieve(completion, prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_history_fetch_error_raises_history_error(self) -> None:
        http = _FakeOutputHTTP(histories=[_ComfyUIHTTPError("history failed")])

        with pytest.raises(OutputHistoryError):
            _OutputRetriever(http).retrieve(_completed(None), prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_missing_prompt_key_raises_history_error(self) -> None:
        payload = {"other-prompt": _output_history({"9": {"images": [_image_entry("result.png")]}})}
        http = _FakeOutputHTTP()

        with pytest.raises(OutputHistoryError):
            _OutputRetriever(http).retrieve(_completed(payload), prompt_id="prompt-123")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "history_payload",
        [
            {"status": {"completed": True}},
            {"outputs": []},
        ],
    )
    def test_missing_or_malformed_outputs_raises_history_error(self, history_payload: dict[str, object]) -> None:
        http = _FakeOutputHTTP()

        with pytest.raises(OutputHistoryError):
            _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_empty_outputs_raises_no_output_image_error(self) -> None:
        http = _FakeOutputHTTP()

        with pytest.raises(NoOutputImageError):
            _OutputRetriever(http).retrieve(_completed(_output_history({})), prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_node_and_image_level_malformed_entries_are_skipped(self) -> None:
        history_payload = _output_history({
            "bad-node": [],
            "no-images": {"gifs": [_image_entry("ignored.gif")]},
            "bad-images": {"images": "not-a-list"},
            "bad-entry": {"images": [{"subfolder": ""}, _image_entry("")]},
            "good": {"images": [_image_entry("result.png")]},
        })
        http = _FakeOutputHTTP(images=[_image_bytes()])

        result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert result.output_node_id == "good"
        assert result.filename == "result.png"

    def test_all_candidates_filtered_out_raises_no_output_image_error(self) -> None:
        history_payload = _output_history({"9": {"images": [_image_entry("result.bmp")]}})
        http = _FakeOutputHTTP()

        with pytest.raises(NoOutputImageError):
            _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_successful_download_on_first_attempt_does_not_retry(self) -> None:
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[_image_bytes()])

        _OutputRetriever(http, download_retries=3).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert len(http.view_calls) == 1

    def test_transient_download_failure_then_success_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("comfyui_client.time.sleep", lambda seconds: sleeps.append(seconds))
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[_ComfyUIHTTPError("temporary"), _image_bytes()])

        result = _OutputRetriever(http, download_retries=2, download_retry_backoff=0.25).retrieve(
            _completed(history_payload),
            prompt_id="prompt-123",
        )

        assert result.filename == "result.png"
        assert len(http.view_calls) == 2
        assert sleeps == [0.25]

    def test_transient_download_failure_exhaustion_raises_download_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("comfyui_client.time.sleep", lambda _seconds: None)
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[_ComfyUIHTTPError("one"), _ComfyUIHTTPError("two"), _ComfyUIHTTPError("three")])

        with pytest.raises(OutputDownloadError):
            _OutputRetriever(http, download_retries=2).retrieve(
                _completed(history_payload),
                prompt_id="prompt-123",
            )

        assert len(http.view_calls) == 3

    def test_404_download_raises_missing_file_without_retry(self) -> None:
        response = Mock()
        response.status_code = 404
        cause = requests.HTTPError("404 Client Error")
        cause.response = response
        error = _ComfyUIHTTPError("ComfyUI HTTP request failed for GET /view: 404 Client Error")
        error.__cause__ = cause
        history_payload = _output_history({"9": {"images": [_image_entry("missing.png")]}})
        http = _FakeOutputHTTP(images=[error, _image_bytes()])

        with pytest.raises(MissingOutputFileError):
            _OutputRetriever(http, download_retries=2).retrieve(
                _completed(history_payload),
                prompt_id="prompt-123",
            )

        assert len(http.view_calls) == 1

    def test_empty_download_is_retryable_then_exhausts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("comfyui_client.time.sleep", lambda _seconds: None)
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[b"", b""])

        with pytest.raises(OutputDownloadError):
            _OutputRetriever(http, download_retries=1).retrieve(
                _completed(history_payload),
                prompt_id="prompt-123",
            )

        assert len(http.view_calls) == 2

    def test_valid_jpeg_and_webp_bytes_are_supported(self) -> None:
        for filename, image_format in [("result.jpg", "JPEG"), ("result.webp", "WEBP")]:
            history_payload = _output_history({"9": {"images": [_image_entry(filename)]}})
            http = _FakeOutputHTTP(images=[_image_bytes(image_format, (7, 6))])

            result = _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

            assert result.width == 7
            assert result.height == 6

    def test_empty_bytes_reaching_validation_raise_corrupt_image_error(self) -> None:
        retriever = _OutputRetriever(_FakeOutputHTTP())  # type: ignore[arg-type]
        candidate = _ImageCandidate("9", "result.png", "", "output", "png")

        with pytest.raises(CorruptImageError):
            retriever._validate_image_bytes(b"", candidate, "prompt-123")

    def test_garbage_bytes_raise_corrupt_image_error(self) -> None:
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[b"not an image"])

        with pytest.raises(CorruptImageError):
            _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_decoded_format_mismatch_raises_unsupported_image_format_error(self) -> None:
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[_image_bytes("JPEG")])

        with pytest.raises(UnsupportedImageFormatError):
            _OutputRetriever(http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

    def test_repeated_calls_are_deterministic(self) -> None:
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        first = _OutputRetriever(_FakeOutputHTTP(images=[_image_bytes()])).retrieve(  # type: ignore[arg-type]
            _completed(history_payload),
            prompt_id="prompt-123",
        )
        second = _OutputRetriever(_FakeOutputHTTP(images=[_image_bytes()])).retrieve(  # type: ignore[arg-type]
            _completed(history_payload),
            prompt_id="prompt-123",
        )

        assert first == second

    def test_logging_behavior_for_success_retry_and_validation_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)
        monkeypatch.setattr("comfyui_client.time.sleep", lambda _seconds: None)
        history_payload = _output_history({"9": {"images": [_image_entry("result.png")]}})
        http = _FakeOutputHTTP(images=[_ComfyUIHTTPError("temporary"), _image_bytes()])

        _OutputRetriever(http, download_retries=1).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        levels = [level for level, _message in logger_spy.entries]
        assert "INFO" in levels
        assert "WARNING" in levels
        assert not any("PNG" in message and "IHDR" in message for _level, message in logger_spy.entries)

        logger_spy.entries.clear()
        bad_http = _FakeOutputHTTP(images=[b"not an image"])
        with pytest.raises(CorruptImageError):
            _OutputRetriever(bad_http).retrieve(_completed(history_payload), prompt_id="prompt-123")  # type: ignore[arg-type]

        assert any(level == "ERROR" and "validation failed" in message for level, message in logger_spy.entries)

    def test_programmer_error_guards(self) -> None:
        retriever = _OutputRetriever(_FakeOutputHTTP())  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            retriever.retrieve(None, prompt_id="prompt-123")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            retriever.retrieve(_failed_completion(), prompt_id="prompt-123")
        with pytest.raises(ValueError):
            retriever.retrieve(_completed(_output_history({"9": {"images": [_image_entry("result.png")]}})))
        with pytest.raises(ValueError):
            _OutputRetriever(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "download_timeout, download_retries, download_retry_backoff",
        [(-0.1, None, None), (None, -1, None), (None, None, -0.1)],
    )
    def test_invalid_retriever_configuration_is_rejected(
        self,
        download_timeout: float | None,
        download_retries: int | None,
        download_retry_backoff: float | None,
    ) -> None:
        with pytest.raises(ValueError):
            _OutputRetriever(
                _FakeOutputHTTP(),  # type: ignore[arg-type]
                download_timeout=download_timeout,
                download_retries=download_retries,
                download_retry_backoff=download_retry_backoff,
            )


def _completion(
    outcome: _CompletionOutcome = _CompletionOutcome.COMPLETED,
    *,
    queue_wait_seconds: float = 1.0,
    generation_seconds: float = 2.0,
    used_http_fallback: bool = False,
) -> _CompletionResult:
    return _CompletionResult(
        outcome=outcome,
        history_payload=_output_history({"9": {"images": [_image_entry("result.png")]}})
        if outcome is _CompletionOutcome.COMPLETED
        else None,
        error_payload={"exception_message": "failed"}
        if outcome is _CompletionOutcome.EXECUTION_ERROR
        else None,
        queue_wait_seconds=queue_wait_seconds,
        generation_seconds=generation_seconds,
        used_http_fallback=used_http_fallback,
    )


def _output_result() -> _OutputResult:
    return _OutputResult(
        prompt_id="prompt-123",
        output_node_id="9",
        filename="result.png",
        subfolder="",
        image_type="output",
        format="png",
        content=b"png",
        width=1280,
        height=720,
    )


class TestMetricsRecorder:
    def test_successful_completion_builds_generation_metrics(self) -> None:
        collector = _FakeMetricsCollector()
        completion = _completion()

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            profile_name="low_vram",
            workflow_hash="hash-abc",
            completions=[completion],
            output=_output_result(),
            num_candidates_requested=1,
            peak_vram_mb=512.0,
            gpu_utilization_percent=42.0,
        )

        assert len(collector.records) == 1
        metrics = collector.records[0]
        assert metrics.video_id == "video-123"
        assert metrics.niche == "gaming"
        assert metrics.profile_name == "low_vram"
        assert metrics.workflow_version == "wf-v1"
        assert metrics.workflow_hash == "hash-abc"
        assert metrics.queue_time_seconds == 1.0
        assert metrics.generation_time_seconds == [2.0]
        assert metrics.total_duration_seconds == 3.0
        assert metrics.generation_retry_count == 0
        assert metrics.identity_retry_count == 0
        assert metrics.failure_reason is None
        assert metrics.peak_vram_mb == 512.0
        assert metrics.gpu_utilization_percent == 42.0

    def test_successful_append_logs_debug_without_metrics_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)

        _ComfyUIMetricsRecorder(_FakeMetricsCollector()).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=[_completion()],
            output=_output_result(),
        )

        debug_entries = [entry for entry in logger_spy.entries if entry[0] == "DEBUG"]
        assert len(debug_entries) == 1
        assert "metrics recorded" in debug_entries[0][1]
        assert logger_spy.kwargs[0] == {
            "video_id": "video-123",
            "failure_reason": None,
        }
        assert "generation_time_seconds" not in debug_entries[0][1]

    def test_attempt_started_at_uses_monotonic_elapsed_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collector = _FakeMetricsCollector()
        monkeypatch.setattr("comfyui_client.time.monotonic", lambda: 15.5)

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=[_completion()],
            output=_output_result(),
            attempt_started_at=10.0,
        )

        assert collector.records[0].total_duration_seconds == 5.5

    def test_retry_aggregation_preserves_completion_order(self) -> None:
        collector = _FakeMetricsCollector()
        completions = [
            _completion(_CompletionOutcome.EXECUTION_ERROR, queue_wait_seconds=0.5, generation_seconds=1.0),
            _completion(_CompletionOutcome.EXECUTION_ERROR, queue_wait_seconds=0.25, generation_seconds=1.5),
            _completion(_CompletionOutcome.COMPLETED, queue_wait_seconds=0.75, generation_seconds=2.5),
        ]

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=completions,
            output=_output_result(),
        )

        metrics = collector.records[0]
        assert metrics.queue_time_seconds == 1.5
        assert metrics.generation_time_seconds == [1.0, 1.5, 2.5]
        assert metrics.generation_retry_count == 2
        assert metrics.failure_reason is None

    @pytest.mark.parametrize(
        "exception, expected",
        [
            (MissingOutputFileError("missing"), "missing_output_file"),
            (CorruptImageError("corrupt"), "corrupt_image"),
            (UnsupportedImageFormatError("unsupported"), "unsupported_image_format"),
            (NoOutputImageError("none"), "no_output_image"),
            (OutputHistoryError("history"), "output_history_error"),
            (OutputDownloadError("download"), "output_download_error"),
            (VRAMExhaustedError("oom"), "vram_exhausted"),
            (ComfyUIQueueError("queue"), "queue_error"),
            (ComfyUIConnectionError("connection"), "connection_error"),
        ],
    )
    def test_exception_types_map_to_stable_failure_reasons(
        self,
        exception: BaseException,
        expected: str,
    ) -> None:
        collector = _FakeMetricsCollector()

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=[_completion(_CompletionOutcome.EXECUTION_ERROR)],
            exception=exception,
        )

        assert collector.records[0].failure_reason == expected

    @pytest.mark.parametrize(
        "completion, output, expected",
        [
            (_completion(_CompletionOutcome.EXECUTION_ERROR), None, "execution_error"),
            (_completion(_CompletionOutcome.TIMEOUT), None, "timeout"),
            (_completion(_CompletionOutcome.COMPLETED), None, "output_missing_uncaptured"),
        ],
    )
    def test_completion_state_classifies_failures_when_exception_is_absent(
        self,
        completion: _CompletionResult,
        output: _OutputResult | None,
        expected: str,
    ) -> None:
        collector = _FakeMetricsCollector()

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=[completion],
            output=output,
        )

        assert collector.records[0].failure_reason == expected

    def test_unclassified_exception_logs_type_without_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collector = _FakeMetricsCollector()
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=[_completion(_CompletionOutcome.EXECUTION_ERROR)],
            exception=RuntimeError("sensitive details"),
        )

        assert collector.records[0].failure_reason == "unclassified_error"
        warnings = [entry for entry in logger_spy.entries if entry[0] == "WARNING"]
        assert len(warnings) == 1
        assert "sensitive details" not in warnings[0][1]
        assert logger_spy.kwargs[0]["exception_type"] == "RuntimeError"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"video_id": ""}, "video_id"),
            ({"niche": ""}, "niche"),
            ({"workflow_version": ""}, "workflow_version"),
            ({"completions": []}, "completions"),
            ({"num_candidates_requested": 0}, "num_candidates_requested"),
            ({"identity_retry_count": -1}, "identity_retry_count"),
        ],
    )
    def test_validation_failures_raise_value_error(self, kwargs: dict[str, object], match: str) -> None:
        collector = _FakeMetricsCollector()
        params: dict[str, object] = {
            "video_id": "video-123",
            "niche": "gaming",
            "workflow_version": "wf-v1",
            "completions": [_completion()],
            "output": _output_result(),
        }
        params.update(kwargs)

        with pytest.raises(ValueError, match=match):
            _ComfyUIMetricsRecorder(collector).record_attempt(**params)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "completion",
        [
            _completion(queue_wait_seconds=-0.1),
            _completion(generation_seconds=-0.1),
        ],
    )
    def test_negative_timing_values_are_rejected(self, completion: _CompletionResult) -> None:
        with pytest.raises(ValueError, match="seconds"):
            _ComfyUIMetricsRecorder(_FakeMetricsCollector()).record_attempt(
                video_id="video-123",
                niche="gaming",
                workflow_version="wf-v1",
                completions=[completion],
                output=_output_result(),
            )

    def test_constructor_rejects_missing_collector(self) -> None:
        with pytest.raises(ValueError, match="collector"):
            _ComfyUIMetricsRecorder(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("error", [MetricsWriteError("disk full"), OSError("permission denied")])
    def test_metrics_write_failures_are_logged_and_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: BaseException,
    ) -> None:
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)

        _ComfyUIMetricsRecorder(_FakeMetricsCollector(error)).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=[_completion()],
            output=_output_result(),
        )

        errors = [entry for entry in logger_spy.entries if entry[0] == "ERROR"]
        assert len(errors) == 1
        assert logger_spy.kwargs[0]["video_id"] == "video-123"
        assert isinstance(logger_spy.kwargs[0]["error"], str)

    def test_max_realistic_retry_count_is_not_truncated(self) -> None:
        collector = _FakeMetricsCollector()
        completions = [
            _completion(_CompletionOutcome.EXECUTION_ERROR, generation_seconds=1.0),
            _completion(_CompletionOutcome.EXECUTION_ERROR, generation_seconds=2.0),
            _completion(_CompletionOutcome.EXECUTION_ERROR, generation_seconds=3.0),
            _completion(_CompletionOutcome.COMPLETED, generation_seconds=4.0),
        ]

        _ComfyUIMetricsRecorder(collector).record_attempt(
            video_id="video-123",
            niche="gaming",
            workflow_version="wf-v1",
            completions=completions,
            output=_output_result(),
        )

        assert collector.records[0].generation_time_seconds == [1.0, 2.0, 3.0, 4.0]
        assert collector.records[0].generation_retry_count == 3


def _workflow(graph: dict[str, object] | None = None) -> BuiltWorkflow:
    return BuiltWorkflow(
        graph={} if graph is None else graph,
        workflow_ref=WorkflowTemplateRef(
            niche="gaming",
            profile_name="PROFILE_LOW_VRAM",
            template_path="workflows/gaming.json",
            workflow_version="workflow_v1",
            template_name="gaming",
        ),
        workflow_hash="workflow-hash",
    )


def _generate_client(
    *,
    completion: _CompletionResult | None = None,
    output: _OutputResult | None = None,
    prompt_id: str = "prompt-123",
) -> ComfyUIClient:
    client = ComfyUIClient.__new__(ComfyUIClient)
    client._client_id = CLIENT_ID
    client._ws = Mock()
    client._http = Mock()
    client._http.submit_prompt.return_value = prompt_id
    client._queue_tracker = Mock()
    client._queue_tracker.await_completion.return_value = completion or _completion()
    client._output_retriever = Mock()
    client._output_retriever.retrieve.return_value = output or _output_result()
    client._metrics_recorder = Mock()
    return client


class TestGenerate:
    def test_success_path_returns_output_and_records_exact_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        completion = _completion(queue_wait_seconds=1.25, generation_seconds=2.5, used_http_fallback=True)
        output = _output_result()
        client = _generate_client(completion=completion, output=output)
        monotonic = Mock(return_value=100.0)
        monkeypatch.setattr("comfyui_client.time.monotonic", monotonic)

        result = client.generate(
            _workflow({"1": {"class_type": "KSampler"}}),
            video_id=" video-123 ",
            num_candidates_requested=4,
            identity_retry_count=2,
            peak_vram_mb=123.4,
            gpu_utilization_percent=56.7,
        )

        assert result is output
        client._ws.ensure_connected.assert_called_once_with()
        client._http.submit_prompt.assert_called_once_with({"1": {"class_type": "KSampler"}}, CLIENT_ID)
        client._queue_tracker.await_completion.assert_called_once_with("prompt-123", CLIENT_ID)
        client._output_retriever.retrieve.assert_called_once_with(completion, prompt_id="prompt-123")
        client._metrics_recorder.record_attempt.assert_called_once()
        kwargs = client._metrics_recorder.record_attempt.call_args.kwargs
        assert kwargs == {
            "video_id": "video-123",
            "niche": "gaming",
            "workflow_version": "workflow_v1",
            "profile_name": "PROFILE_LOW_VRAM",
            "workflow_hash": "workflow-hash",
            "completions": (completion,),
            "output": output,
            "exception": None,
            "num_candidates_requested": 4,
            "identity_retry_count": 2,
            "peak_vram_mb": 123.4,
            "gpu_utilization_percent": 56.7,
            "attempt_started_at": 100.0,
        }

    def test_connection_failure_propagates_and_records_synthetic_completion(self) -> None:
        client = _generate_client()
        error = ComfyUIConnectionError("cannot connect")
        client._ws.ensure_connected.side_effect = error

        with pytest.raises(ComfyUIConnectionError) as raised:
            client.generate(_workflow(), video_id="video-123")

        assert raised.value is error
        client._http.submit_prompt.assert_not_called()
        client._queue_tracker.await_completion.assert_not_called()
        client._output_retriever.retrieve.assert_not_called()
        kwargs = client._metrics_recorder.record_attempt.call_args.kwargs
        assert kwargs["exception"] is error
        assert kwargs["output"] is None
        assert len(kwargs["completions"]) == 1
        synthetic = kwargs["completions"][0]
        assert synthetic.outcome is _CompletionOutcome.EXECUTION_ERROR
        assert synthetic.queue_wait_seconds == 0.0
        assert synthetic.generation_seconds == 0.0

    def test_submission_failure_is_translated_and_chained(self) -> None:
        client = _generate_client()
        transport_error = _ComfyUIHTTPError("submit failed")
        client._http.submit_prompt.side_effect = transport_error

        with pytest.raises(ComfyUIConnectionError) as raised:
            client.generate(_workflow(), video_id="video-123")

        assert raised.value.__cause__ is transport_error
        client._queue_tracker.await_completion.assert_not_called()
        client._output_retriever.retrieve.assert_not_called()
        assert client._metrics_recorder.record_attempt.call_args.kwargs["exception"] is raised.value

    @pytest.mark.parametrize(
        "payload",
        [
            {"exception_type": "OutOfMemoryError", "exception_message": "failed"},
            {"exception_type": "RuntimeError", "exception_message": "CUDA out of memory"},
            {"exception_type": "RuntimeError", "exception_message": "Tried to allocate: out of memory"},
        ],
    )
    def test_execution_error_oom_classification_raises_vram_exhausted_without_retry(
        self,
        payload: dict[str, str],
    ) -> None:
        completion = _completion(_CompletionOutcome.EXECUTION_ERROR)
        completion = _CompletionResult(
            outcome=completion.outcome,
            history_payload=None,
            error_payload=payload,
            queue_wait_seconds=completion.queue_wait_seconds,
            generation_seconds=completion.generation_seconds,
            used_http_fallback=completion.used_http_fallback,
        )
        client = _generate_client(completion=completion)

        with pytest.raises(VRAMExhaustedError) as raised:
            client.generate(_workflow(), video_id="video-123")

        assert str(raised.value) == payload["exception_message"]
        client._queue_tracker.await_completion.assert_called_once_with("prompt-123", CLIENT_ID)
        client._output_retriever.retrieve.assert_not_called()
        assert client._metrics_recorder.record_attempt.call_args.kwargs["exception"] is raised.value

    @pytest.mark.parametrize(
        "payload, expected_message",
        [
            ({"exception_message": "KeyError: 'checkpoint'"}, "KeyError: 'checkpoint'"),
            (None, "ComfyUI execution error"),
            ({}, "ComfyUI execution error"),
        ],
    )
    def test_execution_error_non_oom_raises_queue_error(
        self,
        payload: dict[str, str] | None,
        expected_message: str,
    ) -> None:
        completion = _CompletionResult(
            outcome=_CompletionOutcome.EXECUTION_ERROR,
            history_payload=None,
            error_payload=payload,
            queue_wait_seconds=1.0,
            generation_seconds=2.0,
            used_http_fallback=False,
        )
        client = _generate_client(completion=completion)

        with pytest.raises(ComfyUIQueueError, match=expected_message) as raised:
            client.generate(_workflow(), video_id="video-123")

        client._output_retriever.retrieve.assert_not_called()
        assert client._metrics_recorder.record_attempt.call_args.kwargs["exception"] is raised.value

    def test_timeout_cancels_prompt_and_raises_timeout(self) -> None:
        completion = _completion(_CompletionOutcome.TIMEOUT)
        client = _generate_client(completion=completion)

        with pytest.raises(ComfyUITimeoutError) as raised:
            client.generate(_workflow(), video_id="video-123")

        client._http.interrupt.assert_called_once_with()
        client._http.delete_from_queue.assert_called_once_with("prompt-123")
        client._output_retriever.retrieve.assert_not_called()
        assert client._metrics_recorder.record_attempt.call_args.kwargs["exception"] is raised.value

    def test_timeout_cancellation_failures_are_swallowed(self) -> None:
        completion = _completion(_CompletionOutcome.TIMEOUT)
        client = _generate_client(completion=completion)
        client._http.interrupt.side_effect = _ComfyUIHTTPError("interrupt failed")
        client._http.delete_from_queue.side_effect = _ComfyUIHTTPError("delete failed")

        with pytest.raises(ComfyUITimeoutError):
            client.generate(_workflow(), video_id="video-123")

        client._http.interrupt.assert_called_once_with()
        client._http.delete_from_queue.assert_called_once_with("prompt-123")
        client._metrics_recorder.record_attempt.assert_called_once()

    @pytest.mark.parametrize(
        "error",
        [
            OutputHistoryError("history"),
            NoOutputImageError("none"),
            OutputDownloadError("download"),
            MissingOutputFileError("missing"),
            CorruptImageError("corrupt"),
            UnsupportedImageFormatError("unsupported"),
        ],
    )
    def test_retrieval_failures_propagate_unmodified_and_record_metrics(
        self,
        error: BaseException,
    ) -> None:
        completion = _completion()
        client = _generate_client(completion=completion)
        client._output_retriever.retrieve.side_effect = error

        with pytest.raises(type(error)) as raised:
            client.generate(_workflow(), video_id="video-123")

        assert raised.value is error
        client._output_retriever.retrieve.assert_called_once_with(completion, prompt_id="prompt-123")
        kwargs = client._metrics_recorder.record_attempt.call_args.kwargs
        assert kwargs["completions"] == (completion,)
        assert kwargs["output"] is None
        assert kwargs["exception"] is error

    @pytest.mark.parametrize("video_id", ["", "   "])
    def test_blank_video_id_fails_before_network_and_metrics(self, video_id: str) -> None:
        client = _generate_client()

        with pytest.raises(ValueError, match="video_id"):
            client.generate(_workflow(), video_id=video_id)

        client._ws.ensure_connected.assert_not_called()
        client._http.submit_prompt.assert_not_called()
        client._metrics_recorder.record_attempt.assert_not_called()

    def test_default_metrics_arguments_and_empty_graph_passthrough(self) -> None:
        client = _generate_client()

        client.generate(_workflow({}), video_id="video-123")

        client._http.submit_prompt.assert_called_once_with({}, CLIENT_ID)
        kwargs = client._metrics_recorder.record_attempt.call_args.kwargs
        assert kwargs["num_candidates_requested"] == 1
        assert kwargs["identity_retry_count"] == 0
        assert kwargs["peak_vram_mb"] is None
        assert kwargs["gpu_utilization_percent"] is None

    def test_two_sequential_calls_reconnect_and_use_fresh_timing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = _completion()
        second = _completion(queue_wait_seconds=3.0)
        client = _generate_client(completion=first)
        client._http.submit_prompt.side_effect = ["prompt-1", "prompt-2"]
        client._queue_tracker.await_completion.side_effect = [first, second]
        monkeypatch.setattr("comfyui_client.time.monotonic", Mock(side_effect=[10.0, 20.0]))

        client.generate(_workflow(), video_id="video-123")
        client.generate(_workflow(), video_id="video-123")

        assert client._ws.ensure_connected.call_count == 2
        starts = [
            call.kwargs["attempt_started_at"]
            for call in client._metrics_recorder.record_attempt.call_args_list
        ]
        assert starts == [10.0, 20.0]

    def test_generate_logs_start_success_and_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger_spy = _FakeLogger()
        monkeypatch.setattr("comfyui_client.logger", logger_spy)
        client = _generate_client()

        client.generate(_workflow(), video_id="video-123")

        assert any(level == "INFO" and "Starting ComfyUI generation" in message for level, message in logger_spy.entries)
        assert any(level == "INFO" and "generation completed" in message for level, message in logger_spy.entries)

        logger_spy.entries.clear()
        error_completion = _CompletionResult(
            outcome=_CompletionOutcome.EXECUTION_ERROR,
            history_payload=None,
            error_payload={"exception_message": "bad node"},
            queue_wait_seconds=0.0,
            generation_seconds=0.0,
            used_http_fallback=False,
        )
        error_client = _generate_client(completion=error_completion)
        with pytest.raises(ComfyUIQueueError):
            error_client.generate(_workflow(), video_id="video-123")

        assert any(level == "ERROR" and "generation failed" in message for level, message in logger_spy.entries)

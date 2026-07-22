from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import Mock

import pytest
import requests
import websocket

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from comfyui_client import (  # noqa: E402
    ComfyUIEvent,
    SystemStats,
    _CompletionOutcome,
    _ComfyUIHTTPError,
    _ComfyUIHTTPTransport,
    _ComfyUIWebSocketTransport,
    _QueueTracker,
)
from module7_exceptions import ComfyUIConnectionError  # noqa: E402


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

    def debug(self, message: str, **_kwargs: object) -> None:
        self.entries.append(("DEBUG", message))

    def info(self, message: str, **_kwargs: object) -> None:
        self.entries.append(("INFO", message))

    def warning(self, message: str, **_kwargs: object) -> None:
        self.entries.append(("WARNING", message))

    def error(self, message: str, **_kwargs: object) -> None:
        self.entries.append(("ERROR", message))


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

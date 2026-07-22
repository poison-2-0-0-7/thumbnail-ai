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
    _ComfyUIHTTPError,
    _ComfyUIHTTPTransport,
    _ComfyUIWebSocketTransport,
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

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import requests

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from comfyui_client import SystemStats, _ComfyUIHTTPError, _ComfyUIHTTPTransport  # noqa: E402


BASE_URL = "http://127.0.0.1:8188"
TIMEOUT_SECONDS = 12.5


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


@pytest.mark.parametrize("payload, expected", [({"prompt-123": {"outputs": {}}}, {"prompt-123": {"outputs": {}}}), ({}, None)])
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
    ],
)
def test_invalid_response_shapes_raise_typed_transport_error(
    session: Mock, transport: _ComfyUIHTTPTransport, method_call: object, payload: object
) -> None:
    session.request.return_value = _response(payload=payload)

    with pytest.raises(_ComfyUIHTTPError):
        method_call(transport)  # type: ignore[operator]


def test_http_status_failure_is_translated_with_original_cause(
    session: Mock, transport: _ComfyUIHTTPTransport
) -> None:
    http_error = requests.HTTPError("404 Client Error")
    session.request.return_value = _response(payload={}, request_error=http_error)

    with pytest.raises(_ComfyUIHTTPError) as raised:
        transport.queue_status()
    assert raised.value.__cause__ is http_error


@pytest.mark.parametrize(
    "base_url, timeout_seconds",
    [("", 1.0), (BASE_URL, 0.0), (BASE_URL, -1.0)],
)
def test_invalid_transport_configuration_is_rejected(
    session: Mock, base_url: str, timeout_seconds: float
) -> None:
    with pytest.raises(ValueError):
        _ComfyUIHTTPTransport(base_url, session, timeout_seconds)

"""
test_comfyui_manager.py
=======================

Unit tests for ComfyUIProcessManager covering:
- ComfyUI already running
- Successful startup
- Startup timeout
- Failed startup (premature exit)
- API health polling
- Duplicate instance prevention
- Shutdown on exit behavior
- Configuration options
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODULES_DIR = _PROJECT_ROOT / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from comfyui_manager import ComfyUIProcessManager
from module7_exceptions import ComfyUIStartupError


def test_api_health_polling_success() -> None:
    session = MagicMock(spec=requests.Session)
    response_mock = MagicMock()
    response_mock.status_code = 200
    session.get.return_value = response_mock

    manager = ComfyUIProcessManager(host="127.0.0.1", port=8188, session=session)
    assert manager.is_healthy() is True
    session.get.assert_called_once_with("http://127.0.0.1:8188/system_stats", timeout=2.0)


def test_api_health_polling_failure() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("Connection refused")

    manager = ComfyUIProcessManager(host="127.0.0.1", port=8188, session=session)
    assert manager.is_healthy() is False


def test_comfyui_already_running(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = MagicMock(status_code=200)

    with patch("subprocess.Popen") as mock_popen:
        manager = ComfyUIProcessManager(
            host="127.0.0.1",
            port=8188,
            session=session,
            log_path=tmp_path / "comfy.log",
        )
        result = manager.ensure_started()

        assert result is True
        assert manager.is_spawned is False
        mock_popen.assert_not_called()


def test_successful_startup(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    # First call returns False (not running), second call returns True (healthy after launch)
    session.get.side_effect = [
        requests.ConnectionError("Refused"),
        MagicMock(status_code=200),
    ]

    proc_mock = MagicMock()
    proc_mock.poll.return_value = None
    proc_mock.pid = 12345

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        manager = ComfyUIProcessManager(
            host="127.0.0.1",
            port=8188,
            start_command="python main.py --listen 127.0.0.1",
            startup_timeout=5.0,
            healthcheck_interval=0.01,
            session=session,
            log_path=tmp_path / "comfy.log",
        )
        result = manager.ensure_started()

        assert result is True
        assert manager.is_spawned is True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert "main.py" in args[0] or "python" in args[0][0]


def test_startup_timeout(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("Refused")

    proc_mock = MagicMock()
    proc_mock.poll.return_value = None
    proc_mock.pid = 999

    with patch("subprocess.Popen", return_value=proc_mock):
        manager = ComfyUIProcessManager(
            host="127.0.0.1",
            port=8188,
            startup_timeout=0.05,
            healthcheck_interval=0.01,
            session=session,
            log_path=tmp_path / "comfy.log",
        )
        with pytest.raises(ComfyUIStartupError, match="timed out"):
            manager.ensure_started()

        proc_mock.terminate.assert_called_once()


def test_failed_startup_premature_exit(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("Refused")

    proc_mock = MagicMock()
    proc_mock.poll.return_value = 1
    proc_mock.returncode = 1

    log_file = tmp_path / "comfy.log"
    log_file.write_text("Error: invalid argument\nFailed to start server", encoding="utf-8")

    with patch("subprocess.Popen", return_value=proc_mock):
        manager = ComfyUIProcessManager(
            host="127.0.0.1",
            port=8188,
            startup_timeout=5.0,
            healthcheck_interval=0.01,
            session=session,
            log_path=log_file,
        )
        with pytest.raises(ComfyUIStartupError, match="exited prematurely with code 1"):
            manager.ensure_started()


def test_duplicate_instance_prevention(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        requests.ConnectionError("Refused"),
        MagicMock(status_code=200),
        MagicMock(status_code=200),
    ]

    proc_mock = MagicMock()
    proc_mock.poll.return_value = None

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        manager = ComfyUIProcessManager(
            host="127.0.0.1",
            port=8188,
            session=session,
            log_path=tmp_path / "comfy.log",
        )
        res1 = manager.ensure_started()
        assert res1 is True
        assert mock_popen.call_count == 1

        # Second call should reuse the existing spawned process without calling Popen again
        res2 = manager.ensure_started()
        assert res2 is True
        assert mock_popen.call_count == 1


def test_shutdown_on_exit_behavior(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        requests.ConnectionError("Refused"),
        MagicMock(status_code=200),
    ]

    proc_mock = MagicMock()
    proc_mock.poll.return_value = None

    with patch("subprocess.Popen", return_value=proc_mock):
        manager = ComfyUIProcessManager(
            shutdown_on_exit=True,
            session=session,
            log_path=tmp_path / "comfy.log",
        )
        with manager:
            assert manager.is_spawned is True

        proc_mock.terminate.assert_called_once()


def test_shutdown_on_exit_disabled_leaves_process_running(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        requests.ConnectionError("Refused"),
        MagicMock(status_code=200),
    ]

    proc_mock = MagicMock()
    proc_mock.poll.return_value = None

    with patch("subprocess.Popen", return_value=proc_mock):
        manager = ComfyUIProcessManager(
            shutdown_on_exit=False,
            session=session,
            log_path=tmp_path / "comfy.log",
        )
        with manager:
            assert manager.is_spawned is True

        proc_mock.terminate.assert_not_called()


def test_custom_configuration(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = MagicMock(status_code=200)

    custom_cwd = tmp_path / "comfy_dir"
    custom_cwd.mkdir()
    custom_exe = tmp_path / "python.exe"
    custom_exe.touch()

    manager = ComfyUIProcessManager(
        enabled=True,
        host="192.168.1.50",
        port=9999,
        working_directory=custom_cwd,
        start_command="python custom_app.py",
        executable=custom_exe,
        startup_timeout=30.0,
        healthcheck_interval=0.5,
        shutdown_on_exit=True,
        log_path=tmp_path / "custom.log",
        session=session,
    )

    assert manager.host == "192.168.1.50"
    assert manager.port == 9999
    assert manager.working_directory == custom_cwd
    assert manager.executable == custom_exe
    assert manager.startup_timeout == 30.0
    assert manager.healthcheck_interval == 0.5
    assert manager.shutdown_on_exit is True
    assert manager.base_url == "http://192.168.1.50:9999"

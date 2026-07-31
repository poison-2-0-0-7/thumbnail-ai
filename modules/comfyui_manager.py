"""
comfyui_manager.py
==================

Automatic ComfyUI process lifecycle manager.

Provides process detection, health polling, automated startup, stdout/stderr
log capture, and optional process cleanup/shutdown upon pipeline completion.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import requests
from loguru import logger

from config import (
    COMFYUI_ENABLED,
    COMFYUI_EXECUTABLE,
    COMFYUI_HEALTHCHECK_INTERVAL,
    COMFYUI_HOST,
    COMFYUI_PORT,
    COMFYUI_PROCESS_LOG_PATH,
    COMFYUI_SHUTDOWN_ON_EXIT,
    COMFYUI_START_COMMAND,
    COMFYUI_STARTUP_TIMEOUT,
    COMFYUI_WORKING_DIRECTORY,
)
from module7_exceptions import ComfyUIStartupError


class ComfyUIProcessManager:
    """
    Lifecycle manager for local ComfyUI instance.

    Detects existing instances, handles automated process spawning,
    captures stdout/stderr to project logs, polls API health status,
    and provides optional shutdown on exit.
    """

    def __init__(
        self,
        *,
        enabled: bool = COMFYUI_ENABLED,
        host: str = COMFYUI_HOST,
        port: int = COMFYUI_PORT,
        working_directory: str | Path | None = COMFYUI_WORKING_DIRECTORY,
        start_command: str | Sequence[str] | None = COMFYUI_START_COMMAND,
        executable: str | Path | None = COMFYUI_EXECUTABLE,
        startup_timeout: float = COMFYUI_STARTUP_TIMEOUT,
        healthcheck_interval: float = COMFYUI_HEALTHCHECK_INTERVAL,
        shutdown_on_exit: bool = COMFYUI_SHUTDOWN_ON_EXIT,
        log_path: str | Path | None = COMFYUI_PROCESS_LOG_PATH,
        session: requests.Session | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.host = str(host).strip()
        self.port = int(port)
        self.working_directory = (
            Path(working_directory).resolve() if working_directory else None
        )
        self.start_command = start_command or "python main.py"
        if executable:
            exe_p = Path(executable)
            if not exe_p.is_absolute() and self.working_directory:
                exe_p = (self.working_directory / exe_p).resolve()
            self.executable = exe_p.resolve()
        elif self.working_directory:
            dot_venv_exe = (self.working_directory / ".venv" / "Scripts" / "python.exe").resolve()
            venv_exe = (self.working_directory / "venv" / "Scripts" / "python.exe").resolve()
            if dot_venv_exe.exists():
                self.executable = dot_venv_exe
            elif venv_exe.exists():
                self.executable = venv_exe
            else:
                self.executable = dot_venv_exe
        else:
            self.executable = None
        self.startup_timeout = float(startup_timeout)
        self.healthcheck_interval = float(healthcheck_interval)
        self.shutdown_on_exit = bool(shutdown_on_exit)
        self.log_path = Path(log_path) if log_path else COMFYUI_PROCESS_LOG_PATH
        self._session = session or requests.Session()

        self._process: subprocess.Popen | None = None
        self._was_spawned: bool = False
        self._log_file_handle = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_spawned(self) -> bool:
        return self._was_spawned and self._process is not None and self._process.poll() is None

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """
        Check if the ComfyUI API endpoint is available and responsive.
        """
        url = f"{self.base_url}/system_stats"
        try:
            resp = self._session.get(url, timeout=timeout)
            return resp.status_code == 200
        except (requests.RequestException, OSError, ValueError):
            return False

    def ensure_started(self) -> bool:
        """
        Detect running ComfyUI instance or launch one if not currently running.

        Returns True if ComfyUI is healthy and ready to process requests.
        Raises ComfyUIStartupError if startup fails or times out.
        """
        if os.getenv("_COMFYUI_MANAGER_SPAWNED") == "1":
            raise ComfyUIStartupError(
                "Recursive ComfyUIProcessManager spawn detected. "
                "COMFYUI_START_COMMAND must point to your local ComfyUI installation, "
                "not thumbnail-ai main.py."
            )

        if not self.enabled:
            logger.info("ComfyUI auto-start is disabled (COMFYUI_ENABLED=False).")
            if self.is_healthy():
                return True
            logger.warning(
                "ComfyUI auto-start is disabled and no active instance detected at {url}",
                url=self.base_url,
            )
            return False

        # 1. Detect if ComfyUI is already running
        if self.is_healthy():
            logger.info(
                "ComfyUI is already running at {url} — reusing existing instance.",
                url=self.base_url,
            )
            self._was_spawned = False
            return True

        # Check if already spawned by this instance and initializing
        if self._process is not None and self._process.poll() is None:
            logger.info(
                "ComfyUI process (PID {pid}) already launched, waiting for health check...",
                pid=self._process.pid,
            )
            return self._wait_for_health()

        # 2. Launch ComfyUI process
        logger.info(
            "ComfyUI is not running. Launching process on {url}...",
            url=self.base_url,
        )

        if isinstance(self.start_command, str):
            cmd_args = shlex.split(self.start_command, posix=(os.name != "nt"))
        else:
            cmd_args = list(self.start_command)

        if not cmd_args:
            cmd_args = ["main.py"]

        resolved_exe: Path | None = self.executable
        if resolved_exe and not resolved_exe.is_absolute() and self.working_directory:
            resolved_exe = (self.working_directory / resolved_exe).resolve()

        if not resolved_exe and self.working_directory:
            dot_venv_exe = (self.working_directory / ".venv" / "Scripts" / "python.exe").resolve()
            venv_exe = (self.working_directory / "venv" / "Scripts" / "python.exe").resolve()
            if dot_venv_exe.exists():
                resolved_exe = dot_venv_exe
            elif venv_exe.exists():
                resolved_exe = venv_exe
            else:
                resolved_exe = dot_venv_exe

        if resolved_exe:
            exe_str = str(resolved_exe.resolve())
            if cmd_args[0] in ("python", "python3"):
                cmd_args[0] = exe_str
            elif cmd_args[0].endswith(".py"):
                cmd_args.insert(0, exe_str)
            else:
                cmd_args[0] = exe_str

        cwd = str(self.working_directory.resolve()) if self.working_directory else None
        resolved_exe_path = Path(cmd_args[0]).resolve() if cmd_args else None
        exe_exists = resolved_exe_path.exists() if resolved_exe_path else False
        current_cwd = os.getcwd()
        env_path = os.environ.get("PATH", "")
        cmd_str = " ".join(f'"{a}"' if " " in a else a for a in cmd_args)

        logger.info("Resolved executable: {exe}", exe=resolved_exe_path)
        logger.info("Resolved working directory: {cwd}", cwd=cwd)
        logger.info("Resolved startup command: {cmd}", cmd=cmd_str)
        logger.info("Python executable exists: {exists}", exists=exe_exists)
        logger.info("Current cwd: {cwd}", cwd=current_cwd)
        logger.info("Environment PATH: {path}", path=env_path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._log_file_handle = open(self.log_path, "a", encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Could not open log file {path} for process capture: {exc}",
                path=self.log_path,
                exc=exc,
            )
            self._log_file_handle = subprocess.DEVNULL

        spawn_env = {**os.environ, "_COMFYUI_MANAGER_SPAWNED": "1"}

        try:
            self._process = subprocess.Popen(
                cmd_args,
                cwd=cwd,
                env=spawn_env,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,
            )
            self._was_spawned = True
            logger.info("PID: {pid}", pid=self._process.pid)
            logger.info("Return code: {rc}", rc=self._process.poll())
            logger.info("stdout: {log}", log=self.log_path)
            logger.info("stderr: {log}", log=self.log_path)
        except Exception as exc:
            self._cleanup_log_handle()
            raise ComfyUIStartupError(
                f"Failed to spawn ComfyUI process '{' '.join(cmd_args)}': {exc}"
            ) from exc

        # 3. Wait for API to become healthy
        return self._wait_for_health()

    def _wait_for_health(self) -> bool:
        start_time = time.monotonic()
        deadline = start_time + self.startup_timeout

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                code = self._process.returncode
                log_snippet = self._get_recent_log_snippet()
                pid = self._process.pid
                logger.error("PID: {pid}", pid=pid)
                logger.error("Return code: {code}", code=code)
                logger.error("stdout: {log}", log=log_snippet)
                logger.error("stderr: {log}", log=log_snippet)
                self._cleanup_process()
                raise ComfyUIStartupError(
                    f"ComfyUI process exited prematurely with code {code}. "
                    f"Recent log output:\n{log_snippet}"
                )

            if self.is_healthy():
                logger.info(
                    "ComfyUI API is healthy and ready at {url} (PID: {pid}).",
                    url=self.base_url,
                    pid=self._process.pid if self._process else "unknown",
                )
                return True

            time.sleep(self.healthcheck_interval)

        log_snippet = self._get_recent_log_snippet()
        self._cleanup_process()
        raise ComfyUIStartupError(
            f"ComfyUI startup timed out after {self.startup_timeout}s on {self.base_url}. "
            f"Recent log output:\n{log_snippet}"
        )

    def _get_recent_log_snippet(self, max_lines: int = 20) -> str:
        if self._log_file_handle and hasattr(self._log_file_handle, "flush"):
            try:
                self._log_file_handle.flush()
            except OSError:
                pass
        if not self.log_path.exists():
            return "(No log file found)"
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:]) if lines else "(Log file empty)"
        except OSError as exc:
            return f"(Could not read log file: {exc})"

    def stop(self) -> None:
        """
        Terminate the spawned ComfyUI process if it was started by this manager.
        """
        if self._was_spawned and self._process is not None and self._process.poll() is None:
            logger.info(
                "Stopping spawned ComfyUI process (PID: {pid})...",
                pid=self._process.pid,
            )
            self._cleanup_process()
            logger.info("Spawned ComfyUI process stopped.")

    def _cleanup_process(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        self._process.kill()
                        self._process.wait(timeout=2.0)
                    except OSError:
                        pass
            self._process = None
        self._cleanup_log_handle()

    def _cleanup_log_handle(self) -> None:
        if self._log_file_handle and self._log_file_handle is not subprocess.DEVNULL:
            try:
                self._log_file_handle.close()
            except OSError:
                pass
            self._log_file_handle = None

    def __enter__(self) -> ComfyUIProcessManager:
        self.ensure_started()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.shutdown_on_exit:
            self.stop()


__all__ = ["ComfyUIProcessManager"]

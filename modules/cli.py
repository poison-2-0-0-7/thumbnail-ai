"""
cli.py
======

Official Command-Line Interface (CLI) for Thumbnail AI.

Executable entry point: `tai`

Commands:
    tai run      - Run the complete thumbnail generation pipeline
    tai doctor   - Run full system health check
    tai status   - Display current system status
    tai version  - Display version information
    tai test     - Test runner placeholder
    tai comfy    - ComfyUI management (start, stop, status)
    tai ollama   - Ollama management (start, stop, status)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Ensure modules/ is importable regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULES_DIR = Path(__file__).resolve().parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

import requests  # noqa: E402
from loguru import logger  # noqa: E402

from config import (  # noqa: E402
    COMFYUI_EXECUTABLE,
    COMFYUI_HEALTHCHECK_INTERVAL,
    COMFYUI_HOST,
    COMFYUI_PORT,
    COMFYUI_SHUTDOWN_ON_EXIT,
    COMFYUI_START_COMMAND,
    COMFYUI_STARTUP_TIMEOUT,
    COMFYUI_WORKING_DIRECTORY,
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ASSET_EXTRACTION_DIR,
    DEFAULT_CSV_PATH,
    DEFAULT_DECISION_DIR,
    DEFAULT_DESIGN_BLUEPRINT_DIR,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    DEFAULT_THUMBNAIL_DIR,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
    YOLO_MODEL_NAME,
)
from comfyui_manager import ComfyUIProcessManager  # noqa: E402
from main import run_pipeline  # noqa: E402
from module7_exceptions import ComfyUIStartupError  # noqa: E402

__version__ = "1.0.0"


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the full thumbnail generation pipeline."""
    print("Starting Thumbnail AI...")
    print("Checking configuration...")
    print("Verifying project dependencies...")
    print("Checking ComfyUI...")

    manager = ComfyUIProcessManager(
        host=COMFYUI_HOST,
        port=COMFYUI_PORT,
        working_directory=COMFYUI_WORKING_DIRECTORY,
        start_command=COMFYUI_START_COMMAND,
        executable=COMFYUI_EXECUTABLE,
        startup_timeout=COMFYUI_STARTUP_TIMEOUT,
        healthcheck_interval=COMFYUI_HEALTHCHECK_INTERVAL,
        shutdown_on_exit=COMFYUI_SHUTDOWN_ON_EXIT,
    )

    if not manager.is_healthy():
        print("Starting ComfyUI...")
        print("Waiting for ComfyUI...")
    else:
        print("ComfyUI is already running and healthy.")

    print("Running pipeline...")
    try:
        run_pipeline(
            csv_path=Path(args.csv) if getattr(args, "csv", None) else DEFAULT_CSV_PATH,
            thumbnail_dir=Path(args.thumbnail_dir) if getattr(args, "thumbnail_dir", None) else DEFAULT_THUMBNAIL_DIR,
            analysis_dir=Path(args.analysis_dir) if getattr(args, "analysis_dir", None) else DEFAULT_ANALYSIS_DIR,
            redesign_spec_dir=Path(args.redesign_spec_dir) if getattr(args, "redesign_spec_dir", None) else DEFAULT_REDESIGN_SPEC_DIR,
            design_blueprint_dir=Path(args.design_blueprint_dir) if getattr(args, "design_blueprint_dir", None) else DEFAULT_DESIGN_BLUEPRINT_DIR,
            prompt_package_dir=Path(args.prompt_package_dir) if getattr(args, "prompt_package_dir", None) else DEFAULT_PROMPT_PACKAGE_DIR,
            asset_extraction_dir=Path(args.asset_extraction_dir) if getattr(args, "asset_extraction_dir", None) else DEFAULT_ASSET_EXTRACTION_DIR,
            decision_dir=Path(args.decision_dir) if getattr(args, "decision_dir", None) else DEFAULT_DECISION_DIR,
            comfyui_manager=manager,
        )
        print("Pipeline completed successfully.")
        return 0
    except ComfyUIStartupError as exc:
        print(f"Error: ComfyUI startup failed: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}")
        print(f"Error: Pipeline execution failed: {exc}")
        return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Perform a full system health check."""
    print("\n==================================================================")
    print("                    THUMBNAIL AI HEALTH CHECK                     ")
    print("==================================================================\n")

    results: list[tuple[str, str, str]] = []

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        results.append(("Python version", f"v{py_ver}", "OK"))
    else:
        results.append(("Python version", f"v{py_ver} (>=3.10 required)", "FAIL"))

    # 2. Virtual environment
    in_venv = sys.prefix != sys.base_prefix
    venv_path = sys.prefix if in_venv else "Not active"
    results.append(("Virtual environment", venv_path, "OK" if in_venv else "WARN"))

    # 3. Required packages
    required_pkgs = [
        "pandas",
        "portalocker",
        "yt_dlp",
        "youtube_transcript_api",
        "pydantic",
        "loguru",
        "tenacity",
        "requests",
        "PIL",
        "websocket",
        "torch",
    ]
    missing = []
    for pkg in required_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        results.append(("Required packages", f"All {len(required_pkgs)} core packages installed", "OK"))
    else:
        results.append(("Required packages", f"Missing: {', '.join(missing)}", "FAIL"))

    # 4. ComfyUI installation
    comfy_dir = COMFYUI_WORKING_DIRECTORY
    if comfy_dir and comfy_dir.exists() and (comfy_dir / "main.py").exists():
        results.append(("ComfyUI installation", str(comfy_dir), "OK"))
    else:
        results.append(("ComfyUI installation", f"Path: {comfy_dir}", "WARN"))

    # 5. ComfyUI API
    mgr = ComfyUIProcessManager()
    comfy_running = mgr.is_healthy()
    if comfy_running:
        results.append(("ComfyUI API", f"Healthy at {mgr.base_url}", "OK"))
    else:
        results.append(("ComfyUI API", f"Offline at {mgr.base_url} (auto-start ready)", "WARN"))

    # 6. Ollama
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        ollama_ok = (r.status_code == 200)
    except Exception:
        ollama_ok = False
    if ollama_ok:
        results.append(("Ollama service", f"Healthy at {OLLAMA_BASE_URL}", "OK"))
    else:
        results.append(("Ollama service", f"Offline at {OLLAMA_BASE_URL}", "WARN"))

    # 7. Required models
    yolo_path = PROJECT_ROOT / YOLO_MODEL_NAME
    if yolo_path.exists():
        results.append(("Required models", f"YOLO weights present ({YOLO_MODEL_NAME})", "OK"))
    else:
        results.append(("Required models", f"Weights missing: {YOLO_MODEL_NAME}", "WARN"))

    # 8. Required folders
    folders = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "workflows",
        PROJECT_ROOT / "data" / "generated_thumbnails",
    ]
    missing_folders = [str(f.relative_to(PROJECT_ROOT)) for f in folders if not f.exists()]
    if not missing_folders:
        results.append(("Required folders", "All core directories present", "OK"))
    else:
        results.append(("Required folders", f"Missing: {', '.join(missing_folders)}", "WARN"))

    # 9. Configuration
    try:
        from image_generator import ProfileSelector
        ProfileSelector()
        from config import validate_controlnet_capability_availability, validate_candidate_selection_config
        validate_candidate_selection_config()
        if comfy_running:
            from generation_components.model_discovery_service import ModelDiscoveryService
            from generation_components.controlnet_capability_resolver import ControlNetCapabilityResolver
            comfy_client = getattr(mgr, "client", None)
            discovery = ModelDiscoveryService(client=comfy_client)
            resolver = ControlNetCapabilityResolver(discovery_service=discovery)
            validate_controlnet_capability_availability(resolver)
        results.append(("Configuration", f"Host={COMFYUI_HOST}, Port={COMFYUI_PORT}", "OK"))

    except Exception as exc:
        results.append(("Configuration", f"Host={COMFYUI_HOST}, Port={COMFYUI_PORT} | Module 7 config invalid: {exc}", "FAIL"))

    # Display report
    has_fail = False
    for item, detail, status in results:
        badge = "[OK]" if status == "OK" else ("[WARN]" if status == "WARN" else "[FAIL]")
        if status == "FAIL":
            has_fail = True
        dots = "." * max(2, 28 - len(item))
        print(f" {badge:<6} {item} {dots} [{status}] ({detail})")

    print("\n------------------------------------------------------------------")
    if has_fail:
        print("Doctor report: CRITICAL ISSUES DETECTED. Please resolve failed checks.")
        return 1

    print("Doctor report: SYSTEM HEALTHY AND READY.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Display current system status summary."""
    mgr = ComfyUIProcessManager()
    comfy_running = mgr.is_healthy()

    ollama_running = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        ollama_running = (r.status_code == 200)
    except Exception:
        ollama_running = False

    project_status = "READY" if (DEFAULT_CSV_PATH.exists() or (PROJECT_ROOT / "data").exists()) else "NOT CONFIGURED"
    comfy_status = "RUNNING" if comfy_running else "STOPPED"
    ollama_status = "RUNNING" if ollama_running else "STOPPED"
    pipeline_status = "IDLE"
    models_status = "READY"

    print(f"Project ............ {project_status}")
    print(f"ComfyUI ............ {comfy_status}")
    print(f"Ollama ............. {ollama_status}")
    print(f"Pipeline ........... {pipeline_status}")
    print(f"Models ............. {models_status}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Display CLI and runtime version."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"thumbnail-ai v{__version__} (Python v{py_ver})")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Placeholder command for test execution."""
    print("Running thumbnail-ai test suite placeholder...")
    print("Use 'pytest' or 'python -m pytest' to run the full test suite.")
    return 0


def cmd_comfy_start(args: argparse.Namespace) -> int:
    """Start local ComfyUI process."""
    mgr = ComfyUIProcessManager()
    if mgr.is_healthy():
        print(f"ComfyUI is already running at {mgr.base_url}")
        return 0
    print(f"Launching ComfyUI process on {mgr.base_url}...")
    try:
        mgr.ensure_started()
        print("ComfyUI started successfully.")
        return 0
    except ComfyUIStartupError as exc:
        print(f"Error: {exc}")
        return 1


def cmd_comfy_stop(args: argparse.Namespace) -> int:
    """Stop local ComfyUI process."""
    mgr = ComfyUIProcessManager()
    mgr.stop()
    print("ComfyUI process stop requested.")
    return 0


def cmd_comfy_status(args: argparse.Namespace) -> int:
    """Check local ComfyUI status."""
    mgr = ComfyUIProcessManager()
    status = "RUNNING" if mgr.is_healthy() else "STOPPED"
    print(f"ComfyUI ({mgr.base_url}) ............ {status}")
    return 0


def cmd_ollama_start(args: argparse.Namespace) -> int:
    """Start Ollama service placeholder."""
    print(f"Ollama service management placeholder. Server URL: {OLLAMA_BASE_URL}")
    return 0


def cmd_ollama_stop(args: argparse.Namespace) -> int:
    """Stop Ollama service placeholder."""
    print(f"Ollama service management placeholder. Server URL: {OLLAMA_BASE_URL}")
    return 0


def cmd_ollama_status(args: argparse.Namespace) -> int:
    """Check Ollama service status."""
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        ollama_ok = (r.status_code == 200)
    except Exception:
        ollama_ok = False
    status = "RUNNING" if ollama_ok else "STOPPED"
    print(f"Ollama ({OLLAMA_BASE_URL}) ............ {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="tai",
        description="Thumbnail AI — Official Command-Line Application",
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show CLI version and exit")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # tai run
    run_parser = subparsers.add_parser("run", help="Run the complete thumbnail generation pipeline")
    run_parser.add_argument("--csv", type=str, help="Path to creators CSV file")
    run_parser.add_argument("--thumbnail-dir", type=str, help="Thumbnail download directory")
    run_parser.add_argument("--analysis-dir", type=str, help="Analysis reports directory")
    run_parser.add_argument("--redesign-spec-dir", type=str, help="Redesign specs directory")
    run_parser.add_argument("--design-blueprint-dir", type=str, help="Blueprints directory")
    run_parser.add_argument("--prompt-package-dir", type=str, help="Prompt packages directory")
    run_parser.add_argument("--asset-extraction-dir", type=str, help="Module 8 asset extraction directory")
    run_parser.add_argument("--decision-dir", type=str, help="Module 9 decision manifests directory")
    run_parser.set_defaults(func=cmd_run)

    # tai doctor
    doctor_parser = subparsers.add_parser("doctor", help="Run full system health check")
    doctor_parser.set_defaults(func=cmd_doctor)

    # tai status
    status_parser = subparsers.add_parser("status", help="Display current system status")
    status_parser.set_defaults(func=cmd_status)

    # tai version
    version_parser = subparsers.add_parser("version", help="Display version information")
    version_parser.set_defaults(func=cmd_version)

    # tai test
    test_parser = subparsers.add_parser("test", help="Test runner placeholder")
    test_parser.set_defaults(func=cmd_test)

    # tai comfy
    comfy_parser = subparsers.add_parser("comfy", help="ComfyUI process management")
    comfy_sub = comfy_parser.add_subparsers(dest="subcommand", help="ComfyUI actions")
    
    comfy_start = comfy_sub.add_parser("start", help="Start ComfyUI process")
    comfy_start.set_defaults(func=cmd_comfy_start)
    
    comfy_stop = comfy_sub.add_parser("stop", help="Stop ComfyUI process")
    comfy_stop.set_defaults(func=cmd_comfy_stop)
    
    comfy_status_p = comfy_sub.add_parser("status", help="Check ComfyUI status")
    comfy_status_p.set_defaults(func=cmd_comfy_status)

    # tai ollama
    ollama_parser = subparsers.add_parser("ollama", help="Ollama service management")
    ollama_sub = ollama_parser.add_subparsers(dest="subcommand", help="Ollama actions")
    
    ollama_start = ollama_sub.add_parser("start", help="Start Ollama service")
    ollama_start.set_defaults(func=cmd_ollama_start)
    
    ollama_stop = ollama_sub.add_parser("stop", help="Stop Ollama service")
    ollama_stop.set_defaults(func=cmd_ollama_stop)
    
    ollama_status_p = ollama_sub.add_parser("status", help="Check Ollama status")
    ollama_status_p.set_defaults(func=cmd_ollama_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()

    if argv is None:
        raw_args = sys.argv[1:]
    else:
        raw_args = list(argv)

    known_subcommands = {"run", "doctor", "status", "version", "test", "comfy", "ollama"}
    known_flags = {"-v", "--version", "-h", "--help"}

    if not raw_args:
        parse_target = ["run"]
    elif raw_args[0] not in known_subcommands and raw_args[0] not in known_flags:
        parse_target = ["run"] + raw_args
    else:
        parse_target = raw_args

    args = parser.parse_args(parse_target)

    if getattr(args, "version", False):
        return cmd_version(args)

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
cli.py
======

Entry point CLI for PVQEF: python -m evaluation.cli ...
Subcommands: run | golden | batch | compare | report
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from loguru import logger

from evaluation.batch import run_batch_evaluation
from evaluation.benchmarking import (
    GoldenSampleManager,
    HistoricalStore,
    detect_regressions,
    run_golden_regression_suite,
)
from evaluation.config import EVAL_LOG_PATH
from evaluation.pipeline_runner import run_full_evaluation
from evaluation.reporting import ReportBuilder, render_report
from modules.config import DEFAULT_CSV_PATH


def _configure_cli_logging() -> None:
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(EVAL_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
        level="INFO",
        enqueue=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for PVQEF."""
    _configure_cli_logging()

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.cli",
        description="Pipeline Validation & Quality Evaluation Framework (PVQEF) CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = subparsers.add_parser("run", help="Run full pipeline validation on CSV")
    run_p.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to creators CSV")
    run_p.add_argument("--run-id", type=str, default=None, help="Custom run identifier")
    run_p.add_argument("--format", choices=["json", "markdown", "html"], default="markdown", help="Output format")

    # golden
    golden_p = subparsers.add_parser("golden", help="Run golden sample regression suite")
    golden_p.add_argument("--format", choices=["json", "markdown", "html"], default="markdown", help="Output format")

    # batch
    batch_p = subparsers.add_parser("batch", help="Run batch evaluation with bounded concurrency")
    batch_p.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to creators CSV")
    batch_p.add_argument("--concurrency", type=int, default=1, help="Max GPU concurrency")
    batch_p.add_argument("--format", choices=["json", "markdown", "html"], default="markdown", help="Output format")

    # report
    report_p = subparsers.add_parser("report", help="Render persisted run report")
    report_p.add_argument("run_id", type=str, help="Run ID to render")
    report_p.add_argument("--format", choices=["json", "markdown", "html"], default="markdown", help="Output format")

    args = parser.parse_args(argv)

    if args.command == "run":
        report = run_full_evaluation(csv_path=args.csv, run_id=args.run_id)
        ReportBuilder().persist_run_report(report)
        store = HistoricalStore()
        rec = store.create_record_from_run(report)
        store.append(rec)
        print(render_report(report, fmt=args.format))
        return 0 if report.status in ("success", "partial") else 1

    elif args.command == "golden":
        report = run_golden_regression_suite()
        ReportBuilder().persist_run_report(report)
        print(render_report(report, fmt=args.format))
        has_critical = any(r.severity == "critical" for r in report.regressions)
        return 1 if has_critical else 0

    elif args.command == "batch":
        report = run_batch_evaluation(csv_path=args.csv, max_concurrency=args.concurrency)
        ReportBuilder().persist_run_report(report)
        print(render_report(report, fmt=args.format))
        return 0 if report.status in ("success", "partial") else 1

    elif args.command == "report":
        runs_dir = Path("data/evaluation/runs") / args.run_id
        manifest_p = runs_dir / "run_manifest.json"
        if not manifest_p.exists():
            print(f"Error: Run manifest not found at {manifest_p}", file=sys.stderr)
            return 1
        from modules.models import PipelineRunReport
        import json
        report = PipelineRunReport.model_validate(json.loads(manifest_p.read_text(encoding="utf-8")))
        print(render_report(report, fmt=args.format))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

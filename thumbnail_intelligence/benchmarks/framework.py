"""
framework.py
============

BenchmarkFramework Implementation for Phase 6.1 Benchmark Framework.
High-level facade orchestrating benchmark sessions, leaderboards, failure analysis, and multi-format reporting.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from thumbnail_intelligence.benchmarks.dataset_loader import DatasetLoader
from thumbnail_intelligence.benchmarks.leaderboard import LeaderboardBuilder
from thumbnail_intelligence.benchmarks.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSession,
    DatasetItem,
    LeaderboardEntry,
)
from thumbnail_intelligence.benchmarks.runner import BenchmarkRunner
from thumbnail_intelligence.optimization.models import StoppingPolicy

logger = logging.getLogger(__name__)


class BenchmarkFrameworkError(RuntimeError):
    """Exception raised for benchmark framework errors."""
    pass


class BenchmarkFramework:
    """Master benchmark and evaluation framework facade."""

    def __init__(self, runner: Optional[BenchmarkRunner] = None) -> None:
        self.runner = runner or BenchmarkRunner()

    def run_benchmark(
        self,
        items: List[DatasetItem],
        dataset_name: str = "benchmark_dataset",
        stopping_policy: Optional[StoppingPolicy] = None,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkSession:
        """Run end-to-end benchmark session, generate leaderboard, and export multi-format reports.

        Args:
            items: List of DatasetItem samples to evaluate.
            dataset_name: Name of benchmark dataset.
            stopping_policy: Optional StoppingPolicy thresholds.
            output_directory: Directory to store benchmark session reports and artifacts.
            context_overrides: Additional runtime execution metadata overrides.

        Returns:
            BenchmarkSession containing complete results, leaderboard, and report links.
        """
        if not items:
            raise BenchmarkFrameworkError("Cannot run benchmark framework with empty items list.")

        out_dir = Path(output_directory) if output_directory else Path(tempfile.mkdtemp(prefix=f"bench_fw_{dataset_name}_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== Starting BenchmarkFramework Run for dataset '{dataset_name}' ({len(items)} items) ===")

        # 1. Execute Benchmark Session via BenchmarkRunner
        session = self.runner.run_benchmark(
            items=items,
            dataset_name=dataset_name,
            stopping_policy=stopping_policy,
            output_directory=out_dir,
            context_overrides=context_overrides,
        )

        # 2. Build Leaderboard Entry for current pipeline
        entry = LeaderboardEntry(
            rank=1,
            model_or_pipeline="Thumbnail AI v2.0 (Phase 5.6 Pipeline)",
            avg_quality_score=session.summary.avg_final_score,
            avg_ctr_score=session.summary.avg_ctr_prediction,
            success_rate_pct=session.summary.success_rate_pct,
            avg_runtime_s=session.summary.avg_runtime_s,
            peak_vram_gb=session.summary.peak_vram_gb,
        )

        leaderboard = LeaderboardBuilder.build_leaderboard([entry], leaderboard_id=f"lb_{session.session_id}")

        # 3. Export Reports in Multi-Formats (HTML, Markdown, JSON, CSV)
        json_path = str(out_dir / "benchmark_report.json")
        md_path = str(out_dir / "benchmark_report.md")
        html_path = str(out_dir / "benchmark_report.html")
        csv_path = str(out_dir / "benchmark_report.csv")

        # JSON Export
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(session.to_json())

        # Markdown Export
        md_content = self._generate_markdown_report(session, leaderboard)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # HTML Export
        html_content = self._generate_html_report(session, leaderboard)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # CSV Export
        self._generate_csv_report(session, csv_path)

        report = BenchmarkReport(
            session_id=session.session_id,
            dataset_name=dataset_name,
            summary=session.summary,
            leaderboard=leaderboard,
            html_report_path=html_path,
            markdown_report_path=md_path,
            json_report_path=json_path,
            csv_report_path=csv_path,
        )

        session = session.model_copy(update={"report": report})
        logger.info(f"=== Completed BenchmarkFramework Run for dataset '{dataset_name}' (Reports exported to '{out_dir}') ===")
        return session

    def _generate_markdown_report(self, session: BenchmarkSession, leaderboard: Any) -> str:
        """Generate Markdown format benchmark report."""
        s = session.summary
        lines = [
            f"# Benchmark Evaluation Report: {session.dataset_name}",
            f"**Session ID:** `{session.session_id}` | **Date:** {session.created_at}",
            "",
            "## Executive Summary",
            f"- **Total Samples Evaluated:** {s.total_samples}",
            f"- **Success Rate:** {s.success_rate_pct:.1f}% ({s.successful_samples} Passed, {s.failed_samples} Failed)",
            f"- **Average Quality Score:** {s.avg_final_score:.2f} / 100 (Initial: {s.avg_initial_score:.2f})",
            f"- **Average Score Improvement:** +{s.avg_score_improvement_pts:.2f} pts",
            f"- **Average Estimated CTR Score:** {s.avg_ctr_prediction:.2f} / 100",
            f"- **Average Runtime per Sample:** {s.avg_runtime_s:.2f} s",
            f"- **Peak GPU VRAM Recorded:** {s.peak_vram_gb:.2f} GB",
            "",
            "## System Leaderboard",
            leaderboard.to_markdown(),
            "",
            "## Failure Analysis Breakdown",
            "| Failure Category | Count |",
            "| --- | --- |",
        ]
        for cat, cnt in s.failure_distribution.items():
            lines.append(f"| {cat} | {cnt} |")

        lines.extend([
            "",
            "## Sample Results Breakdown",
            "| Item ID | Success | Initial Score | Final Score | Gain | CTR Score | Runtime (s) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for r in session.results:
            status_str = "PASS" if r.success else "FAIL"
            lines.append(f"| `{r.item_id}` | {status_str} | {r.initial_score:.1f} | {r.final_score:.1f} | +{r.score_gain_pts:.1f} | {r.ctr_prediction:.1f} | {r.runtime_s:.2f}s |")

        return "\n".join(lines)

    def _generate_html_report(self, session: BenchmarkSession, leaderboard: Any) -> str:
        """Generate HTML format benchmark report."""
        s = session.summary
        rows = []
        for r in session.results:
            st = "PASS" if r.success else "FAIL"
            cls = "pass" if r.success else "fail"
            rows.append(
                f"<tr class='{cls}'><td><code>{r.item_id}</code></td><td>{st}</td>"
                f"<td>{r.initial_score:.1f}</td><td>{r.final_score:.1f}</td>"
                f"<td>+{r.score_gain_pts:.1f}</td><td>{r.ctr_prediction:.1f}</td>"
                f"<td>{r.runtime_s:.2f}s</td></tr>"
            )

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {session.dataset_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #0f172a; color: #f8fafc; }}
        h1, h2 {{ color: #38bdf8; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid #38bdf8; }}
        .card .val {{ font-size: 24px; font-weight: bold; color: #f8fafc; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0284c7; color: white; }}
        tr.pass {{ color: #4ade80; }}
        tr.fail {{ color: #f87171; }}
    </style>
</head>
<body>
    <h1>Thumbnail AI Benchmark Report: {session.dataset_name}</h1>
    <p>Session ID: <code>{session.session_id}</code></p>
    
    <div class="metrics-grid">
        <div class="card"><div>Success Rate</div><div class="val">{s.success_rate_pct:.1f}%</div></div>
        <div class="card"><div>Avg Quality Score</div><div class="val">{s.avg_final_score:.1f} / 100</div></div>
        <div class="card"><div>Avg CTR Score</div><div class="val">{s.avg_ctr_prediction:.1f} / 100</div></div>
        <div class="card"><div>Peak VRAM</div><div class="val">{s.peak_vram_gb:.2f} GB</div></div>
    </div>

    <h2>Detailed Results</h2>
    <table>
        <thead>
            <tr><th>Item ID</th><th>Status</th><th>Initial Score</th><th>Final Score</th><th>Gain</th><th>CTR Score</th><th>Runtime</th></tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>"""

    def _generate_csv_report(self, session: BenchmarkSession, csv_path: str) -> None:
        """Generate CSV format benchmark report."""
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Item ID", "Success", "Failure Category", "Initial Score", "Final Score", "Score Gain", "CTR Score", "Iterations", "Runtime (s)", "Peak VRAM (GB)", "Render Cost"])
            for r in session.results:
                writer.writerow([r.item_id, r.success, r.failure_category.value, r.initial_score, r.final_score, r.score_gain_pts, r.ctr_prediction, r.iterations_required, r.runtime_s, r.peak_vram_gb, r.estimated_render_cost])

"""
leaderboard.py
==============

Leaderboard Generator Implementation for Phase 6.1 Benchmark Framework.
Ranks pipeline versions and exports leaderboards in Markdown, CSV, JSON, and HTML formats.
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import List, Union

from thumbnail_intelligence.benchmarks.models import Leaderboard, LeaderboardEntry


class LeaderboardBuilder:
    """Builds and exports system leaderboards."""

    @staticmethod
    def build_leaderboard(entries: List[LeaderboardEntry], leaderboard_id: Optional[str] = None) -> Leaderboard:
        """Rank entries descending by avg_quality_score and return a Leaderboard."""
        lid = leaderboard_id or f"lb_{uuid.uuid4().hex[:8]}"
        sorted_entries = sorted(entries, key=lambda e: (e.avg_quality_score, e.avg_ctr_score), reverse=True)

        ranked_entries: List[LeaderboardEntry] = []
        for rank_idx, entry in enumerate(sorted_entries, start=1):
            ranked_entries.append(entry.model_copy(update={"rank": rank_idx}))

        return Leaderboard(
            leaderboard_id=lid,
            schema_version="1.0.0",
            entries=ranked_entries,
        )

    @staticmethod
    def export_csv(leaderboard: Leaderboard, file_path: Union[str, Path]) -> str:
        """Export leaderboard to a CSV file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Model/Pipeline", "Avg Quality Score", "Avg CTR Score", "Success Rate (%)", "Avg Runtime (s)", "Peak VRAM (GB)"])
            for e in leaderboard.entries:
                writer.writerow([e.rank, e.model_or_pipeline, e.avg_quality_score, e.avg_ctr_score, e.success_rate_pct, e.avg_runtime_s, e.peak_vram_gb])

        return str(path)

    @staticmethod
    def export_html(leaderboard: Leaderboard, file_path: Union[str, Path]) -> str:
        """Export leaderboard to an HTML file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_html = []
        for e in leaderboard.entries:
            rows_html.append(
                f"<tr><td>{e.rank}</td><td><strong>{e.model_or_pipeline}</strong></td>"
                f"<td>{e.avg_quality_score:.2f}</td><td>{e.avg_ctr_score:.2f}</td>"
                f"<td>{e.success_rate_pct:.1f}%</td><td>{e.avg_runtime_s:.2f}s</td>"
                f"<td>{e.peak_vram_gb:.2f} GB</td></tr>"
            )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Thumbnail AI Leaderboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #0f172a; color: #f8fafc; }}
        h1 {{ color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0284c7; color: white; }}
        tr:hover {{ background: #334155; }}
    </style>
</head>
<body>
    <h1>Thumbnail AI Benchmark Leaderboard</h1>
    <table>
        <thead>
            <tr><th>Rank</th><th>Model / Pipeline</th><th>Avg Quality Score</th><th>Avg CTR Score</th><th>Success Rate</th><th>Avg Runtime</th><th>Peak VRAM</th></tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(path)

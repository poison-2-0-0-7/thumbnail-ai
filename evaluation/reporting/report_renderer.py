"""
report_renderer.py
==================

Renders PipelineRunReport into JSON, Markdown, or HTML views.
"""

from __future__ import annotations

import json
from typing import Literal

from modules.models import PipelineRunReport
from .interfaces import IReportRenderer


class JSONReportRenderer(IReportRenderer):
    """Canonical JSON renderer."""

    def render(self, report: PipelineRunReport) -> str:
        return report.model_dump_json(indent=2)


class MarkdownReportRenderer(IReportRenderer):
    """Human-readable Markdown renderer with summary tables."""

    def render(self, report: PipelineRunReport) -> str:
        lines = [
            f"# Pipeline Evaluation Report — {report.run_id}",
            "",
            f"- **Status**: `{report.status}`",
            f"- **Creators Processed**: {report.succeeded}/{report.total_creators} succeeded ({report.skipped} skipped)",
            f"- **Started**: {report.started_at}",
            f"- **Completed**: {report.completed_at}",
            f"- **Total Duration**: {report.total_duration_seconds:.2f}s",
            "",
            "## Module Validation Summary",
            "",
            "| Video ID | Module | Schema Valid | Invariants Failed | Status |",
            "|---|---|---|---|---|",
        ]

        for vid, results in report.module_results.items():
            for r in results:
                failed_str = ", ".join(r.invariants_failed) if r.invariants_failed else "None"
                lines.append(f"| `{vid}` | `{r.module_name}` | `{r.schema_valid}` | {failed_str} | `{r.status}` |")

        if report.quality_reports:
            lines.extend([
                "",
                "## Quality Evaluation Summary",
                "",
                "| Video ID | Weighted Score | Hard Gate Passed | Status |",
                "|---|---|---|---|",
            ])
            for vid, q_rep in report.quality_reports.items():
                lines.append(f"| `{vid}` | {q_rep.weighted_overall_score:.2f} | `{q_rep.hard_gate_passed}` | `{q_rep.status}` |")

        if report.regressions:
            lines.extend([
                "",
                "## Detected Regressions",
                "",
                "| Severity | Rule | Target | Delta | Message |",
                "|---|---|---|---|---|",
            ])
            for reg in report.regressions:
                lines.append(f"| `{reg.severity}` | `{reg.rule_name}` | `{reg.dimension_or_stage or 'overall'}` | {reg.delta:+.3f} | {reg.message} |")

        return "\n".join(lines)


class HTMLReportRenderer(IReportRenderer):
    """Standalone HTML report renderer with CSS styling."""

    def render(self, report: PipelineRunReport) -> str:
        md_content = MarkdownReportRenderer().render(report)
        html_body = md_content.replace("\n", "<br/>\n")
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pipeline Run Report {report.run_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; background: #121212; color: #e0e0e0; }}
        h1, h2 {{ color: #ffffff; border-bottom: 1px solid #333; padding-bottom: 0.3rem; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #1f1f1f; }}
        code {{ background: #262626; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
    </style>
</head>
<body>
    <pre>{md_content}</pre>
</body>
</html>"""


def render_report(
    report: PipelineRunReport,
    fmt: Literal["json", "markdown", "html"] = "markdown",
) -> str:
    """Public helper to render a report into requested format string."""
    if fmt == "json":
        return JSONReportRenderer().render(report)
    elif fmt == "html":
        return HTMLReportRenderer().render(report)
    return MarkdownReportRenderer().render(report)

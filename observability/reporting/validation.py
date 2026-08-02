"""
observability/reporting/validation.py
======================================

Schema and structural validation for RootCauseReport objects in PORCE.
"""

from __future__ import annotations

from typing import Any
from pydantic import ValidationError

from observability.reporting.models import RootCauseReport


class RootCauseValidation:
    """
    Validator for RootCauseReport data objects and persistence integrity.
    """

    @staticmethod
    def validate_report(data: Any) -> bool:
        """
        Validate dictionary or object against RootCauseReport schema.
        Returns True if valid, False otherwise.
        """
        if isinstance(data, RootCauseReport):
            return True
        if isinstance(data, dict):
            try:
                RootCauseReport.model_validate(data)
                return True
            except ValidationError:
                return False
        return False

    @staticmethod
    def validate_report_integrity(report: RootCauseReport) -> bool:
        """
        Verify structural integrity of a RootCauseReport.
        Ensures video_id is set, severity counts match findings, and trace hash is present.
        """
        if not report.video_id:
            return False

        if not report.generated_from_trace_hash:
            return False

        # Validate count totals match findings list
        total_counts = report.fail_count + report.warning_count + report.info_count + report.pass_count
        if total_counts != len(report.findings):
            return False

        return True

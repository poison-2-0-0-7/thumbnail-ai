"""
context.py
==========

RenderJobContext implementation for Phase 4.1 Execution Engine.
Wraps RenderExecutionPackage and job-scoped execution metadata, timing,
attempt counter, correlation ID, and thread-safe cancellation support.
"""

from __future__ import annotations

import time
import threading
import uuid
from typing import Any, Dict, Optional

from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage
from renderer_v2.execution.exceptions import JobCancellationError


class RenderJobContext:
    """
    Read-only job execution context created once per job attempt.
    Wraps validated RenderExecutionPackage + runtime execution metadata.
    Supports cancellation tokens and precise execution timing.
    """

    def __init__(
        self,
        package: RenderExecutionPackage,
        job_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        attempt: int = 1,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._package = package
        self._job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        self._correlation_id = correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
        self._attempt = max(1, attempt)
        self._execution_metadata: Dict[str, Any] = execution_metadata or {}

        # Timing tracking
        self._start_time: float = time.time()
        self._end_time: Optional[float] = None

        # Cancellation support
        self._cancellation_event = threading.Event()
        self._cancel_reason: Optional[str] = None

    @property
    def package(self) -> RenderExecutionPackage:
        """Return the read-only RenderExecutionPackage."""
        return self._package

    @property
    def job_id(self) -> str:
        """Return unique job identifier."""
        return self._job_id

    @property
    def correlation_id(self) -> str:
        """Return correlation identifier for tracing."""
        return self._correlation_id

    @property
    def attempt(self) -> int:
        """Return attempt count."""
        return self._attempt

    @property
    def execution_metadata(self) -> Dict[str, Any]:
        """Return copy of execution metadata dictionary."""
        return dict(self._execution_metadata)

    @property
    def start_time(self) -> float:
        """Return job start timestamp in seconds."""
        return self._start_time

    @property
    def end_time(self) -> Optional[float]:
        """Return job end timestamp in seconds, or None if running."""
        return self._end_time

    @property
    def elapsed_time(self) -> float:
        """Return elapsed execution duration in seconds."""
        end = self._end_time if self._end_time is not None else time.time()
        return max(0.0, end - self._start_time)

    def mark_completed(self) -> float:
        """Mark job execution complete and return total elapsed time."""
        self._end_time = time.time()
        return self.elapsed_time

    def cancel(self, reason: str = "Cancellation requested by caller") -> None:
        """Trigger thread-safe job cancellation."""
        self._cancel_reason = reason
        self._cancellation_event.set()

    def is_cancelled(self) -> bool:
        """Check if job has been cancelled."""
        return self._cancellation_event.is_set()

    def check_cancellation(self) -> None:
        """Check cancellation state and raise JobCancellationError if cancelled."""
        if self.is_cancelled():
            raise JobCancellationError(
                message=self._cancel_reason or "Job execution was cancelled",
                job_id=self._job_id,
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata entry by key."""
        return self._execution_metadata.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        """Set or update execution metadata key."""
        self._execution_metadata[key] = value

    def create_next_attempt_context(self) -> RenderJobContext:
        """Create a new context instance for the next critique-loop retry attempt."""
        return RenderJobContext(
            package=self._package,
            job_id=self._job_id,
            correlation_id=self._correlation_id,
            attempt=self._attempt + 1,
            execution_metadata=self._execution_metadata,
        )

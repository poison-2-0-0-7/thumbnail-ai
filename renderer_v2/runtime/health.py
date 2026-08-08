"""
health.py
=========

HealthMonitor for model readiness, checkpoint validation, and warmup checks in Phase 4.4.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from renderer_v2.runtime.models import BaseModelAdapter, ModelDescriptor, ModelState

logger = logging.getLogger(__name__)


class HealthCheckResult:
    """Container for model health check results."""

    def __init__(self, model_name: str, is_healthy: bool, checks: Dict[str, bool], notes: List[str]) -> None:
        self.model_name = model_name
        self.is_healthy = is_healthy
        self.checks = checks
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_healthy": self.is_healthy,
            "checks": self.checks,
            "notes": self.notes,
        }


class HealthMonitor:
    """Performs integrity, checkpoint, device, and warmup health checks on models."""

    def validate_checkpoint(self, descriptor: ModelDescriptor) -> Tuple[bool, str]:
        """Validate existence and integrity of model checkpoint file on disk."""
        if descriptor.checkpoint_path is None:
            return True, "No checkpoint path required."
        
        path = descriptor.checkpoint_path
        if not os.path.exists(path):
            return False, f"Checkpoint file '{path}' not found."
        
        if os.path.getsize(path) == 0:
            return False, f"Checkpoint file '{path}' is empty (0 bytes)."
        
        return True, f"Checkpoint file '{path}' validated."

    def check_adapter_health(self, adapter: BaseModelAdapter) -> HealthCheckResult:
        """Run health check, device check, checkpoint check, and warmup on a model adapter."""
        model_name = adapter.model_name
        checks: Dict[str, bool] = {}
        notes: List[str] = []

        # 1. Checkpoint Check
        ckpt_ok, ckpt_note = self.validate_checkpoint(adapter.descriptor)
        checks["checkpoint_exists"] = ckpt_ok
        notes.append(ckpt_note)

        # 2. State Check
        state_ok = adapter.state not in {ModelState.FAILED, ModelState.UNREGISTERED}
        checks["state_valid"] = state_ok
        notes.append(f"Model state is {adapter.state.value}")

        # 3. Model Internal Health Check
        try:
            internal_ok = adapter.health_check()
            checks["adapter_health"] = internal_ok
            notes.append(f"Adapter health_check() returned {internal_ok}")
        except Exception as e:
            checks["adapter_health"] = False
            notes.append(f"Adapter health_check() raised exception: {e}")

        # 4. Warmup Check
        if all(checks.values()):
            try:
                warmup_ok = adapter.warmup()
                checks["warmup_passed"] = warmup_ok
                notes.append(f"Warmup check returned {warmup_ok}")
            except Exception as e:
                checks["warmup_passed"] = False
                notes.append(f"Warmup check raised exception: {e}")
        else:
            checks["warmup_passed"] = False

        overall_healthy = all(checks.values())
        return HealthCheckResult(
            model_name=model_name,
            is_healthy=overall_healthy,
            checks=checks,
            notes=notes,
        )

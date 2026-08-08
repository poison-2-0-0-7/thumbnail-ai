"""
fsm.py
======

Execution State Machine (FSM) and Execution Lifecycle for Phase 4.1.
Enforces valid state transitions across job initialization, validation, DAG scheduling,
stage dispatching, stage validation, checkpointing, and completion/failure states.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from renderer_v2.execution.exceptions import ExecutionEngineError, InvalidStateTransitionError

logger = logging.getLogger(__name__)


class ExecutionState(str, Enum):
    """Execution lifecycle state enum."""

    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    VALIDATING = "VALIDATING"
    SCHEDULING = "SCHEDULING"
    DISPATCHING = "DISPATCHING"
    RUNNING_STAGE = "RUNNING_STAGE"
    VALIDATING_STAGE = "VALIDATING_STAGE"
    CHECKPOINTING = "CHECKPOINTING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_DEGRADATION = "COMPLETED_WITH_DEGRADATION"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FSMEvent(BaseKBModel):
    """Event log record for state machine transitions."""

    from_state: ExecutionState = Field(description="State before transition")
    to_state: ExecutionState = Field(description="State after transition")
    timestamp: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Event metadata details")





class ExecutionFSM:
    """
    Execution State Machine enforcing legal execution state transitions.
    Tracks complete state transition history and enforces terminal states.
    """

    # Allowed transitions map: key = current state, value = set of legal target states
    ALLOWED_TRANSITIONS: Dict[ExecutionState, Set[ExecutionState]] = {
        ExecutionState.UNINITIALIZED: {
            ExecutionState.INITIALIZING,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.INITIALIZING: {
            ExecutionState.VALIDATING,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.VALIDATING: {
            ExecutionState.SCHEDULING,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.SCHEDULING: {
            ExecutionState.DISPATCHING,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.DISPATCHING: {
            ExecutionState.RUNNING_STAGE,
            ExecutionState.COMPLETED,
            ExecutionState.COMPLETED_WITH_DEGRADATION,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.RUNNING_STAGE: {
            ExecutionState.VALIDATING_STAGE,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.VALIDATING_STAGE: {
            ExecutionState.CHECKPOINTING,
            ExecutionState.DISPATCHING,
            ExecutionState.COMPLETED,
            ExecutionState.COMPLETED_WITH_DEGRADATION,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.CHECKPOINTING: {
            ExecutionState.DISPATCHING,
            ExecutionState.COMPLETED,
            ExecutionState.COMPLETED_WITH_DEGRADATION,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        # Terminal states
        ExecutionState.COMPLETED: set(),
        ExecutionState.COMPLETED_WITH_DEGRADATION: set(),
        ExecutionState.FAILED: set(),
        ExecutionState.CANCELLED: set(),
    }

    TERMINAL_STATES: Set[ExecutionState] = {
        ExecutionState.COMPLETED,
        ExecutionState.COMPLETED_WITH_DEGRADATION,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    }

    def __init__(self, initial_state: ExecutionState = ExecutionState.UNINITIALIZED) -> None:
        self._current_state = initial_state
        self._event_history: List[FSMEvent] = []

    @property
    def current_state(self) -> ExecutionState:
        """Return current FSM state."""
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self._current_state in self.TERMINAL_STATES

    @property
    def event_history(self) -> List[FSMEvent]:
        """Return shallow copy of transition event history."""
        return list(self._event_history)

    def can_transition_to(self, target_state: ExecutionState) -> bool:
        """Check if transitioning to target_state is allowed from current state."""
        allowed = self.ALLOWED_TRANSITIONS.get(self._current_state, set())
        return target_state in allowed

    def transition_to(self, target_state: ExecutionState, details: Optional[Dict[str, Any]] = None) -> ExecutionState:
        """
        Transition FSM to target_state if legal.
        Raises InvalidStateTransitionError on illegal transitions.
        """
        if self._current_state == target_state:
            return self._current_state  # No-op re-entry

        if not self.can_transition_to(target_state):
            raise InvalidStateTransitionError(
                from_state=self._current_state,
                to_state=target_state,
                reason=f"Current state '{self._current_state.value}' cannot transition to '{target_state.value}'.",
            )

        event = FSMEvent(
            from_state=self._current_state,
            to_state=target_state,
            details=details,
        )
        self._event_history.append(event)
        old_state = self._current_state
        self._current_state = target_state

        logger.debug(f"FSM Transition: {old_state.value} -> {target_state.value}")
        return self._current_state

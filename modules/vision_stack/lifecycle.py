"""Lifecycle constants for AI Vision Stack V2.1."""

from __future__ import annotations

from .models import VisionModelLifecycleState


ALLOWED_LIFECYCLE_TRANSITIONS: dict[
    VisionModelLifecycleState, frozenset[VisionModelLifecycleState]
] = {
    VisionModelLifecycleState.REGISTERED: frozenset(
        {VisionModelLifecycleState.CPU_CACHED, VisionModelLifecycleState.EVICTED}
    ),
    VisionModelLifecycleState.CPU_CACHED: frozenset(
        {VisionModelLifecycleState.GPU_ACTIVE, VisionModelLifecycleState.EVICTED}
    ),
    VisionModelLifecycleState.GPU_ACTIVE: frozenset(
        {VisionModelLifecycleState.CPU_CACHED, VisionModelLifecycleState.EVICTED}
    ),
    VisionModelLifecycleState.EVICTED: frozenset({VisionModelLifecycleState.REGISTERED}),
}

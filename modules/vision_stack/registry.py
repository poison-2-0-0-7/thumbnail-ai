"""Model registry and lifecycle tracking for AI Vision Stack V2.1."""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from .exceptions import VisionStackRegistryError
from .lifecycle import ALLOWED_LIFECYCLE_TRANSITIONS
from .models import (
    RegisteredVisionModel,
    VisionModelConfig,
    VisionModelLifecycleState,
    VisionStackConfig,
)


class ModelRegistry:
    """Register V2.1 model configs and track lifecycle/runtime state."""

    def __init__(self) -> None:
        self._models: dict[str, RegisteredVisionModel] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        config: VisionModelConfig,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredVisionModel:
        """Register one configured model in the initial lifecycle state."""
        with self._lock:
            entry = RegisteredVisionModel(
                name=name,
                config=config,
                lifecycle_state=VisionModelLifecycleState.REGISTERED,
                metadata=dict(metadata or {}),
                runtime_state={},
            )
            self._models[entry.name] = entry
            logger.debug("Registered vision model name={name}", name=entry.name)
            return entry

    def register_stack(self, stack_config: VisionStackConfig) -> tuple[RegisteredVisionModel, ...]:
        """Register every model declared by a validated stack configuration."""
        registered: list[RegisteredVisionModel] = []
        for name, model_config in stack_config.model_items():
            registered.append(self.register(name, model_config))
        logger.info("Registered {count} V2.1 vision models", count=len(registered))
        return tuple(registered)

    def get(self, name: str) -> RegisteredVisionModel:
        """Return one registered model by name."""
        with self._lock:
            try:
                return self._models[name]
            except KeyError as exc:
                raise VisionStackRegistryError(f"Vision model is not registered: {name}") from exc

    def all_models(self) -> tuple[RegisteredVisionModel, ...]:
        """Return all registered models in insertion order."""
        with self._lock:
            return tuple(self._models.values())

    def transition(
        self,
        name: str,
        next_state: VisionModelLifecycleState,
    ) -> RegisteredVisionModel:
        """Move a registered model through an allowed lifecycle transition."""
        with self._lock:
            current = self.get(name)
            allowed = ALLOWED_LIFECYCLE_TRANSITIONS[current.lifecycle_state]
            if next_state not in allowed:
                raise VisionStackRegistryError(
                    "Invalid lifecycle transition for "
                    f"{name}: {current.lifecycle_state.value} -> {next_state.value}"
                )
            updated = current.model_copy(update={"lifecycle_state": next_state})
            self._models[name] = updated
            logger.debug(
                "Vision model lifecycle transition name={name} state={state}",
                name=name,
                state=next_state.value,
            )
            return updated

    def update_metadata(self, name: str, metadata: dict[str, Any]) -> RegisteredVisionModel:
        """Merge metadata into a registered model entry."""
        with self._lock:
            current = self.get(name)
            updated = current.model_copy(update={"metadata": {**current.metadata, **metadata}})
            self._models[name] = updated
            return updated

    def update_runtime_state(self, name: str, runtime_state: dict[str, Any]) -> RegisteredVisionModel:
        """Merge runtime state into a registered model entry."""
        with self._lock:
            current = self.get(name)
            updated = current.model_copy(
                update={"runtime_state": {**current.runtime_state, **runtime_state}}
            )
            self._models[name] = updated
            return updated

    def reset(self) -> None:
        """Clear all registered model entries."""
        with self._lock:
            self._models.clear()
            logger.info("Cleared V2.1 vision model registry")

"""
evaluation/module_validators/interfaces.py
============================================

Interface for all PVQEF module validators.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from modules.models import ModuleValidationResult


class IModuleValidator(ABC):
    """Interface for stage-specific artifact validators."""

    @property
    @abstractmethod
    def module_name(self) -> str:
        """Module identifier, e.g. 'module1_csv_reader'."""

    @abstractmethod
    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        """Check schema conformance and module-specific invariants for a persisted artifact."""

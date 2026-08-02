"""
observability/facts/validation.py
==================================

Schema and structural validation for FactCollection and TraceFacts models in PORCE.
"""

from __future__ import annotations

from typing import Any
from pydantic import ValidationError

from observability.config import OBS_FACTS_VERSION
from observability.facts.models import FactCollection, TraceFacts


class FactValidation:
    """
    Validator for facts collection data and trace facts schema integrity.
    """

    @staticmethod
    def validate_collection_data(data: Any) -> bool:
        """
        Validate data dictionary or object against FactCollection schema.
        Returns True if valid, False otherwise.
        """
        if isinstance(data, FactCollection):
            return True
        if isinstance(data, dict):
            try:
                FactCollection.model_validate(data)
                return True
            except ValidationError:
                return False
        return False

    @staticmethod
    def validate_trace_facts_data(data: Any) -> bool:
        """
        Validate data dictionary or object against TraceFacts schema.
        Returns True if valid, False otherwise.
        """
        if isinstance(data, TraceFacts):
            return True
        if isinstance(data, dict):
            try:
                TraceFacts.model_validate(data)
                return True
            except ValidationError:
                return False
        return False

    @staticmethod
    def check_version_compatibility(fact_version: str) -> bool:
        """
        Check if fact_version is compatible with current system OBS_FACTS_VERSION.
        Major versions must match.
        """
        if not fact_version:
            return False
        current_major = OBS_FACTS_VERSION.split(".")[0]
        target_major = fact_version.split(".")[0]
        return current_major == target_major

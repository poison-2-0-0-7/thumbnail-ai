"""
observability/facts/serializer.py
==================================

Serializer and deserializer for FactCollection objects.
Handles JSON formatting, schema validation, and error handling.
"""

from __future__ import annotations

from pydantic import ValidationError

from observability.exceptions import FactValidationError
from observability.facts.interfaces import IFactSerializer
from observability.facts.models import FactCollection


class FactSerializer(IFactSerializer):
    """
    Handles JSON serialization and deserialization of FactCollection objects.
    """

    def serialize(self, collection: FactCollection) -> str:
        """
        Serialize FactCollection to formatted JSON string.
        """
        try:
            return collection.model_dump_json(indent=2)
        except Exception as exc:
            raise FactValidationError(f"Failed to serialize FactCollection: {exc}") from exc

    def deserialize(self, json_str: str) -> FactCollection:
        """
        Deserialize JSON string into FactCollection object.
        """
        try:
            return FactCollection.model_validate_json(json_str)
        except ValidationError as exc:
            raise FactValidationError(f"Invalid FactCollection schema: {exc}") from exc
        except Exception as exc:
            raise FactValidationError(f"Failed to parse FactCollection JSON: {exc}") from exc

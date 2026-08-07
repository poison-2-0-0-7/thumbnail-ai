"""
serialization.py
================

Production serialization and deserialization engine for the Thumbnail Intelligence Knowledge Base.
Provides lossless conversion between Pydantic models, JSON strings, and Python primitives with
specialized support for Enums, ISO timestamps, UUIDs, Path objects, and NumPy arrays.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel

from thumbnail_intelligence.knowledge_base.exceptions import (
    DeserializationError,
    SerializationError,
    TypeSerializationError,
)

T = TypeVar("T", bound=BaseModel)


class KnowledgeBaseJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder supporting Pydantic models, Enums, datetime, UUID, Path,
    and numpy arrays / scalars with clean deterministic formatting.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        # Handle numpy objects gracefully if numpy is available
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.complexfloating)):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass

        return super().default(obj)


class KBSerializer:
    """
    Central serialization service for the Knowledge Base.
    Guarantees consistent schema encoding, timestamp normalization, and error wrapping.
    """

    @staticmethod
    def to_dict(obj: Any) -> Dict[str, Any]:
        """
        Convert a model or dataclass into a clean JSON-serializable Python dictionary.
        """
        if obj is None:
            return {}
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, dict):
            # Recursively ensure inner objects are primitives
            try:
                raw_json = json.dumps(obj, cls=KnowledgeBaseJSONEncoder)
                return json.loads(raw_json)
            except Exception as e:
                raise SerializationError(
                    message=f"Failed to serialize dictionary payload: {e}",
                    context={"error": str(e)},
                ) from e
        try:
            raw_json = json.dumps(obj, cls=KnowledgeBaseJSONEncoder)
            return json.loads(raw_json)
        except Exception as e:
            raise TypeSerializationError(
                message=f"Unsupported object type '{type(obj)}' for dictionary serialization: {e}",
                context={"type": str(type(obj))},
            ) from e

    @staticmethod
    def from_dict(data: Dict[str, Any], target_cls: Type[T]) -> T:
        """
        Instantiate and validate a target Pydantic model class from a dictionary.
        """
        if not isinstance(data, dict):
            raise DeserializationError(
                message=f"Expected dictionary input for {target_cls.__name__}, got {type(data).__name__}",
                context={"target_class": target_cls.__name__, "received_type": type(data).__name__},
            )
        try:
            return target_cls.model_validate(data)
        except Exception as e:
            raise DeserializationError(
                message=f"Failed to validate {target_cls.__name__} from dictionary: {e}",
                context={"target_class": target_cls.__name__, "validation_error": str(e)},
            ) from e

    @staticmethod
    def serialize(obj: Any, indent: int = 2) -> str:
        """
        Serialize any supported object or model to a formatted JSON string.
        """
        try:
            if isinstance(obj, BaseModel):
                return obj.model_dump_json(indent=indent)
            return json.dumps(
                obj,
                cls=KnowledgeBaseJSONEncoder,
                indent=indent,
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception as e:
            raise SerializationError(
                message=f"Failed to serialize object to JSON: {e}",
                context={"object_type": str(type(obj)), "error": str(e)},
            ) from e

    @staticmethod
    def deserialize(json_str: Union[str, bytes], target_cls: Type[T]) -> T:
        """
        Deserialize and validate a JSON string or bytes into a typed Pydantic model.
        """
        if not isinstance(json_str, (str, bytes)):
            raise DeserializationError(
                message=f"Expected str or bytes for JSON deserialization, got {type(json_str).__name__}",
                context={"target_class": target_cls.__name__, "received_type": type(json_str).__name__},
            )
        try:
            if isinstance(json_str, bytes):
                json_str = json_str.decode("utf-8")
            return target_cls.model_validate_json(json_str)
        except Exception as e:
            raise DeserializationError(
                message=f"Failed to deserialize JSON into {target_cls.__name__}: {e}",
                context={"target_class": target_cls.__name__, "error": str(e)},
            ) from e

    @staticmethod
    def safe_dumps(obj: Any, indent: int = 2) -> str:
        """
        Safe JSON string serialization with fallback to str() on un-serializable objects.
        """
        try:
            return json.dumps(
                obj,
                cls=KnowledgeBaseJSONEncoder,
                indent=indent,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return json.dumps({"unserializable_object": str(obj)})

    @staticmethod
    def safe_loads(json_str: str) -> Any:
        """
        Safe JSON parsing into Python primitives.
        """
        try:
            return json.loads(json_str)
        except Exception as e:
            raise DeserializationError(
                message=f"Invalid JSON string: {e}",
                context={"raw_snippet": json_str[:100] if isinstance(json_str, str) else ""},
            ) from e

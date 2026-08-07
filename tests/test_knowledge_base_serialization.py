"""
Unit tests for Knowledge Base serialization and deserialization engine.
Tests JSON encoders, Pydantic model serialization, dictionary conversions,
NumPy array handling, ISO timestamp preservation, and error wrapping.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest

from thumbnail_intelligence.knowledge_base.exceptions import (
    DeserializationError,
    SerializationError,
)
from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    CompetitorProfile,
    CompetitorStatus,
    EvidenceReference,
    EvidenceSourceType,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.knowledge_base.serialization import (
    KBSerializer,
    KnowledgeBaseJSONEncoder,
)


def test_kb_serializer_model_roundtrip() -> None:
    entry = KnowledgeEntry(
        entry_id="entry_ser_001",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[0.25] * 512,
        embedding_model="OpenCLIP-ViT-B-32",
        source_video_id="vid_test_01",
        niche="gaming",
        facets={"resolution": "1080p", "tags": ["epic", "win"]},
    )

    # 1. Serialize to JSON
    json_str = KBSerializer.serialize(entry, indent=2)
    assert "entry_ser_001" in json_str
    assert "OpenCLIP-ViT-B-32" in json_str

    # 2. Deserialize from JSON string
    loaded_entry = KBSerializer.deserialize(json_str, KnowledgeEntry)
    assert loaded_entry.entry_id == entry.entry_id
    assert loaded_entry.embedding == entry.embedding
    assert loaded_entry.facets["resolution"] == "1080p"

    # 3. Deserialize from bytes
    bytes_data = json_str.encode("utf-8")
    loaded_from_bytes = KBSerializer.deserialize(bytes_data, KnowledgeEntry)
    assert loaded_from_bytes.entry_id == entry.entry_id

    # 4. to_dict and from_dict
    d = KBSerializer.to_dict(entry)
    assert isinstance(d, dict)
    assert d["entry_id"] == "entry_ser_001"
    reconstructed = KBSerializer.from_dict(d, KnowledgeEntry)
    assert reconstructed == entry


def test_kb_serializer_complex_types() -> None:
    data = {
        "uuid": uuid4(),
        "path": Path("/data/test/path.json"),
        "date": datetime.date(2026, 8, 8),
        "datetime": datetime.datetime(2026, 8, 8, 12, 0, 0, tzinfo=datetime.timezone.utc),
        "set_data": {3, 1, 2},
        "numpy_float": np.float32(3.1415),
        "numpy_int": np.int64(42),
        "numpy_array": np.array([1.0, 2.0, 3.0]),
        "status_enum": CompetitorStatus.ACTIVE,
    }

    serialized = KBSerializer.serialize(data)
    assert "2026-08-08" in serialized
    assert "3.14" in serialized
    assert "active" in serialized

    dict_out = KBSerializer.to_dict(data)
    assert dict_out["numpy_int"] == 42
    assert round(dict_out["numpy_float"], 3) == 3.141
    assert dict_out["set_data"] == [1, 2, 3]
    assert dict_out["status_enum"] == "active"


def test_kb_serializer_error_handling() -> None:
    # Deserializing invalid JSON
    with pytest.raises(DeserializationError):
        KBSerializer.deserialize("{ broken json", Archetype)

    # Deserializing non-matching types
    with pytest.raises(DeserializationError):
        KBSerializer.deserialize(12345, Archetype)  # type: ignore

    # from_dict with invalid input
    with pytest.raises(DeserializationError):
        KBSerializer.from_dict("not a dict", Archetype)  # type: ignore

    # from_dict with missing mandatory fields
    with pytest.raises(DeserializationError):
        KBSerializer.from_dict({"archetype_id": ""}, Archetype)


def test_safe_dumps_and_loads() -> None:
    data = {"hello": "world", "num": 100}
    json_str = KBSerializer.safe_dumps(data)
    loaded = KBSerializer.safe_loads(json_str)
    assert loaded["hello"] == "world"
    assert loaded["num"] == 100

    # safe_loads on corrupted json
    with pytest.raises(DeserializationError):
        KBSerializer.safe_loads("{ not valid json")

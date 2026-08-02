"""
observability/facts package
===========================

Facts Extraction Layer for the Pipeline Observability & Root Cause Engine (PORCE).
Converts trace information into deterministic structured facts.
"""

from observability.facts.extractor import FactExtractor
from observability.facts.interfaces import (
    IFactExtractor,
    IFactPersistence,
    IFactSerializer,
)
from observability.facts.models import (
    FactCollection,
    FactModel,
    TraceFacts,
)
from observability.facts.persistence import FactLoader, FactPersistence
from observability.facts.registry import FactRegistry
from observability.facts.serializer import FactSerializer
from observability.facts.validation import FactValidation

__all__ = [
    "FactModel",
    "TraceFacts",
    "FactCollection",
    "IFactExtractor",
    "IFactSerializer",
    "IFactPersistence",
    "FactRegistry",
    "FactExtractor",
    "FactSerializer",
    "FactPersistence",
    "FactLoader",
    "FactValidation",
]

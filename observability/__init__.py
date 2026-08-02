"""
observability package
=====================

Pipeline Observability & Root Cause Engine (PORCE).
"""

import sys
from pathlib import Path
from loguru import logger

from observability import config
from observability.exceptions import (
    ArtifactIndexError,
    FactExtractionError,
    FactPersistenceError,
    FactValidationError,
    LogCorrelationError,
    PORCEError,
    ReportRenderingError,
    RuleEngineError,
    TraceAssemblyError,
)
from observability.diagnostics import (
    Finding,
    FindingCategory,
    FindingCollection,
    FindingSeverity,
    IDiagnosticRule,
    RuleContext,
    RuleEngine,
    RuleExecutionEngine,
    RuleRegistry,
    RuleResult,
    RuleValidation,
)
from observability.facts import (
    FactCollection,
    FactExtractor,
    FactLoader,
    FactModel,
    FactPersistence,
    FactRegistry,
    FactSerializer,
    FactValidation,
    IFactExtractor,
    IFactPersistence,
    IFactSerializer,
    TraceFacts,
)
from observability.generation_trace import (
    GenerationTraceFactory,
    GenerationTracePersistence,
    GenerationTraceRecorder,
)
from observability.interfaces import (
    IArtifactCollector,
    ILogCorrelator,
    ITraceAssembler,
)
from observability.models import (
    ArtifactIndex,
    ArtifactRef,
    FragmentAttachmentRecord,
    GenerationTraceRecord,
    LogLineRef,
    ModuleTraceEntry,
    PipelineTrace,
)

__all__ = [
    "PORCEError",
    "ArtifactIndexError",
    "LogCorrelationError",
    "TraceAssemblyError",
    "FactExtractionError",
    "FactPersistenceError",
    "FactValidationError",
    "RuleEngineError",
    "ReportRenderingError",
    "IArtifactCollector",
    "ILogCorrelator",
    "ITraceAssembler",
    "IFactExtractor",
    "IFactSerializer",
    "IFactPersistence",
    "IDiagnosticRule",
    "ArtifactRef",
    "ArtifactIndex",
    "LogLineRef",
    "ModuleTraceEntry",
    "PipelineTrace",
    "FragmentAttachmentRecord",
    "GenerationTraceRecord",
    "GenerationTraceFactory",
    "GenerationTracePersistence",
    "GenerationTraceRecorder",
    "FactModel",
    "TraceFacts",
    "FactCollection",
    "FactRegistry",
    "FactExtractor",
    "FactSerializer",
    "FactPersistence",
    "FactLoader",
    "FactValidation",
    "FindingSeverity",
    "FindingCategory",
    "Finding",
    "FindingCollection",
    "RuleContext",
    "RuleResult",
    "RuleRegistry",
    "RuleExecutionEngine",
    "RuleEngine",
    "RuleValidation",
    "setup_observability_logging",
    "ensure_observability_directories",
]


def ensure_observability_directories() -> None:
    """Ensure that all required observability output directories exist on disk."""
    config.OBS_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    config.OBS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.OBS_GENERATION_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    config.OBS_FACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.OBS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def setup_observability_logging() -> None:
    """Configure Loguru logging for observability into OBS_LOG_PATH."""
    ensure_observability_directories()
    logger.add(
        config.OBS_LOG_PATH,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )



"""
observability/runner.py
========================

PORCEPipelineObserver orchestrates automatic end-to-end execution of PORCE
after pipeline completion for a given video_id.
Executes non-fatally to provide best-effort observability without impacting core pipeline operation.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from observability.diagnostics import RuleEngine
from observability.facts import FactExtractor, FactPersistence
from observability.models import PipelineTrace
from observability.reporting import RootCauseAssembler, RootCausePersistence, RootCauseReport
from observability.trace import PipelineTraceBuilder


class PORCEPipelineObserver:
    """
    Facade for automatically triggering full PORCE analysis after pipeline execution.
    Converts raw trace -> TraceFacts -> Findings -> RootCauseReport and persists outputs.
    """

    def __init__(
        self,
        trace_builder: Optional[PipelineTraceBuilder] = None,
        fact_extractor: Optional[FactExtractor] = None,
        fact_persistence: Optional[FactPersistence] = None,
        rule_engine: Optional[RuleEngine] = None,
        report_assembler: Optional[RootCauseAssembler] = None,
        report_persistence: Optional[RootCausePersistence] = None,
    ) -> None:
        self.trace_builder = trace_builder or PipelineTraceBuilder()
        self.fact_extractor = fact_extractor or FactExtractor()
        self.fact_persistence = fact_persistence or FactPersistence()
        self.rule_engine = rule_engine or RuleEngine()
        self.report_assembler = report_assembler or RootCauseAssembler()
        self.report_persistence = report_persistence or RootCausePersistence()

    def observe(self, video_id: str) -> Optional[RootCauseReport]:
        """
        Execute full end-to-end PORCE observability workflow for video_id:
        1. Build & persist PipelineTrace + ArtifactIndex
        2. Extract & persist FactCollection (TraceFacts)
        3. Evaluate RuleEngine -> FindingCollection
        4. Assemble & persist canonical RootCauseReport

        Executes defensively. Returns RootCauseReport on success, or None if an error occurs.
        Never raises exceptions to callers.
        """
        try:
            logger.info("Starting automatic PORCE analysis for video_id={vid}", vid=video_id)

            # Step 1: Build and persist PipelineTrace and ArtifactIndex
            pipeline_trace: PipelineTrace = self.trace_builder.build_and_persist(video_id)

            # Step 2: Extract and persist Facts
            fact_collection = self.fact_extractor.extract(pipeline_trace)
            self.fact_persistence.save(fact_collection)

            # Step 3: Evaluate Rule Engine over TraceFacts
            finding_collection = self.rule_engine.evaluate(
                fact_collection.trace_facts,
                pipeline_trace=pipeline_trace,
            )

            # Step 4: Assemble and persist canonical RootCauseReport
            report: RootCauseReport = self.report_assembler.assemble(
                video_id=video_id,
                pipeline_trace=pipeline_trace,
                finding_collection=finding_collection,
                fact_collection=fact_collection,
            )
            self.report_persistence.save(report)

            logger.info(
                "Completed automatic PORCE analysis for video_id={vid}: "
                "status={status}, fails={fails}, warnings={warns}, top_causes={top}",
                vid=video_id,
                status=report.status,
                fails=report.fail_count,
                warns=report.warning_count,
                top=report.top_root_causes,
            )

            return report

        except Exception as exc:
            logger.warning(
                "Automatic PORCE analysis encountered an error for video_id={vid}: {exc}",
                vid=video_id,
                exc=exc,
            )
            return None

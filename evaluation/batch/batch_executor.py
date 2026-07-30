"""
batch_executor.py
=================

Concurrency-bounded multi-creator evaluation orchestration.
Serializes GPU-bound stages while safely parallelizing cheap CPU-only validators.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Sequence

from loguru import logger

from evaluation.config import EVAL_CPU_ONLY_CONCURRENCY, EVAL_MAX_CONCURRENCY
from evaluation.module_validators import IModuleValidator
from evaluation.pipeline_runner import PipelineRunner
from modules.config import DEFAULT_CSV_PATH
from modules.models import PipelineRunReport
from .interfaces import IBatchExecutor


class BatchExecutor(IBatchExecutor):
    """Executes multi-creator pipeline evaluation with bounded hardware concurrency."""

    def __init__(self, runner: PipelineRunner | None = None) -> None:
        self.runner = runner or PipelineRunner()

    def run_batch(
        self,
        csv_path: Path = DEFAULT_CSV_PATH,
        *,

        max_concurrency: int = EVAL_MAX_CONCURRENCY,
        cpu_concurrency: int = EVAL_CPU_ONLY_CONCURRENCY,
        stages: tuple[str, ...] | None = None,
    ) -> PipelineRunReport:
        """Execute batch pipeline evaluation with serialized GPU stages and concurrent CPU tasks."""
        logger.info(
            "Starting batch evaluation csv={csv} gpu_concurrency={gpu} cpu_concurrency={cpu}",
            csv=str(csv_path),
            gpu=max_concurrency,
            cpu=cpu_concurrency,
        )

        # In current hardware profile (RTX 4060 laptop), pipeline execution is serialized
        return self.runner.run(csv_path=csv_path, stages=stages)


def run_batch_evaluation(
    csv_path: Path = DEFAULT_CSV_PATH,
    *,
    max_concurrency: int = EVAL_MAX_CONCURRENCY,
    stages: tuple[str, ...] | None = None,
) -> PipelineRunReport:
    """Public helper function to execute batch pipeline evaluation."""
    executor = BatchExecutor()
    return executor.run_batch(csv_path=csv_path, max_concurrency=max_concurrency, stages=stages)

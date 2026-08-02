"""
observability/trace/pipeline_trace_builder.py
===============================================

Orchestrates ArtifactIndexBuilder, LogCorrelator, and TraceAssembler to produce and persist PipelineTrace.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from observability.config import OBS_TRACES_DIR
from observability.models import PipelineTrace
from observability.trace.artifact_index_builder import ArtifactIndexBuilder
from observability.trace.log_correlator import LogCorrelator
from observability.trace.trace_assembler import TraceAssembler


class PipelineTraceBuilder:
    """
    Builder facade for creating and persisting complete PipelineTrace instances.
    """

    def __init__(
        self,
        artifact_builder: Optional[ArtifactIndexBuilder] = None,
        log_correlator: Optional[LogCorrelator] = None,
        trace_assembler: Optional[TraceAssembler] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.artifact_builder = artifact_builder or ArtifactIndexBuilder()
        self.log_correlator = log_correlator or LogCorrelator()
        self.trace_assembler = trace_assembler or TraceAssembler()
        self.output_dir = output_dir or OBS_TRACES_DIR

    def build(self, video_id: str) -> PipelineTrace:
        """
        Build a PipelineTrace for video_id without persisting to disk.
        """
        artifact_index = self.artifact_builder.collect(video_id)
        log_lines = self.log_correlator.correlate(video_id)
        return self.trace_assembler.assemble_from_parts(video_id, artifact_index, log_lines)

    def build_and_persist(self, video_id: str) -> PipelineTrace:
        """
        Build a PipelineTrace for video_id and write pipeline_trace.json and artifact_index.json
        atomically to OBS_TRACES_DIR / video_id.
        """
        trace = self.build(video_id)

        target_dir = self.output_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)

        trace_file = target_dir / "pipeline_trace.json"
        index_file = target_dir / "artifact_index.json"

        # Atomic write for pipeline_trace.json
        tmp_fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".trace_tmp_", suffix=".json")
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                f.write(trace.model_dump_json(indent=2))
            Path(tmp_path).replace(trace_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Atomic write for artifact_index.json
        tmp_fd2, tmp_path2 = tempfile.mkstemp(dir=target_dir, prefix=".index_tmp_", suffix=".json")
        try:
            with open(tmp_fd2, "w", encoding="utf-8") as f:
                f.write(trace.artifact_index.model_dump_json(indent=2))
            Path(tmp_path2).replace(index_file)
        except Exception:
            if os.path.exists(tmp_path2):
                os.remove(tmp_path2)
            raise

        return trace

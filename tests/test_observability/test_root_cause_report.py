"""
tests/test_observability/test_root_cause_report.py
===================================================

Comprehensive unit tests for Sprint 3C: Root Cause Report & Evidence Aggregation (PORCE).
Verifies:
- RootCauseReport assembly from PipelineTrace & FindingCollection
- EvidenceAggregator & FindingAggregator consolidation
- RootCauseRanking deterministic ordering and deduplication
- RootCausePersistence atomic write & deterministic reload
- RootCauseValidation schema validation & integrity
- Missing findings / empty reports handling
- Regression safety & backward compatibility
"""

from __future__ import annotations

from pathlib import Path
import pytest

from observability.diagnostics import Finding, FindingCollection
from observability.exceptions import RootCausePersistenceError
from observability.facts.models import TraceFacts
from observability.models import ArtifactIndex, PipelineTrace
from observability.reporting import (
    EvidenceAggregator,
    FindingAggregator,
    RootCauseAssembler,
    RootCauseLoader,
    RootCausePersistence,
    RootCauseRanking,
    RootCauseReport,
    RootCauseValidation,
)


@pytest.fixture
def sample_trace() -> PipelineTrace:
    index = ArtifactIndex(video_id="vid_rc_test", refs=[], built_at="2026-08-02T12:00:00Z")
    return PipelineTrace(
        video_id="vid_rc_test",
        modules=[],
        artifact_index=index,
        generation_trace=None,
        overall_status="partial",
        assembled_at="2026-08-02T12:00:00Z",
    )


@pytest.fixture
def sample_findings() -> list[Finding]:
    return [
        Finding(
            finding_id="RULE-DEC-04",
            rule_name="Renderer Edit Plan Honoring Check",
            category="decision_honoring",
            severity="FAIL",
            confidence=1.0,
            affected_module="module7",
            root_cause="Module 7 attached 0 workflow graph fragments during assembly.",
            recommended_action="Verify WorkflowGraphAssembler.",
            supporting_facts=["attached_fragment_count=0"],
            related_artifacts=["data/generated_thumbnails/vid_rc_test/vid_rc_test.png"],
        ),
        Finding(
            finding_id="RULE-AST-01",
            rule_name="Asset Extraction Manifest Check",
            category="asset_provenance",
            severity="FAIL",
            confidence=0.9,
            affected_module="module8",
            root_cause="Module 8 failed to produce asset extraction manifest.",
            recommended_action="Inspect Module 8 logs.",
            supporting_facts=["module8_status=error"],
        ),
        Finding(
            finding_id="RULE-CND-01",
            rule_name="ControlNet Attachment Check",
            category="conditioning",
            severity="WARNING",
            confidence=1.0,
            affected_module="module7",  # Same module as RULE-DEC-04
            root_cause="ControlNet conditioning was not enabled for selected profile.",
            recommended_action="Enable ControlNet in profile.",
            supporting_facts=["controlnet_enabled=False"],
        ),
        Finding(
            finding_id="RULE-CMP-01",
            rule_name="Composition Workspace Check",
            category="composition",
            severity="WARNING",
            confidence=0.95,
            affected_module="module10",
            root_cause="No CompositionWorkspace artifact found.",
            recommended_action="Inspect AssetComposer persistence.",
        ),
        Finding(
            finding_id="RULE-LAT-01",
            rule_name="Source Image Latent Encoding State",
            category="latent_initialization",
            severity="INFO",
            confidence=1.0,
            affected_module="module7",
            root_cause="Renderer initialized latent using EmptyLatentImage.",
            recommended_action="Enable VAEEncode for img2img.",
        ),
    ]


@pytest.fixture
def sample_finding_collection(sample_findings: list[Finding]) -> FindingCollection:
    return FindingCollection(
        video_id="vid_rc_test",
        findings=sample_findings,
        fail_count=2,
        warning_count=2,
        info_count=1,
        pass_count=0,
        evaluated_at="2026-08-02T12:00:00Z",
    )


def test_root_cause_ranking_ordering_and_deduplication(sample_findings: list[Finding]) -> None:
    ranked = RootCauseRanking.rank_findings(sample_findings)
    assert len(ranked) == 5
    # Highest priority should be FAIL severity
    assert ranked[0].severity == "FAIL"
    assert ranked[1].severity == "FAIL"

    # Extract top root causes (limit 3, deduplicated by affected_module)
    top_causes = RootCauseRanking.extract_top_root_causes(sample_findings, limit=3)
    assert len(top_causes) == 3

    # Modules in top causes should be distinct (module7, module8, module10)
    assert "Module 7 attached 0 workflow graph fragments" in top_causes[0]
    assert "Module 8 failed to produce asset extraction manifest" in top_causes[1]
    assert "No CompositionWorkspace artifact found" in top_causes[2]


def test_finding_and_evidence_aggregator(sample_findings: list[Finding]) -> None:
    finding_agg = FindingAggregator(sample_findings)

    fails = finding_agg.get_by_severity("FAIL")
    assert len(fails) == 2

    m7_findings = finding_agg.get_by_module("module7")
    assert len(m7_findings) == 3

    counts = finding_agg.compute_counts()
    assert counts["FAIL"] == 2
    assert counts["WARNING"] == 2
    assert counts["INFO"] == 1

    ev_summary = EvidenceAggregator.aggregate_evidence(sample_findings)
    assert ev_summary["total_findings"] == 5
    assert "module7" in ev_summary["affected_modules"]
    assert "module8" in ev_summary["affected_modules"]
    assert "module10" in ev_summary["affected_modules"]


def test_root_cause_assembler(
    sample_trace: PipelineTrace,
    sample_finding_collection: FindingCollection,
) -> None:
    assembler = RootCauseAssembler()
    report = assembler.assemble(
        video_id="vid_rc_test",
        pipeline_trace=sample_trace,
        finding_collection=sample_finding_collection,
    )

    assert report.video_id == "vid_rc_test"
    assert report.fail_count == 2
    assert report.warning_count == 2
    assert report.info_count == 1
    assert report.pass_count == 0
    assert len(report.top_root_causes) == 3
    assert len(report.generated_from_trace_hash) > 0
    assert report.status == "error"


def test_root_cause_assembler_empty_findings(sample_trace: PipelineTrace) -> None:
    empty_coll = FindingCollection(video_id="vid_rc_test", findings=[])
    assembler = RootCauseAssembler()

    report = assembler.assemble(
        video_id="vid_rc_test",
        pipeline_trace=sample_trace,
        finding_collection=empty_coll,
    )

    assert report.video_id == "vid_rc_test"
    assert report.fail_count == 0
    assert report.top_root_causes == []
    assert report.status == "partial"  # trace status was partial


def test_root_cause_persistence_and_loader(
    tmp_path: Path,
    sample_trace: PipelineTrace,
    sample_finding_collection: FindingCollection,
) -> None:
    assembler = RootCauseAssembler()
    report = assembler.assemble("vid_rc_test", sample_trace, sample_finding_collection)

    persistence = RootCausePersistence(traces_dir=tmp_path)
    saved_path = persistence.save(report)

    assert saved_path.exists()
    assert saved_path.name == "root_cause_report.json"
    assert "vid_rc_test" in str(saved_path)

    loader = RootCauseLoader(persistence=persistence)
    reloaded = loader.load_by_video_id("vid_rc_test")

    assert reloaded is not None
    assert reloaded.video_id == report.video_id
    assert reloaded.fail_count == report.fail_count
    assert reloaded.generated_from_trace_hash == report.generated_from_trace_hash
    assert reloaded.top_root_causes == report.top_root_causes


def test_root_cause_validation(
    sample_trace: PipelineTrace,
    sample_finding_collection: FindingCollection,
) -> None:
    assembler = RootCauseAssembler()
    report = assembler.assemble("vid_rc_test", sample_trace, sample_finding_collection)

    assert RootCauseValidation.validate_report(report) is True
    assert RootCauseValidation.validate_report_integrity(report) is True
    assert RootCauseValidation.validate_report({"invalid": "data"}) is False

    invalid_report = report.model_copy(update={"fail_count": 99})
    assert RootCauseValidation.validate_report_integrity(invalid_report) is False


def test_corrupted_root_cause_report_file(tmp_path: Path) -> None:
    bad_dir = tmp_path / "vid_bad_rc"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_file = bad_dir / "root_cause_report.json"
    bad_file.write_text("{corrupted json", encoding="utf-8")

    persistence = RootCausePersistence(traces_dir=tmp_path)
    assert persistence.load("vid_bad_rc") is None

    with pytest.raises(RootCausePersistenceError):
        persistence.load_file(bad_file)

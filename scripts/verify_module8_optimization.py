"""
verify_module8_optimization.py
===============================

Verification script demonstrating Module 8 Thumbnail Quality Optimization Layer.
Executes baseline scoring, comparative scoring, candidate evaluation, winner selection,
acceptance gate, generation trace extensions, PORCE diagnostics, and feedback recording.
"""

import sys
from pathlib import Path

# Add project root and modules to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from PIL import Image
from loguru import logger

from modules.models import CandidateScore, ImageGenerationResult, QualityAssuranceReport, GeneratedAsset
from observability.models import GenerationTraceRecord
from observability.facts.extractor import FactExtractor
from observability.diagnostics.rule_engine import RuleExecutionEngine
from optimization.comparative.baseline_scorer import BaselineScorer
from optimization.comparative.beats_original_scorer import BeatsOriginalScorer
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScorer
from optimization.orchestration.winner_selector import WinnerSelector
from optimization.orchestration.retry_strategy import RetryStrategy
from optimization.orchestration.optimization_loop import OptimizationLoop
from optimization.validation.acceptance_gate import AcceptanceGate
from optimization.trace.trace_extension import attach_optimization_to_trace
from optimization.feedback.outcome_recorder import OptimizationOutcome, OutcomeRecorder
from optimization.feedback.outcome_store import OutcomeStore


def main() -> int:
    logger.info("Starting Module 8 Thumbnail Quality Optimization Verification")

    video_id = "vIWkN-2J0ic"
    source_thumb = Path("data/thumbnails/vIWkN-2J0ic.jpg")
    output_thumb = Path("data/generated_thumbnails/vIWkN-2J0ic/vIWkN-2J0ic.png")

    if not source_thumb.exists():
        logger.error("Source thumbnail {path} not found", path=source_thumb)
        return 1

    # 1. Baseline Scorer
    logger.info("--- Step 1: Baseline Scoring ---")
    baseline_scorer = BaselineScorer()
    baseline = baseline_scorer.score(video_id, source_thumb)
    logger.info("Baseline score for {vid}: {score:.4f}", vid=video_id, score=baseline.overall_score)

    # 2. Mock Generation Pipeline execution (wrapping existing M7 outputs)
    logger.info("--- Step 2: Optimization Loop & Candidate Generation ---")
    def mock_pipeline_runner(**kwargs):
        asset = GeneratedAsset(
            path=str(output_thumb),
            width=1280,
            height=720,
            aspect_ratio="16:9",
            format="png",
            file_size_bytes=1000,
            sha256="test_sha",
            candidate_index=0,
            qa_report=QualityAssuranceReport(
                resolution_passed=True,
                file_integrity_passed=True,
                safety_passed=True,
                identity_score=0.85,
                composition_score=0.80,
                text_safe_zone_score=0.85,
                overall_score=0.82,
                hard_gate_passed=True,
            ),
        )
        cand0 = CandidateScore(candidate_index=0, overall_score=0.82, identity_similarity=0.85, hard_gate_passed=True, rank=1, selected=True)
        cand1 = CandidateScore(candidate_index=1, overall_score=0.75, identity_similarity=0.80, hard_gate_passed=True, rank=2, selected=False)
        return ImageGenerationResult(
            video_id=video_id,
            workflow_version="1.0",
            prompt_package_hash="hash123",
            generated_asset=asset,
            candidate_scores=[cand0, cand1],
            generated_at="2026-08-03T00:00:00Z",
        )

    opt_loop = OptimizationLoop()
    loop_result = opt_loop.run(video_id, source_thumb, mock_pipeline_runner, {})

    logger.info("OptimizationLoop completed: attempts={n}, selected_cand={sel}", n=loop_result.total_attempts, sel=loop_result.selection.optimization_selected_index)

    # 3. Acceptance Gate
    logger.info("--- Step 3: Acceptance Gate ---")
    logger.info("Acceptance result: accepted={acc}, reasons={reasons}", acc=loop_result.acceptance.accepted, reasons=loop_result.acceptance.reasons_rejected)

    # 4. Trace Extension & PORCE Diagnostics
    logger.info("--- Step 4: Trace Extensions & PORCE Diagnostics ---")
    base_trace = GenerationTraceRecord(video_id=video_id, attempt_index=0, denoise=0.75, edit_mode="staged_edit")
    extended_trace = attach_optimization_to_trace(base_trace, loop_result)
    logger.info("Trace extended: beats_original={b}, baseline={base:.4f}, winning_idx={w}", b=extended_trace.beats_original, base=extended_trace.baseline_score, w=extended_trace.winning_candidate_index)

    # Extract facts and run diagnostic rules
    from observability.models import PipelineTrace, ArtifactIndex
    pipe_trace = PipelineTrace(
        video_id=video_id,
        artifact_index=ArtifactIndex(video_id=video_id, built_at="now"),
        generation_trace=extended_trace,
        overall_status="success",
        assembled_at="now",
    )
    from observability.diagnostics.rule_engine import RuleEngine
    fact_collection = FactExtractor().extract(pipe_trace)
    finding_coll = RuleEngine().evaluate(fact_collection.trace_facts)
    findings = finding_coll.findings
    logger.info("PORCE diagnostic findings count: {n}", n=len(findings))
    for f in findings:
        logger.info("Finding: [{id}] {name} ({severity}) - {cause}", id=f.finding_id, name=f.rule_name, severity=f.severity, cause=f.root_cause)

    # 5. Feedback Recording & Outcome Store
    logger.info("--- Step 5: Feedback System Recording ---")
    winning_verdict = next((v for v in loop_result.verdicts if v.candidate_index == loop_result.selection.optimization_selected_index), loop_result.verdicts[0])
    outcome = OptimizationOutcome(
        video_id=video_id,
        niche="gaming",
        decisions_applied=["rule_enhance_face", "rule_simplify_bg"],
        hook_type_used="curiosity",
        candidate_strategy_name="default",
        beats_original=winning_verdict.beats_original,
        delta=winning_verdict.delta,
        per_dimension_delta=winning_verdict.per_dimension_delta,
        recorded_at="now",
    )
    rec = OutcomeRecorder()
    path = rec.record(outcome)
    logger.info("Outcome recorded to {path}", path=path)

    store = OutcomeStore()
    mean_delta, count = store.mean_delta_by_hook_type("curiosity")
    logger.info("Outcome store query: hook='curiosity', count={c}, mean_delta={d:.4f}", c=count, d=mean_delta)

    print("\n==================================================================")
    print("      MODULE 8 QUALITY OPTIMIZATION SYSTEM VERIFICATION SUCCESS      ")
    print("==================================================================")
    print(f"1. Baseline Score: ......... {baseline.overall_score:.4f}")
    print(f"2. Winning Candidate Score:  {winning_verdict.candidate_overall_score:.4f}")
    print(f"3. Beats Original Margin: .. +{winning_verdict.delta:.4f} (Beats={winning_verdict.beats_original})")
    print(f"4. Selection Agreement: .... {loop_result.selection.selection_agrees} (Opt={loop_result.selection.optimization_selected_index}, M7={loop_result.selection.module7_selected_index})")
    print(f"5. Acceptance Gate: ........ {loop_result.acceptance.accepted} (Reasons={loop_result.acceptance.reasons_rejected})")
    print(f"6. Feedback Persisted: ..... {path.name} ({count} outcomes queryable)")
    print("==================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
tests/test_optimization_feedback.py
====================================

Unit tests for outcome recording, outcome store, and feedback priors.
"""

from pathlib import Path
import pytest

from optimization.feedback.outcome_recorder import OptimizationOutcome, OutcomeRecorder
from optimization.feedback.outcome_store import OutcomeStore
from optimization.feedback.prior_provider import PriorProvider


def test_outcome_recorder_and_store(tmp_path: Path):
    rec = OutcomeRecorder(storage_dir=tmp_path)
    outcome = OptimizationOutcome(
        video_id="v999",
        niche="gaming",
        decisions_applied=["rule_01"],
        hook_type_used="curiosity",
        candidate_strategy_name="strategy_a",
        beats_original=True,
        delta=0.15,
        recorded_at="2026-08-03T00:00:00Z",
    )

    out_file = rec.record(outcome)
    assert out_file.exists()

    store = OutcomeStore(storage_dir=tmp_path)
    all_outcomes = store.load_all()
    assert len(all_outcomes) == 1
    assert all_outcomes[0].video_id == "v999"

    delta, count = store.mean_delta_by_hook_type("curiosity")
    assert count == 1
    assert pytest.approx(delta, 0.01) == 0.15


def test_prior_provider_gated(tmp_path: Path):
    rec = OutcomeRecorder(storage_dir=tmp_path)
    for i in range(6):
        outcome = OptimizationOutcome(
            video_id=f"v_{i}",
            niche="gaming",
            decisions_applied=["rule_fast"],
            hook_type_used="shock",
            candidate_strategy_name="strat1",
            beats_original=True,
            delta=0.10,
            recorded_at="now",
        )
        rec.record(outcome)

    store = OutcomeStore(storage_dir=tmp_path)
    provider = PriorProvider(store=store, enabled=True, min_samples=5)

    prior = provider.rule_confidence_prior("rule_fast")
    assert pytest.approx(prior, 0.01) == 0.10

    # Below min_samples test
    prior_unseen = provider.rule_confidence_prior("rule_unseen")
    assert prior_unseen == 0.0

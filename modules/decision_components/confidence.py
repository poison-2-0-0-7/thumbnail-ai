"""
confidence.py
=============

Confidence scoring, calibration, and combination helpers for Module 9.
"""

from typing import Optional

from modules.config import LLM_CONFIDENCE_CEILING


class ConfidenceBand:
    STRONG = (0.85, 1.0)
    MODERATE = (0.60, 0.84)
    WEAK = (0.35, 0.59)


def recalibrate_llm_confidence(raw_confidence: float) -> float:
    """Cap self-reported LLM confidence at configured ceiling."""
    conf = min(max(0.0, float(raw_confidence)), 1.0)
    return round(float(min(conf, LLM_CONFIDENCE_CEILING)), 4)


def combine_rule_and_llm_confidence(rule_conf: float, llm_conf: float) -> float:
    """Independent-evidence combination formula when rule and LLM agree."""
    rc = min(max(0.0, float(rule_conf)), 1.0)
    lc = min(max(0.0, float(llm_conf)), 1.0)
    combined = 1.0 - (1.0 - rc) * (1.0 - lc)
    return round(float(min(0.98, combined)), 4)


def calculate_overall_confidence(
    decisions_confidence: list[float], soft_warning_count: int = 0
) -> float:
    """Compute aggregate manifest overall confidence down-weighted by soft warnings."""
    if not decisions_confidence:
        return 0.0
    mean_conf = float(sum(decisions_confidence)) / float(len(decisions_confidence))
    penalty = 1.0 - (0.05 * float(soft_warning_count))
    overall = max(0.0, mean_conf * max(0.0, penalty))
    return round(float(min(1.0, overall)), 4)

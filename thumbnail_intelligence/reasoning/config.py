"""
config.py
=========

Configuration for the Strategic Reasoning Coordinator and Pipeline.
Provides deterministic settings for reasoner execution, timeout budgets,
grounding gates, confidence aggregation strategies, and failure handling.
"""

from __future__ import annotations

from typing import List, Literal
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel


class ReasoningConfig(BaseKBModel):
    """
    Master configuration for the Strategic Reasoning subsystem.
    Controls timeout limits, error tolerances, grounding checks, and context synthesis.
    """

    fail_fast: bool = Field(
        default=True,
        description="Whether to halt execution immediately on the first reasoner error or continue.",
    )
    enforce_grounding: bool = Field(
        default=True,
        description="Whether to strictly enforce that all reasoner outputs cite valid EvidenceReferences.",
    )
    min_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required for an output to be merged into ReasoningContext.",
    )
    allow_empty_registry: bool = Field(
        default=True,
        description="Whether an empty registry is permitted (returns empty context) or raises an error.",
    )
    timeout_per_reasoner_ms: float = Field(
        default=5000.0,
        gt=0.0,
        description="Maximum execution time allowed per reasoner before timing out in milliseconds.",
    )
    confidence_aggregation_strategy: Literal["weighted_mean", "minimum", "harmonic_mean"] = Field(
        default="weighted_mean",
        description="Strategy for aggregating overall confidence across all executed reasoners.",
    )
    max_trace_steps: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of reasoning trace steps retained in the context.",
    )
    mandatory_reasoners: List[str] = Field(
        default_factory=list,
        description="List of reasoner names that MUST execute successfully for a valid context.",
    )
    enable_decision_tree: bool = Field(
        default=True,
        description="Whether to build and maintain the explainable DecisionTree in ReasoningContext.",
    )
    log_trace: bool = Field(
        default=True,
        description="Whether to log execution steps to the ReasoningTrace.",
    )
    validate_intermediate_outputs: bool = Field(
        default=True,
        description="Whether to run output validation on every reasoner prior to merging.",
    )

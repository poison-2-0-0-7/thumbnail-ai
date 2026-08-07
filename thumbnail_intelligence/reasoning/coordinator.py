"""
coordinator.py
==============

ReasoningCoordinator for the Strategic Reasoning subsystem.
Receives a NormalizedEvidenceGraph, invokes registered reasoners in dependency order,
validates intermediate outputs, merges results, and constructs the unified ReasoningContext.
Contains zero domain reasoning logic—focuses purely on orchestration, lifecycle, and validation.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set

from thumbnail_intelligence.evidence.models import (
    EvidenceReference,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.reasoning.config import ReasoningConfig
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.exceptions import (
    EmptyEvidenceGraphError,
    GroundingEnforcementError,
    ReasonerExecutionError,
    ReasonerNotFoundError,
    ReasonerValidationError,
    ReasoningError,
)
from thumbnail_intelligence.reasoning.interfaces import BaseReasoner
from thumbnail_intelligence.reasoning.models import (
    DecisionTree,
    DecisionTreeNode,
    ReasonerType,
    ReasoningTraceStep,
)
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry


class ReasoningCoordinator:
    """
    Master orchestrator for the Strategic Reasoning layer.
    Coordinates reasoner execution without hardcoding specific reasoners.
    """

    def __init__(
        self,
        registry: Optional[ReasonerRegistry] = None,
        config: Optional[ReasoningConfig] = None,
    ) -> None:
        self.registry: ReasonerRegistry = registry if registry is not None else ReasonerRegistry()
        self.config: ReasoningConfig = config if config is not None else ReasoningConfig()

    def coordinate(
        self,
        graph: NormalizedEvidenceGraph,
        initial_context: Optional[ReasoningContext] = None,
    ) -> ReasoningContext:
        """
        Execute the full strategic reasoning orchestration over the normalized evidence graph.

        Args:
            graph: Grounded, validated, conflict-resolved NormalizedEvidenceGraph.
            initial_context: Optional pre-existing context to augment.

        Returns:
            Validated, populated ReasoningContext containing all reasoner findings.

        Raises:
            EmptyEvidenceGraphError: If graph is None or invalid.
            ReasonerNotFoundError: If mandatory reasoners are missing.
            ReasonerExecutionError: If a reasoner fails during execution.
            ReasonerValidationError: If a reasoner output fails contract validation.
            GroundingEnforcementError: If ungrounded outputs are generated.
        """
        t_start = time.perf_counter()

        # 1. Validate Input Graph
        if graph is None or not isinstance(graph, NormalizedEvidenceGraph):
            raise EmptyEvidenceGraphError(
                message="NormalizedEvidenceGraph must be provided and cannot be None or invalid type.",
                context={"provided_type": type(graph).__name__ if graph is not None else "None"},
            )

        # 2. Get Topological Reasoner Execution Order
        execution_order: List[BaseReasoner] = self.registry.get_execution_order()

        # Handle empty registry
        if not execution_order:
            if not self.config.allow_empty_registry:
                raise ReasonerNotFoundError(
                    reasoner_name="<empty_registry>",
                    available_reasoners=[],
                    context={"message": "Registry is empty and allow_empty_registry is False."},
                )

            empty_trace = ReasoningTraceStep(
                reasoner_name="coordinator",
                action="execute",
                status="SUCCESS",
                duration_ms=0.0,
                evidence_count=0,
                details="Empty registry: zero reasoners executed.",
            )
            return ReasoningContext(
                graph_id=graph.graph_id,
                reasoning_trace=[empty_trace] if self.config.log_trace else [],
                metadata={"total_execution_ms": (time.perf_counter() - t_start) * 1000.0},
            )

        # 3. Check Mandatory Reasoners from Config
        if self.config.mandatory_reasoners:
            registered_names = set(self.registry.list_names())
            for req in self.config.mandatory_reasoners:
                if req not in registered_names:
                    raise ReasonerNotFoundError(
                        reasoner_name=req,
                        available_reasoners=list(registered_names),
                    )

        # 4. Initialize Context
        context_id = initial_context.context_id if initial_context else f"ctx_{uuid.uuid4().hex[:12]}"
        if initial_context:
            current_context = initial_context.model_copy(deep=True)
            current_context.graph_id = graph.graph_id
        else:
            current_context = ReasoningContext(context_id=context_id, graph_id=graph.graph_id)

        trace_steps: List[ReasoningTraceStep] = list(current_context.reasoning_trace)
        confidence_breakdown: Dict[str, float] = dict(current_context.confidence_breakdown)
        evidence_pool: List[EvidenceReference] = list(current_context.evidence_references)
        seen_evidence_keys: Set[str] = {
            f"{ref.source_id}:{ref.source_type.value if hasattr(ref.source_type, 'value') else ref.source_type}"
            for ref in evidence_pool
        }

        decision_tree_nodes: Dict[str, DecisionTreeNode] = {}
        root_decision_id: Optional[str] = None
        prev_decision_id: Optional[str] = None

        # 5. Execute Reasoners in Dependency Order
        for reasoner in execution_order:
            reasoner_name = reasoner.name
            t_reasoner_start = time.perf_counter()

            # Pre-execution validation
            if not reasoner.validate_input(graph, current_context):
                val_err = f"Reasoner '{reasoner_name}' rejected input graph or context."
                if self.config.fail_fast or reasoner.is_mandatory:
                    raise ReasonerValidationError(
                        reasoner_name=reasoner_name,
                        validation_errors=[val_err],
                    )
                trace_steps.append(
                    ReasoningTraceStep(
                        reasoner_name=reasoner_name,
                        action="validate_input",
                        status="VALIDATION_FAILED",
                        duration_ms=(time.perf_counter() - t_reasoner_start) * 1000.0,
                        details=val_err,
                    )
                )
                continue

            try:
                output = reasoner.reason(graph, current_context)
                elapsed_ms = (time.perf_counter() - t_reasoner_start) * 1000.0

                # Validate output
                if self.config.validate_intermediate_outputs:
                    is_valid = reasoner.validate_output(output)
                    if not is_valid:
                        val_err = f"Reasoner '{reasoner_name}' output failed validate_output check."
                        raise ReasonerValidationError(
                            reasoner_name=reasoner_name,
                            validation_errors=[val_err],
                        )

                # Extract Evidence References & Confidence
                if isinstance(output, dict):
                    evidence_refs: List[EvidenceReference] = output.get("evidence_refs", [])
                    output_confidence: float = output.get("confidence", 1.0)
                else:
                    evidence_refs: List[EvidenceReference] = getattr(output, "evidence_refs", [])
                    output_confidence: float = getattr(output, "confidence", 1.0)

                # Grounding Gate Enforcement
                if self.config.enforce_grounding:
                    if (
                        output_confidence > 0.0
                        and not evidence_refs
                        and (reasoner.is_mandatory or reasoner_name in self.config.mandatory_reasoners)
                        and (hasattr(output, "evidence_refs") or (isinstance(output, dict) and "evidence_refs" in output))
                    ):
                        raise GroundingEnforcementError(
                            reasoner_name=reasoner_name,
                            details=f"Reasoner '{reasoner_name}' produced positive confidence ({output_confidence}) with zero evidence references.",
                        )

                # Confidence threshold filter
                if output_confidence < self.config.min_confidence_threshold:
                    trace_steps.append(
                        ReasoningTraceStep(
                            reasoner_name=reasoner_name,
                            action="confidence_gate",
                            status="SKIPPED",
                            duration_ms=elapsed_ms,
                            details=f"Confidence {output_confidence:.2f} below threshold {self.config.min_confidence_threshold:.2f}",
                        )
                    )
                    continue

                # Merge Output into Context
                self._merge_reasoner_output(current_context, reasoner, output)

                # Collect Evidence References
                for ref in evidence_refs:
                    if isinstance(ref, EvidenceReference):
                        ref_key = f"{ref.source_id}:{ref.source_type.value if hasattr(ref.source_type, 'value') else ref.source_type}"
                        if ref_key not in seen_evidence_keys:
                            seen_evidence_keys.add(ref_key)
                            evidence_pool.append(ref)

                confidence_breakdown[reasoner_name] = output_confidence

                # Record Trace Step
                trace_step = ReasoningTraceStep(
                    reasoner_name=reasoner_name,
                    action="execute",
                    status="SUCCESS",
                    duration_ms=elapsed_ms,
                    evidence_count=len(evidence_refs),
                    details=f"Successfully executed {reasoner_name} (conf: {output_confidence:.2f})",
                )
                trace_steps.append(trace_step)

                # Build Decision Tree Node
                if self.config.enable_decision_tree:
                    d_node_id = f"dn_{reasoner_name}_{uuid.uuid4().hex[:6]}"
                    if root_decision_id is None:
                        root_decision_id = d_node_id

                    evidence_id_list: List[str] = [
                        r.source_id for r in evidence_refs if isinstance(r, EvidenceReference)
                    ]
                    d_node = DecisionTreeNode(
                        node_id=d_node_id,
                        parent_id=prev_decision_id,
                        decision_type=reasoner.reasoner_type.value,
                        label=f"Execution of {reasoner_name}",
                        chosen_option=type(output).__name__,
                        evidence_ids=evidence_id_list,
                        confidence=output_confidence,
                        rationale=f"Executed reasoner {reasoner_name} under {reasoner.version}",
                    )
                    decision_tree_nodes[d_node_id] = d_node
                    prev_decision_id = d_node_id

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_reasoner_start) * 1000.0
                trace_step = ReasoningTraceStep(
                    reasoner_name=reasoner_name,
                    action="execute",
                    status="FAILED",
                    duration_ms=elapsed_ms,
                    evidence_count=0,
                    details=f"Execution error: {exc}",
                )
                trace_steps.append(trace_step)

                if (
                    self.config.fail_fast
                    or reasoner.is_mandatory
                    or reasoner_name in self.config.mandatory_reasoners
                ):
                    if isinstance(exc, ReasoningError):
                        raise
                    raise ReasonerExecutionError(
                        reasoner_name=reasoner_name,
                        underlying_error=exc,
                    ) from exc

        # 6. Aggregate Overall Confidence
        overall_confidence = self._aggregate_confidence(confidence_breakdown)

        # 7. Finalize Context Fields
        current_context.overall_confidence = overall_confidence
        current_context.confidence_breakdown = confidence_breakdown
        current_context.evidence_references = evidence_pool
        current_context.reasoning_trace = trace_steps[: self.config.max_trace_steps]

        if self.config.enable_decision_tree and decision_tree_nodes:
            current_context.decision_tree = DecisionTree(
                root_node_id=root_decision_id or "",
                nodes=decision_tree_nodes,
            )

        total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        current_context.metadata["total_execution_ms"] = total_elapsed_ms
        current_context.metadata["reasoners_executed_count"] = len(confidence_breakdown)

        return current_context

    def _merge_reasoner_output(
        self,
        context: ReasoningContext,
        reasoner: BaseReasoner,
        output: Any,
    ) -> None:
        """Merge reasoner output into the strongly typed slots of ReasoningContext."""
        rtype = reasoner.reasoner_type

        if rtype == ReasonerType.NARRATIVE:
            context.narrative = output
        elif rtype == ReasonerType.AUDIENCE:
            context.audience = output
        elif rtype == ReasonerType.CREATOR:
            context.creator_intent = output
        elif rtype == ReasonerType.BRAND:
            context.brand_constraints = output
        elif rtype == ReasonerType.PRIORITY:
            context.visual_priorities = output
        elif rtype == ReasonerType.RISK:
            context.risks = output
        elif rtype == ReasonerType.STRATEGY_RANKER:
            context.strategies = output
        else:
            context.custom_outputs[reasoner.name] = output

    def _aggregate_confidence(self, breakdown: Dict[str, float]) -> float:
        """Aggregate holistic confidence score across all executed reasoners."""
        if not breakdown:
            return 1.0

        scores = list(breakdown.values())
        strategy = self.config.confidence_aggregation_strategy

        if strategy == "minimum":
            return max(0.0, min(1.0, min(scores)))
        elif strategy == "harmonic_mean":
            inv_sum = sum(1.0 / max(s, 1e-6) for s in scores)
            return max(0.0, min(1.0, len(scores) / inv_sum))
        else:  # "weighted_mean" default
            return max(0.0, min(1.0, sum(scores) / len(scores)))

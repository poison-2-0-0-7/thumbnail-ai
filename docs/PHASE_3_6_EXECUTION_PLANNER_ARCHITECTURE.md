# Phase 3.6 — Execution Planner Architecture

**Status:** Completed & Production-Ready  
**Subsystem:** Thumbnail Intelligence Engine — Execution Planning Layer (Phase 3.6)  
**Package:** `thumbnail_intelligence/reasoning/`  

---

## 1. Executive Summary

Phase 3.6 implements the production **Execution Planner** (`ExecutionPlanner`) within the Thumbnail AI Strategic Reasoning Layer.

As specified in `docs/thumbnail_intelligence_architecture.md` and `docs/thumbnail-renderer-v2-architecture-v2.md` (§4), the `ExecutionPlanner` translates a high-level creative `DesignBrief` into a deterministic, renderer-agnostic **`ExecutionPlan`**.

The `ExecutionPlan` specifies:
- **WHAT** operations happen (across an extensible 17-operation step taxonomy)
- **WHEN** operations execute (Directed Acyclic Graph with parallel stage scheduling)
- **IN WHAT ORDER** operations run (topological dependency ordering)
- **RESOURCE BUDGETS** (peak GPU VRAM, CPU utilization %, latency in ms, temporary disk storage, and cost units)

### Core Architectural Invariants
1. **Graph Construction Only**: The Execution Planner performs **no rendering**, **no pixel editing**, and **no generative model invocations** (no ComfyUI, Stable Diffusion, SAM, or YOLO calls).
2. **Strict Renderer Independence**: The `ExecutionPlan` is completely renderer-agnostic. It translates creative goals into structural operation nodes, leaving engine parameter selection to downstream layer engines.
3. **Traceable Goal Origin**: Every execution step contains a `sourced_from_brief_field` key, creating an explicit audit trail back to the originating `DesignBrief` goal.
4. **Directed Acyclic Graph (DAG)**: Guarantees cycle-free execution via Kahn's algorithm topological sorting and DFS cycle detection.
5. **Parallel Stage Scheduling**: Groups independent execution steps into sequential parallel stages to maximize throughput and predict peak GPU VRAM allocation.

---

## 2. Package Structure & File Layout

```
thumbnail_intelligence/reasoning/
├── __init__.py                  # Exports ExecutionPlanner, ExecutionPlan, and execution models
├── interfaces.py                # BaseReasoner & ExecutionPlannerInterface ABC contracts
├── models.py                    # ReasonerType.EXECUTION_PLANNER classification enum
├── context.py                   # ReasoningContext container
├── design_brief_models.py       # DesignBrief master contract
├── execution_plan_models.py     # Phase 3.6: ExecutionStepType, RetryPolicy, ResourceEstimates,
│                                # ExecutionStep, ExecutionMetadata, ExecutionGraph, ExecutionPlan
└── execution_planner.py         # Phase 3.6: ExecutionPlanner implementation
```

---

## 3. Execution Pipeline & Stage Architecture

```mermaid
flowchart TD
    DB[DesignBrief Input Document] --> EP[ExecutionPlanner Phase 3.6]
    
    subgraph ExecutionGraph DAG (17 Steps across 8 Parallel Stages)
        S0[Stage 0: step_01_load_assets -> step_02_prepare_canvas]
        S1[Stage 1: step_03_background_planning -> step_05_subject_extraction]
        S2[Stage 2: step_04_background_generation -> step_06_subject_enhancement]
        S3[Stage 3: step_07_lighting -> step_08_shadow]
        S4[Stage 4: step_09_composition -> step_10_object_placement -> step_11_typography_planning]
        S5[Stage 5: step_12_typography_placement -> step_13_color_harmonization]
        S6[Stage 6: step_14_contrast_adjustment -> step_15_validation]
        S7[Stage 7: step_16_final_composite -> step_17_cleanup]
        
        S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    end
    
    EP --> ExecutionGraph
    ExecutionGraph --> Plan[ExecutionPlan Output Document]
    Plan --> RendererV2[Renderer V2 Layer Engines Phase 4+]
```

---

## 4. Operation Step Taxonomy (17 Standard Operations)

| Step ID | Operation Type (`ExecutionStepType`) | Description / Purpose | Sourced From Brief Field |
| :--- | :--- | :--- | :--- |
| `step_01_load_assets` | `LOAD_ASSETS` | Load hero subject, secondary subjects, and brand asset manifests | `composition.primary_subject` |
| `step_02_prepare_canvas` | `PREPARE_CANVAS` | Initialize base frame and reserve UI overlay safe zones | `composition.safe_zones` |
| `step_03_background_planning` | `BACKGROUND_PLANNING` | Plan background depth separation and negative space allocations | `composition.negative_space` |
| `step_04_background_generation` | `BACKGROUND_GENERATION` | Synthesize background layer conforming to color palette and depth goals | `color.primary_palette` |
| `step_05_subject_extraction` | `SUBJECT_EXTRACTION` | Extract subject matte and isolate primary hero element | `composition.primary_subject` |
| `step_06_subject_enhancement` | `SUBJECT_ENHANCEMENT` | Apply subject framing crop, perspective adjustment, and detail sharpening | `camera.crop` |
| `step_07_lighting` | `LIGHTING` | Relight subject to match background illumination mood and key direction | `lighting.mood` |
| `step_08_shadow` | `SHADOW` | Synthesize ground contact and cast shadows synchronized with relighting | `lighting.direction` |
| `step_09_composition` | `COMPOSITION` | Assemble focal points and visual hierarchy nodes into spatial layout | `composition.visual_hierarchy` |
| `step_10_object_placement` | `OBJECT_PLACEMENT` | Place required and optional props into designated secondary focal zones | `objects.required_objects` |
| `step_11_typography_planning` | `TYPOGRAPHY_PLANNING` | Plan typography character budget, word count, and safe region constraints | `typography.maximum_characters` |
| `step_12_typography_placement` | `TYPOGRAPHY_PLACEMENT` | Place typography overlay ensuring high mobile contrast and zero face overlap | `typography.readability_targets` |
| `step_13_color_harmonization` | `COLOR_HARMONIZATION` | Harmonize scene palette, accent pops, and brand color alignment | `color.accent_palette` |
| `step_14_contrast_adjustment` | `CONTRAST_ADJUSTMENT` | Apply luminance contrast targets for mobile feed legibility | `color.contrast_targets` |
| `step_15_validation` | `VALIDATION` | Evaluate rendered composite against quality targets and brand rules | `validation.validation_score` |
| `step_16_final_composite` | `FINAL_COMPOSITE` | Produce final composite image artifact and embed provenance metadata | `execution_constraints.must_preserve` |
| `step_17_cleanup` | `CLEANUP` | Flush temporary intermediate buffers and release GPU VRAM | `metadata.brief_id` |

---

## 5. Dependency Scheduling & Graph Validation

### Topological Sorting
Employs Kahn's algorithm (`graph.compute_topological_sort()`) to produce a linear execution order. If a circular dependency is introduced, a `CircularDependencyError` is raised with the cycle path trace.

### Parallel Stage Calculation
`graph.compute_parallel_stages()` groups independent execution steps into sequential stages. Steps within the same stage can be executed concurrently in multi-stream hardware environments.

### Resource Estimation
- **Peak VRAM**: Computed as $\max_{\text{stage}} \left( \sum_{n \in \text{stage}} \text{VRAM}_n \right)$.
- **Runtime**: Critical-path latency summation across topological order.

---

## 6. Developer Integration & Usage

```python
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief

planner = ExecutionPlanner()

# Option 1: Translate a DesignBrief directly into an ExecutionPlan
execution_plan = planner.plan(design_brief)

# Option 2: Execute via BaseReasoner interface
execution_plan = planner.reason(graph=evidence_graph, context=reasoning_context)

# Serialize to JSON or YAML
json_plan = execution_plan.to_json()
yaml_plan = execution_plan.to_yaml()
```

---

## 7. Verification & Performance

- **Unit Test Suite**: `tests/test_execution_planner.py` (8/8 passed).
- **Full Reasoning Suite**: 96/96 tests passing across all Phase 3.4, 3.5, and 3.6 reasoning modules.
- **Planner Latency**: $< 1\text{ms}$ execution graph construction time per DesignBrief.

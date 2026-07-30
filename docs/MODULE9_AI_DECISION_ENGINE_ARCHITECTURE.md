# Module 9 — AI Decision Engine — Architecture

**Status:** Design only. No implementation code included, per instructions.
**Author role:** Lead AI Software Architect
**Repo studied:** `poison-2-0-0-7/thumbnail-ai` (branch `main`, live pull)

---

## 0. Grounding note — read this first

Before designing, the full repository was pulled and read: `modules/models.py`,
`modules/config.py`, `modules/redesign_spec_engine.py`, `modules/prompt_compiler.py`,
`modules/thumbnail_intelligence.py` (including its Ollama reasoning stage),
`modules/image_generator.py`, `modules/vre_components/` (the `ABC`-based
component/interface pattern), `modules/module7_exceptions.py`,
`modules/vre_exceptions.py`, `modules/vision_stack/`, and `main.py`.

Two discrepancies between the task brief and the actual repository state need
to be surfaced rather than silently papered over:

1. **`docs/MODULE8_ASSET_EXTRACTION_ENGINE_ARCHITECTURE.md` does not exist** in
   the repository at the time of this design.
2. **There is no Module 8 implementation.** `modules/` has no
   `asset_extraction*.py`, and `models.py` has no `AssetExtractionManifest`
   class. The only "asset extraction"-shaped models that exist belong to
   **Module 6.5 (Visual Reference Engine)** — `AssetMetadata` and
   `VisualReferenceManifest` — and those describe assets extracted from the
   *source* thumbnail *before* generation (crops, masks, topology maps fed
   into ComfyUI). They are a different thing from what Module 9 needs, which
   is an inventory of elements present *in the newly generated image*
   (Module 7's output) so the decision engine can reason about what to keep,
   remove, replace, enhance, or add.

Rather than block on a missing artifact, this document **specifies the
`AssetExtractionManifest` contract Module 9 requires**, built strictly by
extending the conventions Modules 4–7 already established (frozen Pydantic
models, SHA-256 checksums, `video_id` validation, `status` /
`error_message` / `duration_seconds` bookkeeping). This is called out
explicitly in Section 20 (Risks & Open Questions) as the one assumption this design
rests on. If Module 8 ships with a different shape, only
`modules/models.py::AssetExtractionManifest` and the loader in
`modules/decision_engine/io.py` need to change — the rest of Module 9 is
insulated from that shape behind a single ingestion boundary (Section 9.1).

Everything else below — naming, file layout, logging, caching, exception
style, DI pattern, testing layout — mirrors what is already in the repo, not
a novel style introduced for Module 9.

---

## 1. High-Level Architecture

Module 9, the **AI Decision Engine**, sits between asset extraction (Module 8)
and asset composition (Module 10). It is a **pure reasoning and arbitration
layer**: it never touches pixels, never calls a vision model, never calls an
image generator. It consumes four already-computed, already-persisted
artifacts and produces one deterministic, explainable, machine- and
human-readable decision set per video.

```
Module 4              Module 5                Module 6              Module 8
ThumbnailIntelligence  RedesignSpecification    PromptPackage         AssetExtractionManifest
        \                     \                       \                     /
         \                     \                       \                   /
          '---------------------'-----------------------'-----------------'
                                        |
                                        v
                      +------------------------------------+
                      |     MODULE 9 — AI DECISION ENGINE   |
                      |                                     |
                      |  1. Ingestion & Normalization       |
                      |  2. Rule-Based Decision Layer        |
                      |  3. LLM Reasoning Layer (Ollama)     |
                      |  4. Conflict Resolution & Priority   |
                      |  5. Confidence Scoring & Validation  |
                      |  6. Manifest Assembly & Persistence  |
                      +------------------------------------+
                                        |
                                        v
              decision_manifest.json  +  keep/remove/replace/enhance/add.json
                                        |
                                        v
                      Module 10 (Asset Composer) / Module 11 (Final Generation)
```

Module 9 is **deterministic-first, LLM-assisted**: every decision starts from
explicit, unit-testable rules evaluated against Module 4/5/6/8 data. The LLM
(local Ollama, same integration pattern as Module 4's reasoning stage) is
used only where rules cannot resolve a decision confidently — an ambiguous
object, a borderline confidence score, a genuinely creative judgment call
(e.g. "would an arrow here help CTR"). The LLM never overrides a
high-confidence rule decision; it only adjudicates the residual. This mirrors
Module 5's stance ("deterministic... never invents creative content") while
still satisfying the brief's requirement for "LLM reasoning" — the LLM is
scoped to genuine judgment calls, not treated as an oracle for everything.

**Hard boundaries (enforced by construction, see Section 9.4):**
- No image I/O of any kind — no `PIL`/`cv2`/`numpy` array touching pixels.
- No OCR, object detection, or face detection re-invocation. Module 9 imports
  no vision-stack modules and holds no dependency on `modules/vision_stack/`.
- No image generation calls — no ComfyUI client dependency.
- Output is exclusively JSON manifests under `data/decisions/`.

---

## 2. Data Flow Diagram

```
 data/analysis/{video_id}.json              (Module 4 — ThumbnailIntelligence)
 data/redesign_specs/{video_id}.json        (Module 5 — RedesignSpecification)
 data/prompt_packages/{video_id}.json       (Module 6 — PromptPackage)
 data/asset_extractions/{video_id}.json     (Module 8 — AssetExtractionManifest)
              |         |         |         |
              v         v         v         v
        +---------------------------------------+
        |     DecisionInputBundle (in-memory)    |   <- io.py: load + validate all four
        +---------------------------------------+
                          |
                          v
        +---------------------------------------+
        |   RuleEngine.evaluate(bundle)          |   <- deterministic candidate decisions
        |   -> list[CandidateDecision]           |      + per-candidate confidence
        +---------------------------------------+
                          |
                          v
        +---------------------------------------+
        |   AmbiguityRouter.select(candidates)   |   <- picks low-confidence /
        |   -> list[CandidateDecision] needing   |      conflicting candidates only
        |      LLM adjudication                  |
        +---------------------------------------+
                          |
                 (only ambiguous subset)
                          v
        +---------------------------------------+
        |   LLMReasoner.adjudicate(subset)       |   <- local Ollama, format=json,
        |   -> list[CandidateDecision] (revised) |      same retry/timeout pattern
        +---------------------------------------+      as Module 4's Ollama stage
                          |
                          v
        +---------------------------------------+
        |   ConflictResolver.resolve(all)        |   <- priority ordering, mutual
        |   -> list[ResolvedDecision]            |      exclusion rules (e.g. an
        +---------------------------------------+      element can't be both KEEP
                          |                             and REMOVE)
                          v
        +---------------------------------------+
        |   DecisionValidator.validate(all)      |   <- schema + business-rule
        |   -> ValidationReport                  |      checks; blocks persistence
        +---------------------------------------+      on hard failures
                          |
                          v
        +---------------------------------------+
        |   ManifestAssembler.build(...)         |   <- groups into keep/remove/
        |   -> DecisionManifest                  |      replace/enhance/add +
        +---------------------------------------+      the umbrella manifest
                          |
                          v
        data/decisions/{video_id}/decision_manifest.json
        data/decisions/{video_id}/keep.json
        data/decisions/{video_id}/remove.json
        data/decisions/{video_id}/replace.json
        data/decisions/{video_id}/enhance.json
        data/decisions/{video_id}/add.json
        data/decisions/{video_id}/reasoning_trace.json   (traceability, Section 10.8)
        logs/module9_metrics.jsonl                        (per-run metrics, mirrors M7)
```

---

## 3. Folder Structure

Following the existing flat-module-plus-component-subpackage pattern (compare
`modules/vre_components/` for Module 6.5, `modules/vision_stack/` for the
vision stack):

```
thumbnail-ai/
├── modules/
│   ├── decision_engine.py                 # Public API: run_decision_engine(), orchestration
│   ├── decision_exceptions.py             # Module 9 exception hierarchy
│   └── decision_components/
│       ├── __init__.py
│       ├── interfaces.py                  # ABCs: IRuleEngine, ILLMReasoner,
│       │                                   #   IConflictResolver, IDecisionValidator,
│       │                                   #   IManifestAssembler, IDecisionCache
│       ├── io.py                          # load_input_bundle(), atomic manifest writers,
│       │                                   #   cache lookups (mirrors prompt_compiler.py's
│       │                                   #   load_cached_prompt_package pattern)
│       ├── rule_engine.py                 # RuleEngine: KEEP/REMOVE/REPLACE/ENHANCE/ADD
│       │                                   #   rule sets, one function per rule family
│       ├── rules/
│       │   ├── __init__.py
│       │   ├── keep_rules.py
│       │   ├── remove_rules.py
│       │   ├── replace_rules.py
│       │   ├── enhance_rules.py
│       │   └── add_rules.py
│       ├── ambiguity_router.py            # AmbiguityRouter: routes low-confidence /
│       │                                   #   conflicting candidates to the LLM stage
│       ├── llm_reasoner.py                # LLMReasoner: local Ollama call, same
│       │                                   #   requests + tenacity + format="json"
│       │                                   #   pattern as thumbnail_intelligence.py
│       ├── conflict_resolver.py           # ConflictResolver: priority ordering,
│       │                                   #   mutual-exclusion resolution
│       ├── confidence.py                  # Confidence scoring / calibration helpers
│       ├── validator.py                   # DecisionValidator: schema + business rules
│       ├── manifest_assembler.py          # ManifestAssembler: builds DecisionManifest
│       │                                   #   + the five per-action JSON files
│       └── metrics.py                     # MetricsCollector (mirrors image_generator.py)
├── modules/models.py                       # + new Module 9 Pydantic models (Section 5)
│                                            # + proposed AssetExtractionManifest (Section 0)
├── modules/config.py                       # + Module 9 constants (paths, thresholds,
│                                            #   priority table, Ollama settings)
├── data/
│   ├── asset_extractions/{video_id}.json   # Module 8 output (input to M9)
│   └── decisions/
│       └── {video_id}/
│           ├── decision_manifest.json
│           ├── keep.json
│           ├── remove.json
│           ├── replace.json
│           ├── enhance.json
│           ├── add.json
│           └── reasoning_trace.json
├── logs/
│   ├── module9.log
│   └── module9_metrics.jsonl
├── docs/
│   └── MODULE9_AI_DECISION_ENGINE_ARCHITECTURE.md   # this document
└── tests/
    ├── test_decision_engine.py
    ├── test_rule_engine.py
    ├── test_ambiguity_router.py
    ├── test_llm_reasoner.py
    ├── test_conflict_resolver.py
    ├── test_decision_validator.py
    ├── test_manifest_assembler.py
    └── fixtures/
        └── decision_engine/
            ├── intelligence_sample.json
            ├── redesign_spec_sample.json
            ├── prompt_package_sample.json
            └── asset_extraction_sample.json
```

**Why `decision_components/` mirrors `vre_components/` rather than
`image_generator.py`'s single-file class layout:** Module 6.5's ABC +
component pattern is the repo's precedent for a stage that (a) has several
independently swappable sub-behaviors and (b) benefits from interface-based
testing with fakes. Module 9 has exactly that shape — the rule engine, LLM
reasoner, conflict resolver, and validator are each independently testable
and each has a plausible "swap the implementation later" story (e.g.
replacing rule-based KEEP/REMOVE with a learned classifier per Section 19). Module
7's single-file-with-many-classes style fits a tightly sequential pipeline
with one owning orchestrator (`ImageGeneratorPipeline`); Module 9's shape is
closer to VRE's.

---

## 4. Python Module Layout — Responsibilities at a Glance

| Module | Responsibility | Depends on |
|---|---|---|
| `decision_engine.py` | Public entry point; orchestrates the pipeline end-to-end for one `video_id`; owns caching/resume decision; writes metrics | All `decision_components/*` |
| `decision_exceptions.py` | Exception hierarchy (Section 16) | — |
| `decision_components/interfaces.py` | ABCs for every stage, enabling DI and test doubles | `models.py` |
| `decision_components/io.py` | Loads M4/M5/M6/M8 artifacts into a `DecisionInputBundle`; atomic JSON writers for all six output files; cache-hit lookup | `models.py`, `config.py` |
| `decision_components/rule_engine.py` | Runs all rule families, emits `CandidateDecision` list with rule-based confidence | `rules/*`, `models.py` |
| `decision_components/rules/*_rules.py` | One pure-function rule family per action type | `models.py` |
| `decision_components/ambiguity_router.py` | Decides which candidates need LLM adjudication (Section 10.3) | `config.py` (thresholds) |
| `decision_components/llm_reasoner.py` | Local Ollama call for ambiguous candidates only; strict JSON-schema-constrained output | `config.py` (Ollama settings) |
| `decision_components/conflict_resolver.py` | Applies priority ordering + mutual-exclusion rules (Section 11) | `config.py` (priority table) |
| `decision_components/confidence.py` | Confidence combination/calibration math (rule + LLM + resolution confidence → final) | — |
| `decision_components/validator.py` | Structural + business-rule validation before persistence (Section 10.6) | `models.py` |
| `decision_components/manifest_assembler.py` | Groups `ResolvedDecision`s into the six output artifacts | `models.py` |
| `decision_components/metrics.py` | Per-run metrics row, appended to `module9_metrics.jsonl` (mirrors `image_generator.py::MetricsCollector`) | `config.py` |

---

## 5. Data Models

All models follow the repo's Module 4-7 convention exactly: `pydantic.BaseModel`
subclasses, `model_config = ConfigDict(frozen=True)`, a `video_id` non-empty
validator, `status: Literal[...]`, `error_message: Optional[str]`,
`duration_seconds: float`, and an ISO-8601 UTC `*_at` timestamp. They belong
in `modules/models.py` under a new `# Module 9 - AI Decision Engine` section,
plus one proposed model for the missing Module 8 contract (clearly marked).

### 5.0 Proposed Module 8 contract (assumption - see Sections 0 and 20)

```python
class ExtractedAsset(BaseModel):
    """One element Module 8 located inside the Module 7 generated image."""
    model_config = ConfigDict(frozen=True)

    asset_id: str                       # stable id, e.g. "obj_0", "text_1", "face_0"
    asset_type: Literal["face", "object", "text", "background", "logo", "region"]
    label: str                          # e.g. "hoodie", "creator face", "sky"
    bbox: BoundingBox                   # reuse existing normalized BoundingBox model
    extraction_confidence: float        # [0.0, 1.0] - Module 8's own confidence
    checksum: str                       # sha256 of the cropped asset bytes, if persisted
    source: Literal["ocr", "detection", "face", "segmentation", "manual"]
    linked_source_element: Optional[str] = None  # id/label this traces back to
                                                    # in Module 4's original analysis

class AssetExtractionManifest(BaseModel):
    """Output of Module 8. Input to Module 9."""
    model_config = ConfigDict(frozen=True)

    video_id: str
    generated_image_path: str
    generated_image_hash: str           # sha256, links to Module 7's GeneratedAsset
    extracted_assets: list[ExtractedAsset] = []
    status: Literal["success", "partial", "error"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    extracted_at: str
```

### 5.1 Module 9 core models

```python
class DecisionAction(str, Enum):
    KEEP = "keep"
    REMOVE = "remove"
    REPLACE = "replace"
    ENHANCE = "enhance"
    ADD = "add"

class DecisionSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    RULE_LLM_AGREEMENT = "rule_llm_agreement"   # both agreed independently
    CONFLICT_RESOLUTION = "conflict_resolution" # settled by priority ordering

class TargetElement(BaseModel):
    """What a decision refers to. Not every action targets an existing asset --
    ADD decisions target a *proposed* new element with no source asset."""
    model_config = ConfigDict(frozen=True)

    element_id: str                     # asset_id for existing elements, or a
                                          # synthetic id ("add_arrow_0") for ADD
    element_type: str                   # "face" | "object" | "text" | "background"
                                          # | "lighting" | "effect" | ...
    label: str
    bbox: Optional[BoundingBox] = None  # None for non-spatial actions (e.g. global
                                          # ENHANCE of saturation) and for unplaced ADDs

class CandidateDecision(BaseModel):
    """A not-yet-finalized decision, before conflict resolution."""
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    target: TargetElement
    action: DecisionAction
    confidence: float                   # [0.0, 1.0]
    source: DecisionSource
    rationale: str                      # human-readable
    rule_ids: list[str] = []            # which rule(s) fired, empty if LLM-only
    llm_raw_response_ref: Optional[str] = None  # pointer into reasoning_trace.json

class ResolvedDecision(BaseModel):
    """A finalized decision, after conflict resolution + validation."""
    model_config = ConfigDict(frozen=True)

    decision_id: str
    target: TargetElement
    action: DecisionAction
    confidence: float
    source: DecisionSource
    rationale: str
    priority_rank: int                  # lower = higher priority; see Section 11
    superseded_candidate_ids: list[str] = []  # candidates this one beat out
    machine_reasoning: dict[str, Any] = {}    # structured: {"rule_ids": [...],
                                                # "thresholds_used": {...}, ...}

class DecisionManifest(BaseModel):
    """Output of Module 9. Umbrella manifest; the five per-action files are a
    grouped projection of this same data (see Section 9.6)."""
    model_config = ConfigDict(frozen=True)

    video_id: str
    source_generated_image_path: str
    source_generated_image_hash: str
    decisions: list[ResolvedDecision] = []
    keep_count: int = 0
    remove_count: int = 0
    replace_count: int = 0
    enhance_count: int = 0
    add_count: int = 0
    overall_confidence: float = 0.0     # aggregate, see Section 10.5
    conflicts_resolved: int = 0
    llm_adjudications: int = 0
    status: Literal["success", "partial", "error"] = "success"
    partial_failure_reasons: list[str] = []
    error_message: Optional[str] = None
    total_duration_seconds: float = 0.0
    decided_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()

class ReasoningTraceEntry(BaseModel):
    """One row of the machine-readable audit trail (Section 10.8)."""
    model_config = ConfigDict(frozen=True)

    decision_id: str
    stage: Literal["rule_engine", "ambiguity_router", "llm_reasoner",
                    "conflict_resolver", "validator"]
    input_snapshot: dict[str, Any]
    output_snapshot: dict[str, Any]
    timestamp: str
```

**Design notes:**
- `CandidateDecision` vs `ResolvedDecision` is a deliberate two-stage split --
  it is what makes conflict resolution and validation testable in isolation
  (you can unit-test `ConflictResolver.resolve()` against a hand-built list
  of candidates without running the rule engine or the LLM at all).
- `TargetElement.bbox` is `Optional` because `ENHANCE` and some `ADD`
  decisions are not spatially anchored to one asset (e.g. "enhance global
  saturation +0.1" or "add a subtle vignette").
- Every model that can fail carries the same `status` / `error_message` /
  `duration_seconds` shape Modules 4-7 use, so Module 9 fits the existing
  main-pipeline error-handling code in `main.py` without special-casing.

---

## 6. Public API

Following the pattern where each module exposes a small number of top-level
functions from its main file (`compile_prompt_package()` in
`prompt_compiler.py`, `generate_redesign_specification()` in
`redesign_spec_engine.py`), `modules/decision_engine.py` exposes:

```python
def run_decision_engine(
    video_id: str,
    *,
    force_recompute: bool = False,
    decision_dir: Path = DEFAULT_DECISION_DIR,
) -> DecisionManifest:
    """Full pipeline for one video_id: load inputs, decide, resolve, validate,
    persist. Cache-aware (see Section 12) unless force_recompute=True.
    Never raises for expected per-video failures; returns a manifest with
    status="error" and a populated error_message instead (same contract as
    Module 4/5/6). Raises only DecisionEngineError subclasses for
    programmer-error conditions (e.g. malformed config)."""

def run_decision_engine_batch(
    video_ids: list[str],
    *,
    force_recompute: bool = False,
    decision_dir: Path = DEFAULT_DECISION_DIR,
) -> list[DecisionManifest]:
    """Batch entry point for main.py. Processes videos independently;
    one video's failure never aborts the batch (mirrors main.py's existing
    per-creator isolation pattern)."""

def load_cached_decision_manifest(
    video_id: str,
    decision_dir: Path = DEFAULT_DECISION_DIR,
) -> Optional[DecisionManifest]:
    """Cache lookup only, no computation. Mirrors
    prompt_compiler.load_cached_prompt_package(). Used by Module 10/11 to
    consume Module 9's output without importing decision_engine's internals."""
```

`main.py`'s pipeline orchestration calls `run_decision_engine_batch()` for a
CSV's worth of creators exactly the way it currently calls Module 5's and
Module 6's batch equivalents (see Section 17).

---

## 7. Internal API (component contracts)

Defined in `decision_components/interfaces.py` as `ABC`s, matching
`vre_components/interfaces.py`'s style. Each concrete class in Section 4
implements exactly one of these:

```python
class IRuleEngine(ABC):
    @abstractmethod
    def evaluate(self, bundle: DecisionInputBundle) -> list[CandidateDecision]: ...

class IAmbiguityRouter(ABC):
    @abstractmethod
    def select(
        self, candidates: list[CandidateDecision]
    ) -> tuple[list[CandidateDecision], list[CandidateDecision]]:
        """Returns (confident, needs_llm_review)."""

class ILLMReasoner(ABC):
    @abstractmethod
    def adjudicate(
        self, candidates: list[CandidateDecision], bundle: DecisionInputBundle
    ) -> list[CandidateDecision]:
        """Returns revised candidates for the ambiguous subset only."""

class IConflictResolver(ABC):
    @abstractmethod
    def resolve(self, candidates: list[CandidateDecision]) -> list[ResolvedDecision]: ...

class IDecisionValidator(ABC):
    @abstractmethod
    def validate(self, decisions: list[ResolvedDecision]) -> ValidationReport: ...

class IManifestAssembler(ABC):
    @abstractmethod
    def build(
        self,
        video_id: str,
        bundle: DecisionInputBundle,
        decisions: list[ResolvedDecision],
        validation: ValidationReport,
    ) -> DecisionManifest: ...

class IDecisionCache(ABC):
    @abstractmethod
    def load(self, video_id: str) -> Optional[DecisionManifest]: ...
    @abstractmethod
    def save(self, manifest: DecisionManifest) -> None: ...
```

Interfaces exist for two concrete reasons already established by VRE's
precedent, not novelty for its own sake: (1) `LLMReasoner` needs a fake in
every test that isn't specifically testing LLM adjudication -- Ollama must
never be a hard dependency of `RuleEngine` or `ConflictResolver` tests; (2)
Section 19's roadmap explicitly plans a future swap of `RuleEngine` for a
learned model, and that swap should be a one-line change in
`decision_engine.py`'s wiring, not a rewrite.

---

## 8. Class Responsibilities

| Class | Responsibility | Notes |
|---|---|---|
| `DecisionEngine` (in `decision_engine.py`; not exported, backs the two public functions) | Owns the orchestration sequence: load -> rule-evaluate -> route -> adjudicate -> resolve -> validate -> assemble -> persist -> record metrics. Owns cache-hit short-circuiting. | Analogous to `ImageGeneratorPipeline` |
| `DecisionInputBundle` (dataclass in `io.py`) | Immutable holder for the four loaded inputs (`ThumbnailIntelligence`, `RedesignSpecification`, `PromptPackage`, `AssetExtractionManifest`) plus a computed cross-reference index (asset_id -> which of the four sources mention it) | Not a Pydantic model -- pure in-process value object, never serialized |
| `RuleEngine` | Runs every rule family in `rules/*_rules.py` against the bundle, collects all `CandidateDecision`s, tags each with rule-derived confidence | Implements `IRuleEngine` |
| `keep_rules.py` / `remove_rules.py` / `replace_rules.py` / `enhance_rules.py` / `add_rules.py` | Each exposes one or more pure functions `(bundle) -> list[CandidateDecision]`; no shared mutable state, no I/O | Kept as functions, not classes, matching `redesign_spec_engine.py`'s functional style for deterministic derivations |
| `AmbiguityRouter` | Splits candidates into "confident enough to finalize" vs "needs LLM" using the thresholds in `config.py` (Section 10.3) | Implements `IAmbiguityRouter` |
| `LLMReasoner` | Formats ambiguous candidates + relevant bundle context into a structured prompt, calls local Ollama with `format="json"`, parses/validates the response, retries via `tenacity` on transient failures -- same pattern as `thumbnail_intelligence.py::_call_ollama_api` | Implements `ILLMReasoner` |
| `ConflictResolver` | Applies the priority table and mutual-exclusion rules (Section 11) to merge rule-only, LLM-only, and rule+LLM candidates into one `ResolvedDecision` per target element | Implements `IConflictResolver` |
| `ConfidenceCombiner` (in `confidence.py`) | Pure functions for combining rule confidence + LLM confidence + resolution-adjustment into one final score; also computes `DecisionManifest.overall_confidence` | No class needed; stateless helpers |
| `DecisionValidator` | Structural checks (every decision has a valid target, no orphaned `element_id`s) and business-rule checks (no element both KEPT and REMOVED, ADD decisions have non-empty label) | Implements `IDecisionValidator` |
| `ManifestAssembler` | Builds `DecisionManifest` and groups `ResolvedDecision`s by `action` into the five per-action file payloads | Implements `IManifestAssembler` |
| `DecisionCache` (in `io.py`) | File-existence-based cache exactly like `load_cached_prompt_package()` / `load_cached_redesign_specification()` -- keyed on `video_id` + `generated_image_hash` | Implements `IDecisionCache` |
| `MetricsCollector` (in `metrics.py`) | Appends one JSONL row per run to `logs/module9_metrics.jsonl`: timings per stage, candidate counts, LLM call count/latency, cache hit/miss | Mirrors `image_generator.py::MetricsCollector` |

---

## 9. Decision Pipeline

### 9.1 Ingestion & normalization (`io.py::load_input_bundle`)

Loads all four artifacts for a `video_id`. Missing or unparseable Module 8
output is *not* fatal to the whole pipeline -- Module 9 degrades to
`status="partial"` and proceeds using only Modules 4-6 (it can still make
KEEP/REMOVE/REPLACE/ENHANCE calls from Module 4's `DetectedObject` list and
Module 5's `ObjectDirective`s; it just loses Module 8's *generated-image*
grounding for those decisions, which is recorded in
`partial_failure_reasons`). This is the single point where the "proposed
Module 8 contract" assumption (Section 0) is isolated -- if the real Module 8
manifest differs, only this loader's deserialization changes.

Cross-references are built here once: every `ExtractedAsset.label` is fuzzy-
matched (simple normalized-string containment, no ML) against Module 4's
`DetectedObject.label`s, `FaceAnalysis` regions, and `OCRResult.text_regions`,
and against Module 5's `ObjectDirective.label`s and `elements_to_preserve`.
This produces the `element_id -> source_context` index every rule function
reads from `DecisionInputBundle`.

### 9.2 Rule-based decision layer (`rule_engine.py` + `rules/*`)

Each rule family is a small, named, independently testable function. Example
rule shapes (illustrative, not exhaustive -- full rule catalogue is an
implementation-time deliverable per Section 19):

- **KEEP**: an extracted asset whose `label` matches an entry in
  `RedesignSpecification.elements_to_preserve`, or whose matched
  `ObjectDirective.action == "preserve"`, or a face region when
  `FaceAnalysis` reports a primary creator face -> `KEEP`, high confidence.
- **REMOVE**: an extracted asset matched to an `ObjectDirective.action ==
  "remove"`, or an OCR text region whose `TextRegion` was flagged low-value
  by Module 4's reasoning (`weaknesses` mentioning clutter/text) and not
  covered by `TextOverlaySpec.include_text` -> `REMOVE`.
- **REPLACE**: background/sky-labeled assets when
  `RedesignSpecification.color_direction.warm_or_cool` implies a temperature
  flip incompatible with the current background, or any `ObjectDirective`
  the spec didn't mark `preserve`/`remove` but Module 4 flagged as a
  weakness -> `REPLACE`.
- **ENHANCE**: global, non-asset-anchored candidates derived directly from
  `ColorDirection` deltas (current `ColorProfile.brightness/contrast/
  saturation` vs `target_*`) whenever the delta exceeds a configured
  minimum-perceptible threshold -> `ENHANCE` (lighting/contrast/saturation).
  Face region ENHANCE (eyes, shadows) triggered when `FaceAnalysis` quality
  fields fall below a configured floor.
- **ADD**: derived from `GeminiReasoning.redesign_recommendations` /
  `RedesignSpecification.overall_rationale` text matched against a
  configured keyword taxonomy (arrow, glow, particle, emoji, money, effect)
  *only* as a low/medium-confidence candidate -- ADD is the action family
  most likely to need LLM adjudication because "what to add" is the most
  genuinely creative call, and the brief explicitly lists ADD examples
  (arrows, glow, particles, money, emoji, effects) that a keyword-anchored
  rule can only propose, not fully justify.

Every rule function returns `CandidateDecision`s with `source=RULE`,
populated `rule_ids`, and a confidence computed from how directly the
supporting data supports the action (Section 10.1).

### 9.3 Ambiguity routing (`ambiguity_router.py`)

A candidate is routed to the LLM stage if **any** of:
- `confidence < AMBIGUITY_CONFIDENCE_THRESHOLD` (config default `0.65`).
- Two or more rule-based candidates target the *same* `element_id` with
  *different* actions (a same-element conflict rules alone couldn't settle).
- The candidate's action is `ADD` and it has no `TargetElement.bbox` (i.e.
  no rule could anchor *where*) -- placement is deferred to the LLM, which
  reasons over `LayoutDirection.focal_zone` and the extracted-asset bboxes
  to avoid occlusion.

Everything else passes straight to conflict resolution untouched. This keeps
the LLM's call volume proportional to genuine ambiguity, not to every
decision -- consistent with the "deterministic-first" stance in Section 1.

### 9.4 LLM reasoning layer (`llm_reasoner.py`)

Reuses the exact integration shape of `thumbnail_intelligence.py`'s Ollama
stage: `requests.post` to `f"{OLLAMA_BASE_URL}/api/chat"`, a fixed system
prompt (`_MODULE9_OLLAMA_SYSTEM_PROMPT`) that pins the response schema,
`format="json"` to constrain output, wrapped in the same
`tenacity.retry(stop=stop_after_attempt(...), wait=wait_exponential(...))`
decorator pattern, with connection/timeout/HTTP errors mapped to typed
exceptions (Section 15) exactly as `_call_ollama_api` does. A dedicated
model constant `MODULE9_OLLAMA_MODEL` in `config.py` (may reuse
`OLLAMA_MODEL` or be distinct if a smaller/faster model suffices for
adjudication-only workloads -- an implementation-time tuning decision, not
an architectural one).

The prompt package sent to Ollama includes: the ambiguous candidates
(action, target label, current rule confidence, rationale), the relevant
slice of bundle context (not the whole bundle -- only what that candidate's
rules cited, to keep prompts small and outputs auditable), and an explicit
instruction to return, per candidate, a `{decision, confidence, rationale}`
triple plus (`for ADD only`) a proposed `bbox`. The LLM **never introduces a
brand-new target element outside the candidate set** it was given, except
when explicitly asked to propose ADD placement -- this is the guardrail that
keeps the LLM from silently inventing decisions the rule layer never
surfaced, and it is enforced by `DecisionValidator` (Section 9.5), not just
by prompting.

Raw Ollama responses are persisted to `reasoning_trace.json` (Section 10.8)
before parsing, the same debug-dump discipline `_dump_raw_ollama_response`
already uses for Module 4.

### 9.5 Conflict resolution (`conflict_resolver.py`) -- see Section 11 for the
full priority table and mutual-exclusion rules.

### 9.6 Validation (`validator.py`)

Runs after conflict resolution, before persistence:
- **Structural**: every `ResolvedDecision.target.element_id` referenced by a
  non-ADD action must exist in the input bundle's cross-reference index;
  every `ADD` decision must have a non-empty `label` and, if it has a
  `bbox`, that bbox must be within `[0,1]` normalized bounds.
- **Business-rule**: no `element_id` appears with two mutually exclusive
  actions in the final resolved set (this should be structurally impossible
  after Section 11's resolution, so the validator's job here is a defensive
  double-check, not primary enforcement).
- **Coverage**: warns (does not block) if an element Module 4 marked as a
  face/logo has *no* decision at all -- every KEEP-worthy element should get
  an explicit KEEP, not silent omission, so downstream modules never have to
  guess "no decision" means "keep."

Failures are split into `hard` (block persistence, pipeline returns
`status="error"`) and `soft` (logged as `partial_failure_reasons`,
`status="partial"`, persistence proceeds). Hard failures are reserved for
genuinely unusable output (e.g. a decision referencing a nonexistent
element_id that would corrupt Module 10's input); soft failures are things
like the coverage warning above.

### 9.7 Manifest assembly & persistence (`manifest_assembler.py` + `io.py`)

Builds `DecisionManifest`, then partitions `decisions` by `action` into five
lists and writes seven files total (the umbrella manifest, five per-action
files, plus `reasoning_trace.json`) atomically -- write to a `.tmp` sibling
path then `os.replace()`, exactly matching the write pattern already used in
`redesign_spec_engine.py`/`prompt_compiler.py`'s save functions.

---

## 10. Confidence Scoring System

### 10.1 Rule-based confidence

Each rule function assigns confidence from a small fixed vocabulary mapped
to numeric bands in `config.py`, not ad-hoc floats scattered through rule
code:

| Band | Range | When a rule uses it |
|---|---|---|
| `STRONG` | `0.85 - 1.0` | Direct, unambiguous match (e.g. `ObjectDirective.action` explicitly says `remove`) |
| `MODERATE` | `0.6 - 0.84` | Inferred from a threshold crossing (e.g. saturation delta exceeds the perceptible minimum) |
| `WEAK` | `0.35 - 0.59` | Keyword/heuristic match with no structural confirmation (e.g. ADD candidates from recommendation-text keyword matching) |

Exact scalar within a band is a deterministic function of how far the
underlying metric is from its threshold (e.g. a saturation delta of 0.3
against a 0.1 threshold scores near the top of `MODERATE`; 0.11 scores near
the bottom) -- so confidence is reproducible and unit-testable, never
randomized.

### 10.2 LLM confidence

The LLM is prompted to self-report a confidence per adjudicated candidate,
but that self-report is **never taken at face value** -- `confidence.py`
recalibrates it: `final_llm_confidence = min(llm_reported_confidence,
LLM_CONFIDENCE_CEILING)` (default ceiling `0.9`, configurable), reflecting
that this repo already treats LLM output as advisory (see Module 4's
`status="partial"` handling when Ollama degrades) rather than authoritative.

### 10.3 Ambiguity threshold

`AMBIGUITY_CONFIDENCE_THRESHOLD` (default `0.65`) lives in `config.py`
alongside every other cross-module threshold constant (mirroring
`CLUTTER_HIGH_THRESHOLD`, `MIN_SUBJECT_AREA_RATIO`, etc. already there for
Module 5). Tunable without touching code.

### 10.4 Combined confidence (post rule+LLM agreement)

When a candidate reaches conflict resolution having been through both rule
and LLM stages and they *agree* on the action, `ConfidenceCombiner` boosts
confidence: `combined = 1 - (1 - rule_conf) * (1 - llm_conf)` (standard
independent-evidence combination), capped at `0.98` (never let automated
scoring claim full certainty). When they *disagree*, resolution (Section 11)
decides which one wins, and the losing side's confidence is not blended in.

### 10.5 Manifest-level `overall_confidence`

`DecisionManifest.overall_confidence` is the count-weighted mean of every
`ResolvedDecision.confidence`, down-weighted by a penalty per unresolved
soft-validation warning (`overall_confidence *= (1 - 0.05 *
soft_warning_count)`, floored at `0.0`). This gives Module 10/11 a single
cheap signal for "how much should I trust this manifest wholesale" without
having to inspect every decision.

### 10.6 Decision validation (cross-reference to Section 9.6)

Validation is confidence-aware: a hard-failing decision is *excluded* from
the persisted manifest rather than persisted with a warning label, keeping
the invariant "everything in `decision_manifest.json` is something Module 10
can act on directly."

### 10.7 Human-readable vs machine-readable reasoning

Every `ResolvedDecision.rationale` is a plain-English sentence (e.g. "Kept:
matches `elements_to_preserve` entry 'creator face' from Module 5, and
Module 4 flagged this as the primary face region.") suitable for a human
reviewing `decision_manifest.json` directly. `ResolvedDecision.
machine_reasoning` is the structured counterpart: `{"rule_ids": [...],
"thresholds_used": {...}, "source_confidences": {"rule": 0.9, "llm": null},
"conflict_beaten": [...]}, ` designed for Module 10/11 or a future
learning system (Section 19) to consume programmatically without string
parsing.

### 10.8 Decision traceability (`reasoning_trace.json`)

A flat, append-only list of `ReasoningTraceEntry` rows, one per stage each
decision passed through, written alongside the five manifest files. This is
the audit log that answers "why did the system decide X" at a stage-by-stage
granularity the terse `rationale` string can't fully capture -- e.g. the
exact ambiguity-router inputs that triggered LLM routing, or the raw Ollama
response before parsing. Not consumed by Module 10/11; purely for debugging,
QA, and future training-data mining (Section 19).

---

## 11. Decision Conflict Resolution

Conflicts arise in two shapes:

**A. Same-element, cross-source disagreement** -- e.g. a rule says `KEEP`
the background (matched `elements_to_preserve`), but a different rule says
`REPLACE` it (matched a Module 4 weakness). Resolved by:

1. **Priority ordering** (`config.py::DECISION_PRIORITY_ORDER`, a fixed
   list, highest first): `KEEP` (explicit preserve directive) > `REMOVE`
   (explicit remove directive) > user/creator-face protection rules (never
   overridden) > `REPLACE` > `ENHANCE` > `ADD`. Explicit directives from
   Module 5 (`ObjectDirective.action`) always outrank inferred/keyword-based
   candidates regardless of numeric confidence -- an explicit `preserve`
   beats an inferred `replace` even if the replace candidate scored higher,
   because Module 5 already did deterministic reasoning over Module 4's
   output and re-litigating it would contradict the pipeline's own upstream
   decision.
2. If priority ordering doesn't separate them (same tier), **higher combined
   confidence wins** (Section 10.4).
3. If still tied, the candidate with `source=LLM` wins over `source=RULE`
   only when the LLM was specifically routed to adjudicate *that* conflict
   (Section 9.3's second routing condition) -- since in that case the LLM
   saw both options and chose; a rule that never saw the conflict shouldn't
   out-rank the adjudicator that did.

The losing candidate(s) become `superseded_candidate_ids` on the winning
`ResolvedDecision`, preserved for traceability, not discarded silently.

**B. Structural mutual exclusion** -- enforced regardless of confidence:
- An element cannot be both `KEEP` and any of `REMOVE`/`REPLACE`.
- An element cannot be both `REMOVE` and `ENHANCE` (nothing to enhance if
  it's gone).
- Multiple `ADD` candidates proposing near-identical bbox + label
  (IoU > `ADD_DEDUP_IOU_THRESHOLD`, default `0.7`) are merged into one,
  keeping the higher-confidence rationale and recording the merge in
  `machine_reasoning`.

`ConflictResolver.resolve()` is implemented as a pure function over the full
candidate list (group by `element_id`, apply A within each group, then apply
B across the whole resolved set) -- no per-element mutable state, which
keeps it trivially unit-testable with hand-built candidate lists (Section
16).

---

## 12. Caching Strategy

Identical philosophy to Modules 5/6: **the persisted output file *is* the
cache.** `data/decisions/{video_id}/decision_manifest.json` existing and
readable is a cache hit; `DecisionEngine` calls
`io.load_cached_decision_manifest(video_id)` first and returns it unless
`force_recompute=True`.

One refinement specific to Module 9: cache validity additionally checks that
`DecisionManifest.source_generated_image_hash` matches the *current*
Module 7 `GeneratedAsset`'s hash for that `video_id` (loaded cheaply from
`data/generated_thumbnails` metadata) -- a stale decision manifest from a
prior generation run must not be silently reused against a new image. If the
hash differs, it's treated as a cache miss (recompute), not an error. This
mirrors the intent (though not the literal mechanism) of Module 6.5's
`VRE_CACHE_ENABLED` + source-hash-keyed caching already in `config.py`.

No separate cache store (Redis, sqlite, etc.) is introduced -- consistent
with "local-first execution" and every other module's file-based approach.

---

## 13. Resume Strategy

Two levels, matching the repo's existing granularity:

- **Cross-video resume** (batch level): `run_decision_engine_batch()`
  iterates `video_ids`, and for each one, cache-hit short-circuiting
  (Section 12) means an interrupted batch simply picks up where it left off
  on the next run -- already-decided videos are skipped, not recomputed.
  Same behavior `main.py` already relies on for Modules 3-7.
- **Within-video resume** (pipeline-stage level): `DecisionEngine` writes
  `reasoning_trace.json` incrementally, one stage at a time, rather than
  only at the end. If the process is killed mid-run (e.g. Ollama hangs),
  the next invocation can detect a partial trace file and skip re-running
  the rule engine (deterministic, cheap to recompute anyway, so this is a
  minor optimization) but *always* re-runs the LLM stage from scratch
  (the expensive, non-idempotent-feeling step) rather than trying to
  resume a half-received Ollama response. This keeps resume logic simple:
  the only genuinely resumable unit of work is "have I finished this video
  at all," not sub-stage checkpointing, since rule evaluation is fast
  (milliseconds) and only the LLM call is worth protecting from redundant
  re-execution -- which caching (Section 12) already does at the
  whole-video granularity.

Incremental execution (batch grows over time) is naturally supported: adding
new `video_id`s to a batch only computes the new ones, since existing
manifests remain valid cache hits.

---

## 14. Logging

Exactly matches Module 5's `_configure_logger()` pattern:

```python
_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"

def _configure_logger() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE9_LOG_PATH),      # logs/module9.log
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )
```

Called once at module import time in `decision_engine.py`, same as every
other module. `loguru` throughout (already the project's sole logging
library -- no new dependency).

**Log levels used:**
- `DEBUG`: per-rule evaluation results, per-candidate confidence math,
  cache hit/miss detail.
- `INFO`: per-video pipeline start/end with duration, decision counts by
  action, cache hits.
- `WARNING`: soft validation failures, LLM confidence recalibration events,
  ADD-candidate deduplication.
- `ERROR`: hard validation failures, Ollama connection/timeout failures
  after retries exhausted, malformed input artifacts.

Metrics (`module9_metrics.jsonl`) are a separate structured stream from
logs, mirroring Module 7's `MODULE7_METRICS_PATH` split between
human-oriented logs and machine-oriented metrics.

---

## 15. Exception Hierarchy

`modules/decision_exceptions.py`, following the flat-base-plus-typed-leaves
shape of `module7_exceptions.py` and `vre_exceptions.py`:

```python
class DecisionEngineError(Exception):
    """Base exception for every Module 9 failure."""

class InputBundleError(DecisionEngineError):
    """Base for failures loading/validating M4/M5/M6/M8 inputs."""

class MissingArtifactError(InputBundleError):
    """Raised when a required upstream artifact file does not exist."""

class ArtifactValidationError(InputBundleError):
    """Raised when an upstream artifact fails Pydantic validation."""

class AssetExtractionManifestError(InputBundleError):
    """Raised when Module 8's manifest is malformed; distinct from
    MissingArtifactError since a missing M8 manifest degrades to
    status='partial' (Section 9.1) but a *malformed* one is a hard stop --
    we should not guess at corrupted extraction data."""

class RuleEvaluationError(DecisionEngineError):
    """Raised when a rule function raises unexpectedly (a bug, not a data
    problem -- data-shape issues should be handled inside the rule and
    surfaced as a low-confidence or absent candidate, not an exception)."""

class LLMReasoningError(DecisionEngineError):
    """Base for local Ollama adjudication failures."""

class OllamaConnectionError(LLMReasoningError):
    """Could not reach the local Ollama server."""

class OllamaTimeoutError(LLMReasoningError):
    """Ollama request exceeded its configured deadline."""

class OllamaResponseParseError(LLMReasoningError):
    """Ollama's JSON response could not be parsed or failed schema
    validation after configured retries."""

class ConflictResolutionError(DecisionEngineError):
    """Raised when conflict resolution cannot converge (e.g. priority table
    misconfiguration produces a genuine cycle -- should never happen with a
    correctly configured DECISION_PRIORITY_ORDER, but guarded against)."""

class DecisionValidationError(DecisionEngineError):
    """Raised for a hard validation failure that blocks persistence."""

class ManifestPersistError(DecisionEngineError):
    """Raised when the decision manifest or per-action files cannot be
    atomically written."""

class DecisionCacheError(DecisionEngineError):
    """Raised when the decision cache cannot be read or written."""
```

`OllamaTransientError`-style transient/retry classification (see Module 4's
`_OllamaTransientError`) is reused conceptually: `llm_reasoner.py`'s
`tenacity` retry decorator only retries `OllamaConnectionError` and
`OllamaTimeoutError`; `OllamaResponseParseError` is not retried (a schema
violation on well-formed JSON is a prompt/parsing bug, not a transient
condition, and retrying it wastes a GPU-bound local inference call for no
benefit).

Per Module 4/5/6's established contract, `DecisionEngineError` subclasses
are **only raised for programmer-error / infrastructure conditions**.
Ordinary per-video failures (a bad upstream artifact for one creator, an
Ollama hiccup that exhausts retries) are caught inside
`run_decision_engine()` and converted into a `DecisionManifest` with
`status="error"` and a populated `error_message`, so one video's failure
never raises out of a batch run.

---

## 16. Testing Strategy

Mirrors the existing `tests/test_*.py` flat layout, one file per module/
component, using `pytest` (already the project's framework per `pytest.ini`).

| Test file | Focus |
|---|---|
| `test_decision_engine.py` | End-to-end orchestration with all dependencies faked (fake `IRuleEngine`, fake `ILLMReasoner`, etc.) -- verifies sequencing, cache short-circuiting, error-to-status conversion, batch isolation |
| `test_rule_engine.py` | Every rule family in `rules/*_rules.py` against hand-built `DecisionInputBundle` fixtures covering: explicit preserve/remove directives, threshold-crossing enhance triggers, keyword-based add triggers, and the "no rule fires" empty case |
| `test_ambiguity_router.py` | Threshold boundary behavior, same-element-conflict routing, unplaced-ADD routing |
| `test_llm_reasoner.py` | HTTP layer mocked (`responses` or `requests_mock`, matching how `test_thumbnail_intelligence.py` already mocks Ollama calls) -- retry behavior, timeout mapping, malformed-JSON handling, confidence recalibration ceiling |
| `test_conflict_resolver.py` | Priority ordering correctness (every pairwise tier comparison), mutual-exclusion enforcement, ADD deduplication via IoU, tie-breaking by combined confidence |
| `test_decision_validator.py` | Hard vs soft failure classification, orphaned-element_id detection, coverage warnings |
| `test_manifest_assembler.py` | Correct grouping into the five per-action files, count fields match `len(decisions)` per action, `overall_confidence` math |

**Fixtures** (`tests/fixtures/decision_engine/*.json`): hand-built minimal
valid instances of `ThumbnailIntelligence`, `RedesignSpecification`,
`PromptPackage`, and the proposed `AssetExtractionManifest`, small enough to
reason about by eye, covering both a "clean" case (unambiguous decisions,
no LLM needed) and an "ambiguous" case (forces LLM routing + conflict
resolution) so the full pipeline can be exercised without live model
inference in CI.

**Coverage bar**: match the repo's existing pattern of near-exhaustive
per-branch testing visible in `test_thumbnail_intelligence.py` and
`test_redesign_spec_engine.py` (large test files with one test per rule
branch and per error path) -- Module 9's decision logic is exactly the kind
of branchy, business-rule-heavy code that benefits most from that style.

**No live Ollama dependency in the standard test suite.** A small number of
integration tests (`tests/test_llm_reasoner_integration.py`, skipped by
default via a `pytest.mark.integration` marker, matching how a live-ComfyUI
integration path would be marked for Module 7) may exist for manual/CI-opt-in
verification against a real local Ollama instance.

---

## 17. Integration with Modules 1-8

- **Modules 1-3** (CSV Reader, YouTube Metadata, Thumbnail Downloader): no
  direct dependency. Module 9 only reads `video_id`-keyed artifacts already
  produced downstream of these.
- **Module 4** (`ThumbnailIntelligence`): primary source of *original*
  thumbnail structure (OCR regions, faces, objects, colors, composition,
  and the Gemini/Ollama `GeminiReasoning` narrative) that rule functions
  cross-reference against Module 8's *generated*-image assets.
- **Module 5** (`RedesignSpecification`): the single most load-bearing input
  -- `elements_to_preserve` and `object_directives` directly drive
  KEEP/REMOVE/REPLACE rules, and `color_direction`/`layout_direction` drive
  ENHANCE rules. Module 9 treats Module 5's deterministic directives as
  higher-priority than its own inferred candidates (Section 11), by design
  -- Module 9 does not re-litigate Module 5's already-deterministic
  reasoning, only fills the gaps Module 5 intentionally left open (ADD,
  fine-grained ENHANCE targets, generated-image-specific corrections).
- **Module 6** (`PromptPackage`): used as corroborating context for ADD/
  ENHANCE rationale (e.g. if `subject_instructions` or
  `typography_instructions` already asked for something, a matching
  ExtractedAsset gets a KEEP boost) and for `GenerationParameters` (width/
  height) needed to validate that ADD bboxes are sane.
- **Module 7** (Image Generation Engine): not read directly by Module 9,
  but its `GeneratedAsset` hash is what Module 8's manifest -- and hence
  Module 9's cache invalidation (Section 12) -- is keyed against.
- **Module 8** (Asset Extraction Engine): primary spatial/inventory input,
  per the proposed contract in Section 5.0. This is the one integration
  point flagged as provisional (Section 0, Section 20).
- **`main.py`**: Module 9 slots in as the next stage after Module 8 in the
  per-creator pipeline loop, called via `run_decision_engine_batch()`
  exactly where Module 5/6's batch calls currently sit, with the same
  per-video try/except-and-continue isolation `main.py` already applies
  around each module boundary.

## 18. Integration with Modules 10-11

- **Module 10 (Asset Composer)** is the primary consumer.
  `load_cached_decision_manifest(video_id)` gives it the umbrella manifest;
  it is expected to iterate `DecisionManifest.decisions` grouped by
  `action`, or read the five pre-grouped per-action files directly if it
  prefers not to filter itself -- both are kept in sync by
  `ManifestAssembler` so either consumption style works. `ResolvedDecision.
  target.bbox` gives Module 10 the spatial anchor it needs to actually
  perform composition; `machine_reasoning` gives it structured parameters
  (e.g. target color deltas for ENHANCE) without needing to re-derive them.
- **Module 11 (Final Generation Pipeline)** consumes `overall_confidence`
  and `status` as a gate -- a `status="error"` or low-`overall_confidence`
  manifest is a natural signal to skip or flag a creator for manual review
  rather than proceeding to final generation automatically. This is a
  policy decision for Module 11, not enforced by Module 9 itself (Module 9
  only produces the signal; it does not gate the pipeline).
- Both downstream modules should treat `DecisionManifest` and the five
  per-action files as **read-only, immutable inputs** (the `frozen=True`
  Pydantic config already enforces this in-process) -- any downstream
  correction belongs in Module 10/11's own logic, not by mutating Module
  9's output, keeping the audit trail in `reasoning_trace.json` trustworthy.

## 19. Implementation Roadmap

Split into agent-sized phases, each independently compilable and testable,
minimal blast radius, following the brief's requirement:

**Phase 0 -- Contracts.** Add `AssetExtractionManifest`/`ExtractedAsset`
(Section 5.0, pending reconciliation with real Module 8 once it exists) and
all Module 9 models (Section 5.1) to `modules/models.py`. Add Module 9
constants to `modules/config.py` (paths, thresholds, priority table, Ollama
settings). Deliverable: models import cleanly, round-trip JSON
serialize/deserialize in a small smoke test. No behavior yet.

**Phase 1 -- Ingestion.** `decision_components/io.py::load_input_bundle()` +
`DecisionInputBundle` + cross-reference index construction.
`decision_exceptions.py`'s `InputBundleError` family. Deliverable:
loads/validates all four artifact types from fixtures, handles missing
Module 8 gracefully (`status="partial"` path), full unit coverage of
malformed-input cases.

**Phase 2 -- Rule engine.** `rules/*_rules.py` + `rule_engine.py` +
`confidence.py`'s Section 10.1 band logic. No LLM, no conflict resolution
yet -- `RuleEngine.evaluate()` is independently runnable and testable end to
end against fixtures. Deliverable: given a fixture bundle, produces a
plausible, fully-rule-sourced candidate list with sane confidences.

**Phase 3 -- Ambiguity routing + conflict resolution (no LLM call yet).**
`ambiguity_router.py` + `conflict_resolver.py` with `LLMReasoner` mocked/
stubbed to a no-op passthrough. Deliverable: rule-only candidates flow
through to `ResolvedDecision`s correctly, priority ordering and mutual
exclusion fully tested per Section 11's rules, independent of Ollama being
available at all.

**Phase 4 -- LLM reasoning.** `llm_reasoner.py`, real Ollama integration,
`reasoning_trace.json` raw-response dumping, retry/timeout handling.
Deliverable: ambiguous candidates get adjudicated against a live local
Ollama instance (manual verification) and against mocked HTTP in the
standard suite; confidence recalibration ceiling enforced.

**Phase 5 -- Validation + manifest assembly + persistence.**
`validator.py`, `manifest_assembler.py`, atomic writers for all six output
files, `DecisionCache` (Section 12). Deliverable: full pipeline
`run_decision_engine()` produces a real `decision_manifest.json` +
per-action files + trace on disk for a fixture video, cache hit/miss
behavior verified.

**Phase 6 -- Orchestration, batching, metrics, main.py wiring.**
`decision_engine.py`'s public API, `run_decision_engine_batch()`,
`metrics.py`, logging setup, `main.py` integration at the correct pipeline
position. Deliverable: a full CSV batch run exercises Module 9 end to end
alongside the existing pipeline, with per-creator failure isolation
verified.

**Phase 7 (post-MVP, not blocking Module 10) -- Future learning
capability.** The brief asks the architecture to "support future learning
capability." This design supports it structurally without building it now:
`ReasoningTraceEntry` rows plus `ResolvedDecision.machine_reasoning` are
already shaped as labeled training examples (rule/LLM decision + eventual
downstream outcome, once Module 10/11 or human review feedback is wired
back). A future phase could add a `IRuleEngine`-conforming learned-model
implementation that consumes this accumulated trace data, swapped in via
the same interface `RuleEngine` implements today (Section 7) -- no other
component needs to change.

## 20. Risks and Open Questions

1. **Module 8 does not exist yet (highest-priority open item).** This
   entire design's ingestion boundary (Section 9.1, Section 5.0) rests on a
   *proposed* `AssetExtractionManifest` contract, not a verified one. Before
   Phase 1 implementation begins, the actual Module 8 architecture/output
   shape needs to be finalized and reconciled against Section 5.0 --
   ideally by whoever builds Module 8 reviewing this document's proposed
   contract, since Module 9's rule layer (Section 9.2) is written assuming
   `ExtractedAsset.label`/`bbox`/`asset_type` exist in roughly this shape.
2. **Cross-reference matching (Section 9.1) is intentionally simple
   (normalized string containment).** It's a real risk for silent
   mismatches (e.g. Module 4 says "man wearing hoodie," Module 8 extracts
   "hoodie" as a separate object) producing spurious KEEP/REMOVE conflicts.
   Mitigated by routing low-confidence matches to the LLM stage (which sees
   full label text and can reason about it) rather than trusting the string
   match alone, but this is a known weak point worth revisiting once real
   Module 8 output is available to test against.
3. **LLM model choice for `MODULE9_OLLAMA_MODEL`** (reuse Module 4's model
   vs. a smaller/faster one tuned for structured adjudication) is left
   open, to be settled empirically during Phase 4 against real RTX 4060
   VRAM budgets, especially if Module 4's model is still resident from an
   earlier pipeline stage in the same run.
4. **Priority-table stability.** `DECISION_PRIORITY_ORDER` (Section 11) is a
   single fixed list; as more rule families are added in Phase 2+, new
   action sub-types may need finer-grained tiers than the five top-level
   actions provide (e.g. should "remove because explicit directive" always
   beat "enhance because explicit directive" even when they don't target
   the same element and thus never actually conflict?). Current design
   only resolves *same-element* conflicts, so this risk is scoped, but
   worth flagging for review once the full rule catalogue exists.
5. **ADD placement quality** is the single most subjective decision family
   (arrows, glow, particles, money, emoji, effects) and the hardest to
   validate automatically beyond bbox-bounds checking (Section 9.6). Module
   10/11 should treat ADD decisions with the lowest baseline trust of the
   five action types until real output is reviewed against actual creator
   thumbnails.
6. **No feedback loop exists yet from Module 10/11 back to Module 9.**
   Phase 7's future-learning story depends on such a loop (e.g. "this ADD
   decision was manually reverted") eventually being defined; it's out of
   scope for Module 9 itself but worth flagging as a dependency for anyone
   designing Module 10/11's own architecture.


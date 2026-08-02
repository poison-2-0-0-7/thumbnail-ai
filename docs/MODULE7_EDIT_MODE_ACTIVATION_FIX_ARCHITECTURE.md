# Module 7 — Edit Mode Activation Fix Architecture

**Enhancement to the existing Module 7 V2 (Editing Engine) and PORCE (Pipeline Observability & Root Cause Engine).**
**Repository:** `poison-2-0-0-7/thumbnail-ai`
**Scope:** `modules/config.py`, `modules/image_generator.py` (wiring only, no stage redesign), `observability/diagnostics/rules/` (one new rule, reusing PORCE's existing engine). Module 7 V2's staged-edit stage implementations, `WorkflowBuilder`'s fragment-assembly mechanics, PORCE's trace/facts/reporting layers, and the rendering pipeline itself are treated as fixed, correctly-built, reused contracts.
**Status:** Architecture only. Zero implementation code, zero tests, zero repository modification.

---

## 0. Grounding — Reconciling the Investigation Document Against the Current `main` Branch

Per instructions, `docs/MODULE7_RENDERING_ROOT_CAUSE_INVESTIGATION_ARCHITECTURE.md` (delivered as an uploaded document, not present under `docs/` on `main`) is treated as the authoritative investigation and is not repeated here. Its confirmed finding — `PROFILE_STANDARD_EDIT` is excluded from `MODULE7_PROFILE_PREFERENCE`, so `edit_mode="auto"` can never resolve to `"staged_edit"` via VRAM-based profile auto-selection — is verified, byte-for-byte, against the current repository: `config.py` line 442 defines `PROFILE_STANDARD_EDIT` with `edit_mode_default="staged_edit"`; line 447's `MODULE7_PROFILE_PREFERENCE` tuple contains exactly `("PROFILE_PREMIUM", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")` — `PROFILE_STANDARD_EDIT` is absent. This document takes that finding as its starting premise.

**However, the repository has moved forward since the investigation was authored** (the multi-candidate/strategy-pack architecture — `MODULE7_PHASE4_MULTI_CANDIDATE_GENERATION_ARCHITECTURE.md` — has since landed and refactored `ImageGeneratorPipeline.run()`'s per-candidate body into a new `_process_single_candidate()` method). Re-tracing the exact code path the investigation traced, against `main` as it stands today, surfaces **two additional, currently-real gaps** that the config-level fix alone does not close. Both are reported here because an activation architecture that fixes only the config defect would not, in fact, activate staged editing — and this document's job is specifically to make activation work, not merely plausible:

- **Gap B — the resolved edit mode is never threaded past `run()`.** `ImageGeneratorPipeline.run()` (`image_generator.py:1208-1218`) still computes `effective_edit_mode` exactly as the investigation traced. But in the current, refactored code, `effective_edit_mode` is a local variable that is **never passed into `_process_single_candidate()`** (its parameter list, `image_generator.py:1445-1459`, has no `edit_mode`/`effective_edit_mode` field), and `_process_single_candidate()` in turn calls `self.workflow_library.resolve(niche, profile)` — **without** the `edit_mode` argument, at both of its call sites (the primary path and the VRAM-fallback retry path). `WorkflowLibrary.resolve()` (`workflow_library.py`) has an `edit_mode: str = "legacy_txt2img"` parameter and correctly selects `{niche}_edit.json`/`general_edit.json` when `edit_mode == "staged_edit"` — this part is real and correctly built, exactly as the investigation found — but it is called with its **default** argument every time, because nothing above it forwards the value `run()` already computed. This means: even if Gap A (the config ordering) is fixed today, and even if a caller explicitly passes `edit_mode="staged_edit"`, the workflow template actually resolved is still the legacy one, silently, with no exception — the same "completes successfully while producing unrelated images" symptom class the investigation describes, from a second independent cause.
- **Gap C — the Python-orchestrated staged-edit stage pipeline is constructed but never invoked.** `ImageGeneratorPipeline.__init__` (`image_generator.py:1160-1196`) constructs `region_plan_validator`, `base_latent_stage`, `masked_composite_stage`, `background_edit_stage`, `object_edit_stage`, `typography_stage`, and `harmonization_stage` — exactly the five-stage pipeline `docs/MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §9 designed (`RegionPlanValidator` → `BaseLatentStage` → `BackgroundEditStage`/`ObjectEditStage` → `MaskedCompositeStage` → `TypographyStage` → `HarmonizationStage`). A repository-wide search confirms none of these seven attributes is referenced anywhere outside `__init__` — no `self.region_plan_validator.classify(...)`, no `self.masked_composite_stage.composite(...)`, nothing. `_process_single_candidate()`'s actual generation flow is a single `WorkflowBuilder.build()` + single `client_obj.generate()` call, identical in shape whether `effective_edit_mode` is `"legacy_txt2img"` or `"staged_edit"` — the only thing `staged_edit` currently *could* change, once Gap B is fixed, is which base ComfyUI graph template and which conditioning fragments get assembled (a graph-level swap to `VAEEncode`-based latents plus mask/ControlNet/IPAdapter fragments). The paste-back-guarantee, per-region masked compositing, and identity-anchored staged sampling that `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §9 describes as the actual editing behavior are not wired to run at all today.

**This document scopes itself accordingly.** Per the brief's explicit constraints — "avoid redesigning Module 7," "require minimal code changes," "reuse existing Module 7 V2 stages" — this document architects the fix for **Gap A and Gap B**: making `staged_edit` genuinely *reachable*, end-to-end, through the graph-level template-and-fragment mechanism that already exists and is already correctly built (`WorkflowLibrary`'s `_edit.json` resolution, `WorkflowBuilder`'s fragment assembly, the `workflows/fragments/*.json` library). This alone converts the renderer from "always ignores the source image" to "conditions on the source image via VAE-encoded latents, masks, ControlNet, and IPAdapter when an edit-capable profile is active" — which directly addresses the reported symptom (unrelated images). **Gap C — actually invoking the seven-stage Python orchestration — is identified but explicitly deferred to §16 (Future Extensions)** rather than folded into this fix, because wiring seven new stage invocations into the candidate loop's control flow is a materially larger change than the brief's "minimal code changes" / "reuse, don't redesign" bar permits for what is fundamentally a *reachability* fix, not a rebuild of the editing engine's execution model. Reusing existing Module 7 V2 stages means treating them as they already are — constructed, tested in isolation (presumably; see §14), and **already scheduled for their own wiring step** — not silently expanding this document's scope to also perform that wiring.

---

## 1. Problem Statement

`staged_edit` — the ComfyUI-graph-level editing mode that conditions generation on the actual source thumbnail (`VAEEncode`, masks, ControlNet, IPAdapter, via `{niche}_edit.json` templates and `workflows/fragments/*.json`) — is architecturally complete but operationally unreachable. `main.py` already requests `edit_mode="auto"` (line 578); `ImageGeneratorPipeline.run()` already resolves `"auto"` against the selected profile's `edit_mode_default`; `PROFILE_STANDARD_EDIT` already declares `edit_mode_default="staged_edit"`. Despite this, every production run resolves to `"legacy_txt2img"`, for the two independent reasons in §0 (Gap A: the edit-capable profile is excluded from the auto-selection preference order; Gap B: even the resolved mode is never forwarded to the component that would act on it). The result is exactly the symptom the investigation documented: the pipeline completes successfully, raises nothing, and produces images with no structural relationship to the source thumbnail. This document's objective is narrow and specific: make `staged_edit` reachable in production, by the shortest correct path, without touching the rendering pipeline's internals, PORCE, or any already-correct Module 7 V2 stage.

---

## 2. Current Architecture

**Profile layer (`config.py`, `models.py`).** `GenerationProfile` (frozen Pydantic model) carries `edit_mode_default: Literal["legacy_txt2img", "staged_edit"] | None = None` — an explicit, typed, per-profile declaration of what "auto" means for that profile. Five profiles are configured in `MODULE7_GENERATION_PROFILES`: `PROFILE_PREMIUM`, `PROFILE_STANDARD`, `PROFILE_STANDARD_EDIT`, `PROFILE_FAST`, `PROFILE_LOW_VRAM`. Only `PROFILE_STANDARD_EDIT` sets `edit_mode_default="staged_edit"`; the other four leave it `None`. `MODULE7_PROFILE_PREFERENCE` is a separate, ordered `tuple[str, ...]` consulted only by `ProfileSelector.select()` for VRAM-based auto-selection; it currently omits `PROFILE_STANDARD_EDIT` entirely.

**Selection layer (`image_generator.py`, `ProfileSelector`).** `ProfileSelector.select(available_vram_gb, requested_profile)` is, and is designed to remain, **edit-mode-agnostic**. When `requested_profile != "auto"` (an explicit profile name, e.g. from `MODULE7_PROFILE` config or a direct caller), it returns that profile if it fits VRAM, else falls back to the preference-ordered walk. When `requested_profile == "auto"` (today's default, `MODULE7_PROFILE = "auto"`), it walks `MODULE7_PROFILE_PREFERENCE` in tuple order and returns the first profile that fits available VRAM, with `PROFILE_LOW_VRAM` as an unconditional final fallback. Nothing in this method inspects `edit_mode_default` — profile selection and edit-mode selection are, by design, two separate concerns that meet only at the point described next.

**Mode-resolution layer (`image_generator.py::ImageGeneratorPipeline.run()`).** After a profile is selected, `run()` computes `effective_edit_mode`: if the caller passed `edit_mode="auto"`, it becomes `getattr(profile, "edit_mode_default", "legacy_txt2img") or "legacy_txt2img"` — i.e., it inherits whatever the *already-selected* profile declares. If the caller passed an explicit `"legacy_txt2img"` or `"staged_edit"`, that value is used as-is, regardless of which profile was selected. This is the "profile encapsulates the decision" pattern `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` describes: no separate top-level edit-mode-selection algorithm exists or is intended — `auto` is defined purely in terms of profile metadata.

**Template-resolution layer (`workflow_library.py::WorkflowLibrary.resolve()`).** Accepts `(niche, profile, edit_mode="legacy_txt2img")`. When `edit_mode == "staged_edit"` (after its own internal `"auto"` handling, mirroring `run()`'s), it resolves `{niche}_edit.json` if present in `MODULE7_NICHE_WORKFLOW_MAP`'s niche set and the file exists on disk, else falls back to `general_edit.json`, else falls back further to the legacy filename. Nine `_edit.json` templates already exist under `workflows/` (`general_edit.json`, `gaming_edit.json`, `tech_edit.json`, etc.), plus `workflows/fragments/edit_region_mask.json` and `workflows/fragments/inpaint_base.json`. This method is correctly built and already handles every case this document needs — it is a pure consumer of whatever `edit_mode` string it is given.

**The break (Gap B, §0).** `run()`'s loop body — `_process_single_candidate()` — calls `self.workflow_library.resolve(niche, profile)` at both its primary and VRAM-fallback call sites, supplying no `edit_mode` argument at all. `effective_edit_mode`, computed one method up, never reaches this call. This is the specific, narrow point this document's fix targets.

---

## 3. Root Cause Summary

| # | Cause | Layer | Status |
|---|---|---|---|
| A | `PROFILE_STANDARD_EDIT` absent from `MODULE7_PROFILE_PREFERENCE` — VRAM-based `"auto"` profile selection can never choose an edit-capable profile | Configuration (`config.py`) | Confirmed by investigation; re-confirmed here |
| B | `effective_edit_mode`, resolved in `run()`, is never forwarded to `_process_single_candidate()` → `WorkflowLibrary.resolve()` — even an explicitly-requested `staged_edit` currently resolves legacy templates | Wiring (`image_generator.py`) | Newly identified in §0 against current `main`; not present in the investigation document as uploaded, because it postdates the multi-candidate refactor |
| C | The seven-stage Python-orchestrated staged-edit pipeline (`RegionPlanValidator` → ... → `HarmonizationStage`) is constructed but never invoked | Orchestration (`image_generator.py`) | Newly identified in §0; explicitly deferred to §16, out of scope for this fix |

**Combined effect:** Cause A and Cause B are both individually sufficient to keep production on `legacy_txt2img` — fixing only one leaves the other fully blocking. Both must be fixed together for `edit_mode="auto"` (as `main.py` already requests) to reach `staged_edit` in practice. Cause C does not block the graph-level fix this document delivers; it bounds what "staged_edit" currently *means* once reachable (a ComfyUI-graph conditioning swap, not yet the full masked-composite staged pipeline) — this is stated plainly in §16 rather than left implicit.

---

## 4. Design Goals

Restating the brief's ten questions as design goals this architecture must satisfy:

1. `edit_mode="auto"` must, for at least one auto-selectable VRAM tier, actually resolve to `"staged_edit"` in production — not merely be capable of doing so in principle.
2. `ProfileSelector` must remain edit-mode-agnostic — no new selection algorithm, no new constructor parameter, no behavioral branch on `edit_mode` inside `ProfileSelector.select()`.
3. Edit-capable profiles must be registered by explicit, typed metadata (`edit_mode_default`) — never inferred from profile name string patterns.
4. `MODULE7_PROFILE_PREFERENCE` must evolve by a documented, mechanical convention that scales to future edit-capable profiles without per-profile special-casing in code.
5. `staged_edit`'s reachability must not depend on `main.py` changing its call — `main.py`'s existing `edit_mode="auto"` (line 578) is the only production entry point this fix needs to satisfy.
6. `legacy_txt2img` must remain fully functional, unmodified in behavior, and remain the explicit default for any caller that does not opt into `"auto"`/`"staged_edit"`.
7. Adding a future edit-capable profile must be possible without touching `ProfileSelector`, `WorkflowLibrary`, or `ImageGeneratorPipeline` — configuration-only.
8. Configuration must be **validatable** — a structurally-unreachable edit profile (Cause A's exact shape) must be detectable before it silently ships to production again.
9. The fix itself must be checkable by PORCE, reusing its existing rule-engine contract, not a bespoke validator living outside the observability system.
10. The fix must be delivered in independently-testable, independently-committable phases, per the project's established `Implementation → Tests → tai → Commit` rhythm.

---

## 5. Proposed Architecture

```
main.py (unchanged: edit_mode="auto")
   │
   ▼
ImageGeneratorPipeline.run()
   │  profile = ProfileSelector.select(vram, MODULE7_PROFILE)      [unchanged — still VRAM-only]
   │  effective_edit_mode = resolve("auto", profile.edit_mode_default)  [unchanged logic]
   │
   ▼  (NEW: effective_edit_mode now threaded as a parameter)
_process_single_candidate(..., effective_edit_mode=effective_edit_mode)
   │
   ▼  (NEW: forwarded to the call already accepting it)
WorkflowLibrary.resolve(niche, profile, edit_mode=effective_edit_mode)   [unchanged method, newly given its real argument]
   │
   ▼
{niche}_edit.json + fragments   ── when effective_edit_mode == "staged_edit" and an edit-capable profile was selected
{niche}.json                    ── otherwise (today's behavior, unchanged)
```

Configuration side, evaluated once at import time (`config.py`), independent of any request:

```
MODULE7_GENERATION_PROFILES              [unchanged shape; PROFILE_STANDARD_EDIT already present]
        │
        ▼  (NEW, derived, read-only)
MODULE7_EDIT_CAPABLE_PROFILES: frozenset[str]   — every profile name whose edit_mode_default == "staged_edit"
        │
        ▼  (fixed, per §7)
MODULE7_PROFILE_PREFERENCE: tuple[str, ...]     — now includes PROFILE_STANDARD_EDIT at its documented position
        │
        ▼  (NEW, §9)
validate_module7_edit_reachability()            — raises at import/startup if MODULE7_EDIT_CAPABLE_PROFILES ∩ MODULE7_PROFILE_PREFERENCE == ∅
```

Two code changes total: (1) `config.py` — add one derived constant, reorder one existing tuple, add one validation function called at module load (matching `validate_qa_weights()`'s existing precedent in `image_generator.py`, which already runs at `ProfileSelector.__init__` time). (2) `image_generator.py` — thread one additional parameter through `_process_single_candidate()`'s two existing `workflow_library.resolve()` call sites. No new class, no new stage, no new public method signature beyond one added keyword-defaulted parameter.

---

## 6. Edit Mode Resolution

`edit_mode="auto"` behaves exactly as `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` originally specified and as `run()` already implements: **it inherits the selected profile's `edit_mode_default`, full stop.** This document does not change that resolution rule — Design Goal 2 explicitly forbids making `ProfileSelector` edit-aware, and re-deriving `"auto"` from anything other than the already-selected profile would be exactly the kind of redesign the brief prohibits.

What this document changes is what "the selected profile" *can be*, and what happens to the resolved value afterward:

- **Before this fix:** `"auto"` profile selection can only ever choose among `PROFILE_PREMIUM`, `PROFILE_STANDARD`, `PROFILE_FAST`, `PROFILE_LOW_VRAM` — none declare `edit_mode_default="staged_edit"` — so `"auto"` edit-mode resolution is **structurally** always `"legacy_txt2img"`, independent of VRAM, independent of caller intent.
- **After this fix (§7):** `"auto"` profile selection can choose `PROFILE_STANDARD_EDIT` when it is the richest fitting profile in preference order for the available VRAM. When it is chosen, `"auto"` edit-mode resolution naturally becomes `"staged_edit"` — no new logic, the existing one-line `getattr` in `run()` is untouched and now has real data to act on.
- **After this fix (Gap B closed, §5):** the resolved `effective_edit_mode` — `"staged_edit"` or `"legacy_txt2img"`, whichever it computed to — is forwarded to `_process_single_candidate()` and on to `WorkflowLibrary.resolve()`, so the value `run()` computes and the templates that actually get built are, for the first time, the same thing.

Explicit non-`"auto"` requests (`edit_mode="staged_edit"` or `edit_mode="legacy_txt2img"` passed directly to `run()`/`run_image_generation_pipeline()`) are unaffected by profile selection at all — they already bypass the `getattr` branch entirely (`run()`'s `if effective_edit_mode == "auto":` guard) and are honored as stated, once Gap B's forwarding is in place. `main.py` does not need to change; it already requests `"auto"`.

---

## 7. Profile Selection Strategy

`ProfileSelector.select()` requires **zero code changes**. Its contract — deterministically choose the richest VRAM-fitting profile from an ordered preference list, or an explicitly requested profile if it fits, or `PROFILE_LOW_VRAM` as a last resort — already does exactly what editing-profile selection needs, provided the preference list actually contains an edit-capable profile. This is the crux of Design Goal 2 and the reason Cause A is a configuration defect, not an architecture defect: the selection *algorithm* was always correct; only its *input data* was incomplete.

**Fix:** `MODULE7_PROFILE_PREFERENCE` becomes:

```
("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")
```

`PROFILE_STANDARD_EDIT` is inserted immediately after `PROFILE_PREMIUM` and before `PROFILE_STANDARD` — the two profiles share the same `expected_vram_gb` (7.5 GB, per `config.py`'s existing values) and the same quality tier (both SDXL, `dpmpp_2m`/`karras`, 30 steps, `codeformer` restoration), differing only in `controlnet_enabled`/`edit_mode_default`. Placing the edit-capable variant first within its tier means: **any run with enough VRAM for `PROFILE_STANDARD`-tier quality now gets staged editing by default**, which is the correct default — editing (conditioning on the real source image) is strictly more correct than unconditioned generation whenever the hardware budget allows it, and this ordering expresses that as data, not as a new rule.

`PROFILE_PREMIUM` is intentionally left ahead of `PROFILE_STANDARD_EDIT` and is intentionally left non-edit-capable by this document — per §16, a future `PROFILE_PREMIUM_EDIT` is a natural follow-on but is not required to close Gaps A/B, and inventing one here would be scope creep against "minimal code changes."

Explicit-request selection (`MODULE7_PROFILE` config set to a literal profile name, or a direct caller passing `requested_profile="PROFILE_STANDARD_EDIT"`) already works correctly today with **no change needed** — `ProfileSelector.select()`'s `requested_profile != "auto"` branch has never depended on `MODULE7_PROFILE_PREFERENCE` membership. This document's fix is specifically for the `"auto"` path, which is what `main.py` uses.

---

## 8. Configuration Model

**Registration — explicit metadata, never name inference (Design Goal 3).** `GenerationProfile.edit_mode_default` already is the single, explicit, typed registration mechanism (`Literal["legacy_txt2img", "staged_edit"] | None`). This document does not add a second mechanism. Name-based inference (e.g., "any profile whose name ends in `_EDIT`") is explicitly rejected: it would silently misclassify a future profile named without that suffix, it duplicates information already captured by a proper field, and it violates the same "closed, typed enums over string-matching" convention `DesignBlueprint`'s `Literal` fields and `GenerationProfile.checkpoint_family` already establish elsewhere in this codebase.

**New derived constant, `config.py`:**

```
MODULE7_EDIT_CAPABLE_PROFILES: frozenset[str] = frozenset(
    name for name, profile in MODULE7_GENERATION_PROFILES.items()
    if profile.edit_mode_default == "staged_edit"
)
```

Computed once at import time directly from `MODULE7_GENERATION_PROFILES` — the single source of truth. Not hand-maintained, not editable independently of the profiles themselves, so it can never drift out of sync with what profiles actually declare. Used only for validation (§9) and PORCE's static rule (§10) — never for selection (§7 keeps selection purely VRAM/preference-order driven).

**`MODULE7_PROFILE_PREFERENCE` evolution convention (Design Goal 4, answering Q5):** documented as a comment directly above the tuple in `config.py`, not just in this document:

> When adding a new edit-capable profile (`edit_mode_default="staged_edit"`), insert its name into this tuple immediately adjacent to the non-edit profile it is a variant of, on the side (before = preferred, after = fallback) matching whether editing should be the default outcome at that VRAM tier. Never add an edit-capable profile's `GenerationProfile` entry to `MODULE7_GENERATION_PROFILES` without also placing it in this tuple (or explicitly, deliberately, leaving it reachable only via explicit `MODULE7_PROFILE` request) — an edit-capable profile absent from both is unreachable via `"auto"` by construction, which is precisely Cause A. `validate_module7_edit_reachability()` (§9) enforces the first half of this mechanically; the "adjacent to its non-edit variant" placement convention is a human review convention, not machine-enforced, since "which non-edit profile is this a variant of" is not currently modeled data.

This directly answers Q8 ("how should future profiles be added without breaking auto selection?") — by convention plus one machine-checked invariant (§9), not by new selection code.

---

## 9. Validation Strategy

**New function, `config.py` (or `image_generator.py`, colocated with `validate_qa_weights()` for consistency — implementation detail, not an architectural choice this document needs to fix):**

`validate_module7_edit_reachability(preference: tuple[str, ...] = MODULE7_PROFILE_PREFERENCE, edit_capable: frozenset[str] = MODULE7_EDIT_CAPABLE_PROFILES) -> None`

Raises `Module7Error` (existing exception type, no new exception class needed) if `edit_capable` is non-empty but `set(preference) & edit_capable` is empty — i.e., **at least one edit-capable profile is configured, but none is reachable via auto-selection.** This is exactly Cause A's shape, generalized: it catches not just today's specific omission but any future regression of the same class (someone adds a new `_EDIT` profile and forgets the preference-tuple step from §8's convention). If `edit_capable` is empty (no edit-capable profile configured at all — a legitimate, if regressive, configuration choice), the function passes silently; this validation is about *reachability of what's configured*, not about *mandating that editing be configured at all*.

Called once, at the same point `validate_qa_weights(MODULE7_QA_WEIGHTS)` is already called today — `ProfileSelector.__init__` — so it runs automatically on every `ProfileSelector()` construction (i.e., on every pipeline start), with zero new call sites for callers to remember to add. A misconfiguration is caught at process-startup time, before any generation work begins, matching the project's established "fail loudly and typed, before spending compute" convention (the same phrase `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §7.0 uses to describe `RegionPlanValidator`'s own placement).

This answers Q9 directly: unreachable edit profiles are detected by a pure-data set-intersection check, run automatically at the same lifecycle point existing config validation already runs, raising the same exception family every other Module 7 startup-validation failure already raises.

---

## 10. PORCE Integration

Per the brief ("reuse PORCE," "how should PORCE validate edit-mode activation") and per the investigation document's own forward pointer (§0, "`RULE-LAT-01`'s docstring... someone already diagnosed this pattern... never wired up"), this document specifies **one new PORCE rule**, added to the existing rule engine exactly as every other rule already is — no new engine, no new report model, no new persistence path.

**`RULE-EDIT-02` — Edit Capability Reachability** (static configuration check, `observability/diagnostics/rules/edit_mode_resolution_rules.py`, new file alongside the existing six rule files): implements `IDiagnosticRule`, following the exact shape `SourceNeverEncodedRule`/`RULE-LAT-01` already establishes in `latent_initialization_rules.py`. Unlike every existing rule, this one does not need a `TraceFacts`/`RuleContext` populated from a generation run at all — it evaluates `MODULE7_GENERATION_PROFILES` and `MODULE7_PROFILE_PREFERENCE` directly, the same two config objects §9's `validate_module7_edit_reachability()` checks. It produces a `Finding` with `severity="FAIL"`, `category="conditioning"`, `affected_module="module7_profile_selection"`, `root_cause` naming the exact excluded profile(s), and `recommended_action` describing the §7/§8 fix, whenever the same condition §9 validates at startup is true.

This is **deliberately redundant with §9**, by design, not by oversight: §9's `validate_module7_edit_reachability()` is a hard, fail-fast, synchronous startup gate that prevents the misconfiguration from ever running in the first place; `RULE-EDIT-02` is PORCE's independent, advisory, trace-and-config-driven detection of the same condition, consistent with every other PORCE rule's non-blocking `Finding`-based design (per the investigation document's own risk table: "it's advisory, matching every existing rule's `Finding`-based design"). The two serve different purposes — §9 prevents; `RULE-EDIT-02` diagnoses and documents, including retroactively against the nine historical traces the investigation already has, and going forward against every future trace, providing an audit trail independent of whether the startup gate was ever bypassed (e.g., a config change applied without restarting the process in some deployment topology). Implementing `RULE-EDIT-02` here, as part of this activation fix, also directly fulfills the investigation document's own §14 Phase 1 dependency, which named this exact rule as a prerequisite for validating its own retroactive investigation of the nine historical traces — this document's Phase (§13) that adds `RULE-EDIT-02` should be sequenced so the investigation's Phase 1 can consume it, rather than each document re-implementing it independently.

No change to `RuleExecutionEngine`, `RuleRegistry`, `RootCauseAssembler`, or any existing rule file. `RULE-EDIT-02` is registered into the existing registry exactly as the six existing rule modules already are.

---

## 11. Migration Plan

1. Land `MODULE7_EDIT_CAPABLE_PROFILES` and `validate_module7_edit_reachability()` (§8, §9) with `MODULE7_PROFILE_PREFERENCE` **unchanged** — the validator will (correctly) not yet fail, since the profile-preference fix hasn't landed; this step is pure new, inert code, matching the project's established "land inert, then activate" precedent (`MODULE7_PHASE4...`'s own §15 uses the identical sequencing).
2. Land the `_process_single_candidate()` parameter-threading fix (Gap B, §5) — additive, keyword-defaulted (`effective_edit_mode: str = "legacy_txt2img"`), so existing callers/tests that don't pass it see no behavior change. At this point, `staged_edit` is reachable **only** via explicit `edit_mode="staged_edit"` + explicit `MODULE7_PROFILE="PROFILE_STANDARD_EDIT"` — a fully manual, opt-in path, useful for validating Gap B's fix in isolation before touching auto-selection at all.
3. Land the `MODULE7_PROFILE_PREFERENCE` reordering (§7). This is the step that actually changes default production behavior — from this point on, `main.py`'s existing `edit_mode="auto"` call can resolve to `staged_edit` for any run with ≥7.5 GB usable VRAM. `validate_module7_edit_reachability()` (already landed in step 1) now passes for the first time and will catch any future regression of this exact ordering.
4. Land `RULE-EDIT-02` (§10), independent of steps 1–3's runtime behavior — it can be validated against the historical nine traces (all of which will correctly produce a `FAIL` finding against pre-fix config, and a `PASS` against post-fix config) before or after steps 1–3 ship.
5. No data migration, no artifact schema change, no `ImageGenerationResult`/`GeneratedAsset`/manifest field changes anywhere in this document — every change is either new, inert code or a two-line reordering of existing configuration.

---

## 12. Backward Compatibility

- `legacy_txt2img` is untouched in every respect: template resolution, workflow graph shape, every existing profile's `expected_vram_gb`/`checkpoint`/etc. Any caller that does not pass `edit_mode` at all still gets `run()`'s existing default, `"legacy_txt2img"` — unchanged.
- `_process_single_candidate()`'s new parameter is keyword-defaulted to `"legacy_txt2img"`, so any direct caller of this method (tests, future extensions) that doesn't know about the new parameter continues to get exactly today's behavior.
- `WorkflowLibrary.resolve()` is called with zero changes to its own signature or logic — it already accepts `edit_mode` and already defaults to `"legacy_txt2img"`; this fix only ensures its *real* argument is finally supplied when one is available.
- `ProfileSelector.select()` — zero changes, per Design Goal 2.
- Every existing `GenerationProfile` entry — `PROFILE_PREMIUM`, `PROFILE_STANDARD`, `PROFILE_FAST`, `PROFILE_LOW_VRAM` — is untouched. `PROFILE_STANDARD_EDIT` itself is untouched (it already exists with correct fields); only its position in a separate tuple changes.
- The only externally-observable behavior change for any existing deployment is: runs that previously auto-selected `PROFILE_STANDARD` (VRAM ≥ 7.5 GB, no explicit profile override) now auto-select `PROFILE_STANDARD_EDIT` instead, and — because Gap B is now fixed — actually run `staged_edit` when the caller requested `"auto"`. This is the intended, correct outcome per the brief, not a regression; it is called out explicitly here because it is the one place this fix is deliberately *not* behavior-preserving, and operators relying on `PROFILE_STANDARD`'s exact non-edit behavior at that VRAM tier should note it (see §15, risk row 1).

---

## 13. Implementation Phases

Each phase follows the required rhythm: **Implementation → Tests → `tai` → Commit.** `tai doctor` (existing system health-check subcommand, `modules/cli.py`) is run at the end of every phase as the `tai` gate, since it already exercises config-load-time validation (relevant directly to §9's new startup check) without requiring a live ComfyUI/GPU session.

**Phase 1 — Reachability validation, inert.**
Implementation: add `MODULE7_EDIT_CAPABLE_PROFILES` and `validate_module7_edit_reachability()` to `config.py`/`image_generator.py`; wire the call into `ProfileSelector.__init__` alongside the existing `validate_qa_weights()` call.
Tests: unit test asserting the validator raises against today's (pre-fix) `MODULE7_PROFILE_PREFERENCE`, and passes against a synthetic corrected tuple — proves the check is correctly discriminating before it's relied on.
`tai`: `tai doctor` — confirm it surfaces the (expected, pre-fix) validation failure clearly, since `MODULE7_PROFILE_PREFERENCE` has not yet been corrected in this phase.
Commit: config validation only; production behavior unchanged (staged_edit still unreachable — expected and correct at this point).

**Phase 2 — Thread `effective_edit_mode` through the candidate loop (Gap B).**
Implementation: add a keyword-defaulted `effective_edit_mode` parameter to `_process_single_candidate()`; pass it from `run()`'s already-computed value at both existing call sites; forward it into both of `_process_single_candidate()`'s existing `workflow_library.resolve()` calls (primary + VRAM-fallback).
Tests: unit test on `_process_single_candidate()` (or an integration test on `run()` with a stubbed `ComfyUIClient`) asserting that `edit_mode="staged_edit"` combined with an explicitly-requested edit-capable profile (`MODULE7_PROFILE="PROFILE_STANDARD_EDIT"`) results in `WorkflowLibrary.resolve()` being called with `edit_mode="staged_edit"` and an `_edit.json` template being selected — and a regression test that `edit_mode="legacy_txt2img"` (today's default) is byte-for-byte unaffected.
`tai`: `tai doctor` plus a manual `tai run` smoke test (per the CLI's existing subcommands) with `MODULE7_PROFILE` explicitly pinned to `PROFILE_STANDARD_EDIT`, confirming a non-legacy template is actually selected end-to-end.
Commit: `staged_edit` now reachable via explicit opt-in only; `"auto"` still resolves to `legacy_txt2img` (Cause A not yet fixed) — expected at this point.

**Phase 3 — Fix `MODULE7_PROFILE_PREFERENCE` (Gap A) — activation.**
Implementation: reorder the tuple per §7. Add the `config.py` comment documenting the §8 convention.
Tests: the Phase 1 validator test now flips — assert `validate_module7_edit_reachability()` passes against the corrected tuple (already written in Phase 1 as the "synthetic corrected tuple" case; now it's the real one). Add an end-to-end test with `MODULE7_PROFILE="auto"` and sufficient mocked VRAM, asserting `PROFILE_STANDARD_EDIT` is selected and `effective_edit_mode == "staged_edit"`.
`tai`: `tai doctor` (validator passes clean for the first time) + `tai run` against a real or representative-VRAM ComfyUI instance, confirming an actual staged-edit generation completes and the output visibly reflects the source thumbnail (the concrete symptom this whole fix targets).
Commit: `main.py`'s existing `edit_mode="auto"` now reaches `staged_edit` in production for the first time. This is the activation commit.

**Phase 4 — PORCE `RULE-EDIT-02`.**
Implementation: add `observability/diagnostics/rules/edit_mode_resolution_rules.py` with `RULE-EDIT-02`; register it in the existing rule registry.
Tests: unit test per the existing `tests/test_observability/` convention, against both pre-fix and post-fix synthetic config, asserting `FAIL`/`PASS` findings respectively; golden-file test replaying the nine historical traces (pre-fix data) confirming a `FAIL` finding for all nine, matching the investigation's manual conclusion.
`tai`: `tai doctor`.
Commit: PORCE can now detect this defect class going forward and can retroactively confirm the historical traces' root cause mechanically, closing the loop the investigation document opened.

---

## 14. Testing Strategy

- **Unit — `validate_module7_edit_reachability()` (§9):** table-driven over `(preference_tuple, edit_capable_set)` pairs — empty intersection with non-empty `edit_capable` → raises; empty `edit_capable` → passes regardless of preference tuple; non-empty intersection → passes. No I/O, no mocking, matching the style of `validate_qa_weights()`'s existing tests.
- **Unit — `MODULE7_EDIT_CAPABLE_PROFILES` derivation:** asserts it is correctly and automatically derived from `MODULE7_GENERATION_PROFILES` without hand-maintenance — add a synthetic profile with `edit_mode_default="staged_edit"` to a copy of the profiles dict and confirm it appears in the derived set.
- **Unit — Gap B threading:** the single highest-value test in this document. Stub `WorkflowLibrary.resolve()` (already mockable via `ImageGeneratorPipeline`'s existing constructor injection) and assert, for a call to `run()` with `edit_mode="staged_edit"`, that `resolve()` receives `edit_mode="staged_edit"` — not its default. Mirror for `edit_mode="auto"` with an edit-capable vs. non-edit-capable selected profile, and for the VRAM-fallback retry path specifically (both call sites in `_process_single_candidate()` must be covered, since Gap B affects both identically).
- **Regression — `legacy_txt2img` byte-identical:** the existing `tests/test_image_generator.py` baseline fixture (used identically in `MODULE7_PHASE4...`'s own testing strategy) must produce an unchanged `ImageGenerationResult` for the default, no-`edit_mode`-argument call shape — proving Phases 2–3 are additive, not disruptive, exactly as Design Goal 6 requires.
- **Integration — end-to-end `"auto"` activation:** with a stubbed `ComfyUIClient` and mocked `available_vram_gb` sufficient for `PROFILE_STANDARD_EDIT`, assert the full `run()` call selects the edit-capable profile, resolves `staged_edit`, and builds a workflow graph whose `_meta.name`/template path corresponds to an `_edit.json` file — this is the test that directly proves the symptom (unrelated images) is fixed at the mechanism level, without requiring a live GPU.
- **PORCE — `RULE-EDIT-02`:** per §13 Phase 4 — synthetic config fixtures (pass/fail) plus the golden-file replay of the nine historical traces, following `tests/test_observability/`'s established structure exactly, per the investigation document's own §15.
- **No test in this document requires a live ComfyUI server or GPU** — every check operates on config objects, mocked collaborators, or already-persisted trace JSON, consistent with how `WorkflowLibrary`/`ProfileSelector`/PORCE are already tested elsewhere in the repository.

---

## 15. Risks

| Risk | Mitigation |
|---|---|
| Reordering `MODULE7_PROFILE_PREFERENCE` changes default behavior for any deployment currently relying on `PROFILE_STANDARD` being auto-selected at its VRAM tier (§12) | Called out explicitly as the one intended, non-preserving behavior change; sequenced as its own commit (Phase 3) separable from Phases 1/2/4, so it can be reverted independently (revert the tuple order only) if an operator needs to roll back just the activation step while keeping the reachability/wiring fixes |
| `PROFILE_STANDARD_EDIT` has never actually run in production (per the investigation's nine traces, all pre-activation) — its `controlnet_enabled=True`/`ipadapter_enabled=True` combination, and the `_edit.json` templates' fragment assembly, are unexercised at runtime even though correctly built per static inspection | Phase 3's `tai run` step (§13) is deliberately a real end-to-end smoke test, not just a unit/mock test, specifically to surface any first-run-only defect in the fragment assembly path before this is considered fully activated; §0's Gap-C boundary (graph-level only, not the Python staged-stage pipeline) narrows what's actually being exercised for the first time, reducing surface area |
| Gap C (§0, §16) means "staged_edit" post-fix delivers graph-level conditioning only, not the full masked-composite paste-back guarantee `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` ultimately specifies — a reader of that document could reasonably expect more than this fix delivers | §0 and §16 state this boundary explicitly and by name, rather than letting "staged editing pipeline activated" imply completeness it doesn't have; this document's title and scope statement are deliberately narrow ("activation fix," not "editing engine completion") |
| `RULE-EDIT-02`'s config-only check could pass while Gap B (a code-wiring issue, not a config issue) remains broken, giving a false sense of completeness if Phase 4 ships before Phase 2 | Phases are explicitly sequenced (§13) so Phase 2 (Gap B fix) lands before Phase 3 (activation); `RULE-EDIT-02` only ever claims to check profile-preference reachability (Cause A), not end-to-end wiring — its `Finding` text should say so precisely, per §10, to avoid over-claiming what a config-only rule can verify |
| Future contributors add a new edit-capable profile and follow the §8 convention comment inconsistently (human-review-dependent placement, not machine-enforced) | `validate_module7_edit_reachability()` (§9) mechanically catches the failure mode that actually matters (unreachable, not merely sub-optimally placed) — placement quality is a review-time concern, reachability is a build-time gate |

---

## 16. Future Extensions

- **Gap C — wiring the Python-orchestrated staged-edit pipeline.** `RegionPlanValidator` → `BaseLatentStage` → `BackgroundEditStage`/`ObjectEditStage` → `MaskedCompositeStage` → `TypographyStage` → `HarmonizationStage` (`MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §9) are built but inert (§0). A follow-on architecture — deliberately not this one — should design how `_process_single_candidate()` branches into this stage sequence when `effective_edit_mode == "staged_edit"`, including how it coexists with this document's graph-level template swap (the two are not mutually exclusive: the `_edit.json` template/fragment mechanism this document activates could remain the ComfyUI-submission layer, with the Python stages wrapping around it for region planning and paste-back compositing, exactly as §9 of the V2 document originally laid out).
- **`PROFILE_PREMIUM_EDIT`.** Following the exact pattern `PROFILE_STANDARD_EDIT` already establishes, a `flux`-family edit-capable profile could be added at the top VRAM tier, purely as new `MODULE7_GENERATION_PROFILES`/`MODULE7_PROFILE_PREFERENCE` data, requiring no further code change beyond what this document already delivers — direct validation that Design Goal 7's "configuration-only" claim holds.
- **Per-niche edit-capability validation.** §9's validator confirms *a* profile is reachable; it does not confirm every niche has a corresponding `{niche}_edit.json` (today, `WorkflowLibrary.resolve()`'s existing `general_edit.json` fallback covers any gap silently). A future, stricter validation mode could enumerate `MODULE7_NICHE_WORKFLOW_MAP`'s keys and assert an `_edit.json` variant exists for each — deliberately not required here, since silent fallback-to-general is itself the existing, correct, already-tested behavior this document has no reason to change.
- **`RULE-GRAPH-01` (submitted-graph capture, per the investigation document's own §7.1/§14 Phase 2).** Once Gap B is fixed and staged_edit actually submits `_edit.json`-based graphs in production, the investigation's own proposed submitted-graph snapshot becomes immediately useful for verifying, per real run, that `has_vae_encode_node`/`has_controlnet_node`/`has_ipadapter_node` are true when expected — this document's Phase 3 activation is the natural trigger for prioritizing that investigation-side work, though implementing it remains that document's scope, not this one's.

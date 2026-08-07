# Phase 3.1 — Knowledge Base Foundation Architecture

**Status:** Completed & Production-Ready  
**Subsystem:** Thumbnail Intelligence Engine — Phase 3.1 Knowledge Base Foundation  
**Package:** `thumbnail_intelligence/knowledge_base/` (aliased at `intelligence_kb/`)  

---

## 1. Executive Summary

Phase 3.1 establishes the production **Knowledge Base Foundation** for the Thumbnail Intelligence Engine. As specified in `docs/thumbnail_intelligence_architecture.md` (§5–§10, §15, §19) and `docs/thumbnail-renderer-v2-architecture-v2.md` (§3), this subsystem provides:

1. **Production Data Models**: Strictly validated, frozen Pydantic v2 data contracts for cross-video, cross-creator, and competitive intelligence.
2. **Local Atomic Persistence**: Zero-external-database, crash-safe local JSON storage using temporary-file-then-atomic-replace with `os.fsync` and automated historical version backup.
3. **Pluggable Registry Engine**: Generic, typed in-memory and disk-backed registries supporting lifecycle CRUD operations, pagination, sorting, version history, and **IndexHook** listener interfaces for future vector/embedding indexes without modifying public APIs.
4. **Semantic Versioning & Schema Migration Pipeline**: Automated `SemVer` comparison and registered migration transform chains ensuring forward and backward compatibility.
5. **Enforced Grounding Validation Gate**: Hard validation requiring every `DesignReason` to cite at least one valid, non-empty `EvidenceReference` (§19.2 "Interpretation, not invention").
6. **Structured Exception Hierarchy**: Rich error types with machine-readable error codes and context payloads across all operations.

---

## 2. Package & Folder Structure

```
thumbnail_intelligence/
├── __init__.py
└── knowledge_base/
    ├── __init__.py          # Unified exports for all models, registries, storage, and errors
    ├── models.py            # Complete Pydantic v2 domain models
    ├── storage.py           # AtomicFileWriter and VersionedFileStorage
    ├── repository.py        # KnowledgeBaseRepository composing all typed registries
    ├── registry.py          # KnowledgeRegistry[T] and IndexHook[T] protocol
    ├── versioning.py        # SemVer parser, version ordering, and MigrationRegistry
    ├── serialization.py     # KBSerializer with ISO timestamps and custom JSON encoder
    ├── validation.py        # ModelValidator, EvidenceValidator, ConstraintValidator
    ├── exceptions.py        # Structured KnowledgeBaseError hierarchy
    └── config.py            # KnowledgeBaseConfig, directory paths, and storage policies

intelligence_kb/
└── __init__.py              # Direct alias to thumbnail_intelligence.knowledge_base
```

### On-Disk Storage Structure
```
data/
└── intelligence_kb/
    ├── entries/                     # KnowledgeEntry multimodal index records
    │   └── versions/                # Historical version snapshots (id@version.json)
    ├── creator_profiles/            # CreatorProfile multi-channel identity records
    ├── channel_profiles/            # ChannelProfile per-channel style signatures
    ├── competitors/                 # CompetitorProfile competitor intelligence
    ├── archetypes/                  # Archetype templates with structural predicates
    ├── design_patterns/             # DesignPattern psychological & visual patterns
    ├── visual_patterns/             # VisualPattern granular lighting & composition cues
    ├── thumbnail_patterns/          # Composite thumbnail patterns
    ├── brand_constraints/           # BrandConstraint brand rules and palettes
    ├── identity_constraints/        # IdentityConstraint facial & instance locking
    ├── design_briefs/               # DesignBrief synthesis records
    └── backups/                     # Emergency restore snapshots
```

---

## 3. Data Models Specification

| Model | Primary ID | Purpose & Key Fields |
|---|---|---|
| `KnowledgeEntry` | `entry_id` | Unified multimodal corpus entry across all types (`ARCHETYPE_EXAMPLE`, `HISTORICAL_THUMBNAIL`, `COMPETITOR_THUMBNAIL`, `DESIGN_PATTERN`), storing OpenCLIP 512-dim embedding, facets, and outcome linkage. |
| `CreatorProfile` | `creator_id` | Stable multi-channel identity, primary niche, cross-channel consistency score, and extracted `brand_rules` (`DesignReason` list). |
| `ChannelProfile` | `channel_id` | Per-channel visual signature referencing `CreatorStyleEmbedding`, archetype affinity frequencies, dominant hook types, and `brand_stability_score`. |
| `CompetitorProfile` | `competitor_id` | Competitor channel intelligence, style embedding centroid, dominant archetypes, hook types, color palette signature, and ingestion status. |
| `Archetype` | `archetype_id` | Named holistic thumbnail design template (e.g. `big_face_reaction`, `curiosity_gap`, `before_after_split`) with checkable `defining_scene_graph_pattern` predicates. |
| `EvidenceReference` | `source_id` | Grounding reference linking claims directly to scene elements, relationships, creator style signatures, or historical outcomes. |
| `BrandConstraint` | `constraint_id` | Enforced brand palette, font references, logo placement rules, mandatory elements, and prohibited tropes. |
| `IdentityConstraint` | `constraint_id` | Creator protection constraint specifying locked instance IDs, pose lock requirements, and facial similarity thresholds (default ≥ 0.90). |
| `VisualPattern` | `pattern_id` | Granular lighting/composition techniques (e.g. `rim_light_subject_edge`, `high_contrast_vignette`) with niche frequency and evidence grade. |
| `DesignPattern` | `pattern_id` | Reusable design pattern categorized by `pattern_scope` (`audience_psychology` or `visual_design`). |
| `ThumbnailPattern` | `pattern_id` | Composite thumbnail blueprint combining archetype, visual patterns, composition rules, and historical CTR uplift tracking. |
| `DesignReason` | `reason_id` | Explainable strategic reason enforcing the Grounding Gate (`len(evidence) >= 1`). |
| `ArchetypeMatch` | `video_id` | Auditable result of matching a thumbnail against the Archetype library with match confidence and method. |
| `DifferentiationSummary` | `channel_id` | Comparative differentiation assessment contrasting a creator against their competitive set with convergence risk tracking. |

---

## 4. Subsystem Responsibilities

### 4.1 Storage & Atomic Persistence (`storage.py`)
- **`AtomicFileWriter`**: Writes data to a sibling `.tmp` file, calls `flush()` and `os.fsync()`, and replaces the target file using `os.replace` (POSIX atomic rename, Windows safe replace). Orphan `.tmp` files are automatically cleaned up if an exception occurs.
- **`VersionedFileStorage[T]`**: Generic namespace-sharded file store. Automatically archives previous schema versions into `versions/<id>@<version>.json` upon updates and applies automated migration on read if the payload version is older than the target model.

### 4.2 Pluggable Registry (`registry.py`)
- **`KnowledgeRegistry[T]`**: High-performance in-memory caching layered over `VersionedFileStorage`.
- **API Methods**: `register()`, `lookup()`, `get()`, `update()`, `remove()`, `list()`, `version()`, `count()`, `exists()`, `clear()`.
- **Future Vector Index Hooks**: Implements the `IndexHook[T]` protocol (`on_registered`, `on_updated`, `on_removed`). Future vector indexing and retrieval backends (Phase 3.2+) register via `register_index_hook()` without modifying registry interfaces or consumer code.

### 4.3 Semantic Versioning & Migration (`versioning.py`)
- **`SemVer`**: Immutable semantic version representation supporting mathematical comparison (`<`, `<=`, `==`, `>=`, `>`) and backward compatibility evaluation (`is_compatible`).
- **`MigrationRegistry`**: Directed graph migration pipeline using breadth-first search (BFS) to automatically resolve and execute multi-step migration paths between arbitrary schema versions.

### 4.4 Validation & Grounding Gate (`validation.py`)
- **`EvidenceValidator`**: Rejects any ungrounded `DesignReason` that has an empty evidence list or invalid confidence scores (§19.2).
- **`ConstraintValidator`**: Verifies that `BrandConstraint` does not contain contradictory mandatory and prohibited elements and ensures `IdentityConstraint` similarity thresholds are within `[0.0, 1.0]`.
- **`ModelValidator`**: Validates embedding vector dimensions (512-dim), ensures finite float values (no NaNs), and checks non-empty string invariants.
- **`SchemaIntegrityValidator`**: Pre-validates raw JSON dictionaries against required schema keys prior to model deserialization.

### 4.5 Repository Orchestration (`repository.py`)
- **`KnowledgeBaseRepository`**: Central facade orchestrating all individual typed registries (`entries`, `creator_profiles`, `channel_profiles`, `competitor_profiles`, `archetypes`, `brand_constraints`, `identity_constraints`, `visual_patterns`, `design_patterns`, `thumbnail_patterns`).
- **Seed Data Bootstrap**: Includes `seed_default_archetypes()` and `seed_default_patterns()` providing production-ready curated seeds for industry-standard thumbnail templates.

---

## 5. Future Extension Points (Phase 3.2+)

1. **Vector Retrieval Engine (Phase 3.2)**:
   - Attach an `IndexHook` to `repo.entries` and `repo.archetypes` to maintain in-memory NumPy/OpenCLIP cosine similarity arrays.
   - Implement two-stage hybrid retrieval (`RetrievalQuery` -> Hard Facet Filter -> Top-K Vector Ranking).
2. **Text Embedding Backend**:
   - Wrap OpenCLIP's text encoder to generate 512-dim vectors for video titles, headlines, and curiosity hooks.
3. **Strategic Reasoning Engines**:
   - Visual Storytelling Engine (`StoryFrame`).
   - CTR Reasoning Engine (`CTRHypothesis`).
   - Emotion Reasoning Engine (`EmotionProfile`).
   - Audience Psychology Engine (`AudiencePattern`).
4. **DesignBrief Generator**:
   - Consolidate evidence from the Strategic Reasoning Layer into a validated `DesignBrief` consumed additively by Module 5 (`RedesignSpecification`), Module 5.5 (`DesignBlueprint`), and Module 9 (`DecisionManifest`).

---
title: Module 6.5 – Visual Reference Engine
version: 1.0
status: Architecture v1.0 (Approved)
author: Mohammed Afsar
architecture: Gemini 2.5 Pro
formatting: Claude
---

# Module 6.5 — Visual Reference Engine

## Table of Contents

- [Overview](#overview)
- [Design Goals](#design-goals)
- [Non Goals](#non-goals)
- [Pipeline Position](#pipeline-position)
- [Responsibilities](#responsibilities)
- [Data Flow](#data-flow)
- [Internal Architecture](#internal-architecture)
  - [1. `VisualReferenceEngine` (Core Orchestrator)](#1-visualreferenceengine-core-orchestrator)
  - [2. `FaceProcessor`](#2-faceprocessor)
  - [3. `SegmentationProcessor`](#3-segmentationprocessor)
  - [4. `TopologyProcessor`](#4-topologyprocessor)
  - [5. `AssetWriter`](#5-assetwriter)
  - [6. `ManifestBuilder`](#6-manifestbuilder)
- [Folder Structure](#folder-structure)
- [Data Models](#data-models)
  - [1. `BoundingBox`](#1-boundingbox)
  - [2. `AssetMetadata`](#2-assetmetadata)
  - [3. `VisualReferenceManifest`](#3-visualreferencemanifest)
- [Manifest Schema](#manifest-schema)
- [Processor Interfaces](#processor-interfaces)
- [Cache Strategy](#cache-strategy)
- [Validation Strategy](#validation-strategy)
- [Logging Strategy](#logging-strategy)
- [Exception Hierarchy](#exception-hierarchy)
- [Configuration](#configuration)
- [Testing Strategy](#testing-strategy)
  - [Unit Tests (`tests/test_visual_reference_engine.py`)](#unit-tests-teststest_visual_reference_enginepy)
  - [Integration Tests](#integration-tests)
  - [Contract Tests](#contract-tests)
  - [Failure Tests](#failure-tests)
  - [Performance Tests](#performance-tests)
- [Integration with Module 4](#integration-with-module-4)
- [Integration with Module 5](#integration-with-module-5)
- [Integration with Module 6](#integration-with-module-6)
- [Integration with Module 7](#integration-with-module-7)
- [Future Extensions](#future-extensions)
- [Risks](#risks)
- [Open Questions](#open-questions)
- [Implementation Notes](#implementation-notes)

---

## Overview

Module 6.5 (Visual Reference Engine, or VRE) serves as the deterministic conditioning preparation layer situated between the semantic prompt compilation stage (Module 6) and the ComfyUI/ControlNet execution layer (Module 7). In production visual generation pipelines, raw user or creator assets (such as creator face images, source product crops, and raw scene plates) cannot be passed directly to diffusion models without structural decomposition. VRE ingests raw visual assets and metadata outputs from upstream stages, performs non-generative computer vision analysis, extracts precise spatial control maps, and compiles an explicit `ReferenceManifest`.

Every architectural decision within VRE is bound by strict constraints: it executes zero image generation, enforces deterministic folder sharding based on `video_id`, relies on aggressive caching to bypass redundant inference, and guarantees absolute adherence to downstream ComfyUI requirements.

---

## Design Goals

* **Zero Generation Mandate:** Guarantee that VRE contains no diffusion, inpainting, or generative code paths. Its sole duty is asset transformation, decomposition, and analytical mapping.
* **Deterministic Asset Sharding:** Ensure all outputs are written to isolated directories keyed by `video_id` to prevent cross-talk in multi-threaded containerized worker environments.
* **Idempotent Execution with Content Hashing:** Implement SHA-256 source-asset fingerprinting to instantly return cached manifests when input assets remain unchanged.
* **Strict Boundary Enforcement:** Decouple CV processing libraries (e.g., OpenCV, MediaPipe, PyTorch-based depth/segmentation models) from the core orchestration logic via rigorous Processor Interfaces.
* **ComfyUI ControlNet Readiness:** Format all geometric maps (Canny, Depth, Segmentation masks) to exact source resolutions matching downstream node requirements.

---

## Non Goals

* **Text-to-Image / Image-to-Image Inference:** VRE will never invoke a Stable Diffusion checkpoint, LoRA weight, or VAE decoder.
* **Prompt Engineering:** Prompt compilation, weighting, and LLM orchestration remain strictly within Module 6.
* **WebSocket Transport & Queue Management:** Transport handling, progress polling, and execution scheduling are the sole responsibilities of Module 7.
* **Dynamic Resolution Rescaling:** VRE operates strictly on source dimensions unless explicitly overridden by explicit configuration constraints.

---

## Pipeline Position

```text
[Module 4: Metadata Ingestion] 
       │
       ▼
[Module 5: Analysis & Scoring] 
       │
       ▼
[Module 6: Prompt Compilation] 
       │
       ▼
【Module 6.5: Visual Reference Engine】 ◄── (Current Specification)
       │
       ▼
[Module 7: ComfyUI Generation Engine]

```

VRE consumes compiled prompt packages and asset references from Module 6, reads raw metadata and image paths established by Modules 1 through 5, and yields a fully formed reference manifest consumed directly by Module 7 during ComfyUI workflow payload construction.

---

## Responsibilities

* **Creator Face Extraction & Masking:** Locate creator faces within source thumbnails, generate tight bounding box crops, and produce alpha-blended binary face masks for identity preservation (IP-Adapter/Inpainting).
* **Object Isolation & Semantic Cropping:** Isolate salient foreground objects from raw scene plates, outputting cropped object plates and corresponding segmentation masks.
* **Foreground / Background Decomposition:** Separate composite image plates into clean foreground subject layers and background environment plates.
* **Topological Map Generation:** Compute metric depth maps (via lightweight monocular depth estimators) and structural edge maps (via Canny operators) optimized for ControlNet conditioning.
* **Manifest Compilation & Persistence:** Aggregate file paths, checksums, dimensions, and confidence metrics into a versioned JSON manifest (`reference_manifest.json`).

---

## Data Flow

```mermaid
graph TD
    A[Module 6: Prompt Package] -->|video_id & Source Paths| B(VisualReferenceEngine Core)
    C[Raw Thumbnail Asset] --> B
    
    B -->|Check Cache| D{Manifest Exists & Hash Valid?}
    D -->|Yes| E[Return Cached Manifest]
    D -->|No| F[Initialize Processor Pipeline]
    
    F --> G[FaceProcessor]
    F --> H[SegmentationProcessor]
    F --> I[TopologyProcessor]
    
    G -->|creator_face.png, face_mask.png| J[AssetWriter]
    H -->|object_crop.png, object_mask.png, fg/bg plates| J
    I -->|depth_map.png, canny_map.png| J
    
    J -->|Write Files to Disk| K[Disk Storage /data/visual_references/{video_id}/]
    J -->|Pass Metadata| L[ManifestBuilder]
    
    L -->|Compile & Validate| M[reference_manifest.json]
    M --> E
    E -->|Output Manifest| N[Module 7: ComfyUI Integration]

```

---

## Internal Architecture

### 1. `VisualReferenceEngine` (Core Orchestrator)

* **Responsibility:** Acts as the primary entry point and state machine managing execution flow, caching validation, pipeline error handling, and writer coordination.
* **Public Methods:**
* `prepare_assets(video_id: str, source_image_path: str, options: Optional[Dict] = None) -> VisualReferenceManifest`
* `clean_assets(video_id: str) -> bool`


* **Internal Helpers:**
* `_compute_asset_hash(file_path: str) -> str`
* `_verify_cache(video_id: str, asset_hash: str) -> Optional[VisualReferenceManifest]`
* `_dispatch_processors(source_image: np.ndarray, target_dir: Path)`


* **Dependencies:** `FaceProcessor`, `SegmentationProcessor`, `TopologyProcessor`, `AssetWriter`, `ManifestBuilder`, `Config`.
* **Failure Modes:** `SourceImageNotFoundError` on missing inputs; `AssetWriteError` on permission or disk-quota limits.

### 2. `FaceProcessor`

* **Responsibility:** Detects human facial structures, extracts canonical face crops, and compiles strict black-and-white face masks.
* **Public Methods:**
* `process(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]`


* **Internal Helpers:**
* `_detect_face_bounding_box(image: np.ndarray) -> Optional[BoundingBox]`
* `_generate_alpha_mask(image: np.ndarray, bbox: BoundingBox) -> np.ndarray`


* **Dependencies:** OpenCV, MediaPipe / InsightFace detection bindings.
* **Failure Modes:** `FaceDetectionFailedWarning` (non-fatal; logs warning and returns `None` values to permit pipeline continuation without face conditioning).

### 3. `SegmentationProcessor`

* **Responsibility:** Performs foreground/background matte extraction and salient object localization.
* **Public Methods:**
* `process(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` (Returns foreground, background, object crop, object mask)


* **Internal Helpers:**
* `_extract_matte(image: np.ndarray) -> np.ndarray`
* `_locate_salient_object(image: np.ndarray, alpha_matte: np.ndarray) -> BoundingBox`


* **Dependencies:** Rembg / PyTorch segmentation weights.
* **Failure Modes:** `SegmentationInferenceError` on model timeout or memory exhaustion.

### 4. `TopologyProcessor`

* **Responsibility:** Computes structural geometry representations required for ControlNet depth and edge conditioning.
* **Public Methods:**
* `generate_depth_map(image: np.ndarray) -> np.ndarray`
* `generate_canny_map(image: np.ndarray) -> np.ndarray`


* **Internal Helpers:**
* `_apply_monocular_depth(image: np.ndarray) -> np.ndarray`
* `_apply_canny_edge_detection(image: np.ndarray) -> np.ndarray`


* **Dependencies:** OpenCV Canny implementation, MiDaS / ZoeDepth lightweight ONNX runtime.
* **Failure Modes:** `TopologyExtractionError` on malformed numpy arrays or unsupported color channels.

### 5. `AssetWriter`

* **Responsibility:** Handles atomic disk persistence, directory creation, and file permissions for all generated artifacts.
* **Public Methods:**
* `write_image(array: np.ndarray, destination_path: Path) -> bool`
* `purge_directory(target_dir: Path) -> bool`


* **Internal Helpers:**
* `_ensure_directory_exists(path: Path)`
* `_atomic_write(array: np.ndarray, path: Path)`


* **Dependencies:** Standard Python `pathlib`, OpenCV image encoding.
* **Failure Modes:** `AssetWriteError` when disk space is exhausted or paths are read-only.

### 6. `ManifestBuilder`

* **Responsibility:** Aggregates processing metadata, file hashes, and relative or absolute URI paths into a structured Pydantic schema.
* **Public Methods:**
* `build(video_id: str, source_path: str, asset_paths: Dict[str, str], metadata: Dict[str, Any]) -> VisualReferenceManifest`
* `serialize_to_disk(manifest: VisualReferenceManifest, destination_path: Path)`


* **Internal Helpers:**
* `_validate_schema(manifest_dict: dict)`


* **Dependencies:** Pydantic validation models.
* **Failure Modes:** `ManifestValidationError` if mandatory asset fields are omitted or types mismatch.

---

## Folder Structure

```text
thumbnail ai project source/
├── modules/
│   ├── __pycache__/
│   ├── comfyui_client.py
│   ├── config.py
│   ├── csv_reader.py
│   ├── image_generator.py
│   ├── models.py
│   ├── module7_exceptions.py
│   ├── prompt_compiler.py
│   ├── redesign_spec_engine.py
│   ├── thumbnail_downloader.py
│   ├── thumbnail_intelligence.py
│   ├── visual_reference_engine.py         # Module 6.5 Core Orchestrator
│   ├── workflow_library.py
│   ├── youtube_metadata.py
│   └── vre_components/                    # VRE Internal Processors
│       ├── __init__.py
│       ├── face_processor.py
│       -── segmentation_processor.py
│       ├── topology_processor.py
│       ├── asset_writer.py
│       └── manifest_builder.py
├── data/
│   └── visual_references/                 # Generated Sharded Artifacts Root
│       └── {video_id}/
│           ├── creator_face.png
│           ├── face_mask.png
│           ├── object_crop.png
│           ├── object_mask.png
│           ├── foreground.png
│           ├── background.png
│           ├── depth_map.png
│           ├── canny_map.png
│           └── reference_manifest.json
└── tests/
    └── test_visual_reference_engine.py    # Comprehensive Test Suite

```

---

## Data Models

All models must be implemented using Pydantic v2 within `modules/models.py` or imported cleanly into the engine.

### 1. `BoundingBox`

* **`x`** (int): Top-left absolute pixel X coordinate.
* **`y`** (int): Top-left absolute pixel Y coordinate.
* **`width`** (int): Bounding box horizontal span in pixels.
* **`height`** (int): Bounding box vertical span in pixels.
* *Rationale:* Provides precise rectangular isolation bounds for crops without ambiguity.

### 2. `AssetMetadata`

* **`asset_type`** (str): Identifier category (e.g., `"face_mask"`, `"depth_map"`).
* **`file_path`** (str): Absolute local filesystem path to the asset.
* **`checksum`** (str): SHA-256 cryptographic hash of the generated artifact bytes.
* **`resolution`** (Tuple[int, int]): Image dimensions expressed as `(width, height)`.
* **`confidence_score`** (Optional[float]): ML model detection confidence rating (0.0 to 1.0).
* *Rationale:* Ensures full traceability, cache invalidation verification, and downstream quality checks.

### 3. `VisualReferenceManifest`

* **`video_id`** (str): Unique string key matching YouTube or asset ingestion identifiers.
* **`source_image_path`** (str): Origin file path used for asset decomposition.
* **`source_hash`** (str): SHA-256 fingerprint of the source image used for cache validation.
* **`created_at`** (str): ISO-8601 UTC timestamp of manifest generation.
* **`assets`** (Dict[str, Optional[AssetMetadata]]): Mapped collection of all prepared reference artifacts.
* **`processing_metadata`** (Dict[str, Any]): Execution metrics, runtime duration, and processor flags.
* *Rationale:* Represents the complete immutable contract passed to Module 7.

---

## Manifest Schema

```json
{
  "video_id": "eWzsmjA1vOo",
  "source_image_path": "/path/to/thumbnail ai project source/data/thumbnails/eWzsmjA1vOo.jpg",
  "source_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "created_at": "2026-04-12T14:22:01.482Z",
  "assets": {
    "creator_face": {
      "asset_type": "creator_face",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/creator_face.png",
      "checksum": "8f434346648f6b96df89dda9b7852b855e3b0c44298fc1c149afbf4c8996fb924",
      "resolution": [512, 512],
      "confidence_score": 0.982
    },
    "face_mask": {
      "asset_type": "face_mask",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/face_mask.png",
      "checksum": "149afbf4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c44298fc",
      "resolution": [1280, 720],
      "confidence_score": 0.982
    },
    "object_crop": {
      "asset_type": "object_crop",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/object_crop.png",
      "checksum": "991b7852b855e3b0c44298fc1c149afbf4c8996fb924e3b0c44298fc1c149afbf4",
      "resolution": [640, 640],
      "confidence_score": 0.915
    },
    "object_mask": {
      "asset_type": "object_mask",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/object_mask.png",
      "checksum": "27ae41e4649b934ca495991b7852b855e3b0c44298fc1c149afbf4c8996fb9",
      "resolution": [1280, 720],
      "confidence_score": 0.915
    },
    "foreground": {
      "asset_type": "foreground",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/foreground.png",
      "checksum": "44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c",
      "resolution": [1280, 720],
      "confidence_score": 0.950
    },
    "background": {
      "asset_type": "background",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/background.png",
      "checksum": "fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c4428",
      "resolution": [1280, 720],
      "confidence_score": 0.950
    },
    "depth_map": {
      "asset_type": "depth_map",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/depth_map.png",
      "checksum": "89dda9b7852b855e3b0c44298fc1c149afbf4c8996fb9248f434346648f6b96",
      "resolution": [1280, 720],
      "confidence_score": null
    },
    "canny_map": {
      "asset_type": "canny_map",
      "file_path": "/path/to/thumbnail ai project source/data/visual_references/eWzsmjA1vOo/canny_map.png",
      "checksum": "5e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8",
      "resolution": [1280, 720],
      "confidence_score": null
    }
  },
  "processing_metadata": {
    "engine_version": "1.0.0",
    "total_duration_ms": 1142,
    "processors_executed": ["FaceProcessor", "SegmentationProcessor", "TopologyProcessor"],
    "cached_hit": false
  }
}

```

*Explanation of Manifest Fields:*

* `video_id`: Primary key correlating asset outputs across modules.
* `source_image_path`: Absolute path referencing the input image.
* `source_hash`: Cryptographic token used by the caching layer to prevent redundant re-computations.
* `created_at`: UTC timestamp recording exact generation time.
* `assets`: Dictionary containing objects for each expected asset type. If a non-mandatory asset fails detection, its value is explicitly set to `null`.
* `processing_metadata`: Operational telemetry tracking engine version, execution time in milliseconds, executed sub-processors, and cache status.

---

## Processor Interfaces

To enforce strict adherence to interface contracts, abstract base classes must be implemented using Python's `abc` module.

```python
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from modules.models import VisualReferenceManifest, AssetMetadata

class IFaceProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Detects face, extracts crop, and builds binary mask."""
        pass

class ISegmentationProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decomposes image into foreground, background, object crop, and object mask."""
        pass

class ITopologyProcessor(ABC):
    @abstractmethod
    def generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """Computes metric monocular depth map."""
        pass

    @abstractmethod
    def generate_canny_map(self, image: np.ndarray) -> np.ndarray:
        """Computes Canny structural edge map."""
        pass

class IAssetWriter(ABC):
    @abstractmethod
    def write_image(self, array: np.ndarray, destination_path: Path) -> bool:
        """Atomically persists image numpy array to disk."""
        pass

    @abstractmethod
    def purge_directory(self, target_dir: Path) -> bool:
        """Deletes target sharded directory contents."""
        pass

class IManifestBuilder(ABC):
    @abstractmethod
    def build(self, video_id: str, source_path: str, asset_paths: Dict[str, str], metadata: Dict[str, Any]) -> VisualReferenceManifest:
        """Constructs and validates the VisualReferenceManifest schema."""
        pass

```

*Rationale:* Explicit interfaces isolate underlying ML frameworks (e.g., swapping PyTorch MiDaS for TensorRT ZoeDepth requires zero changes to the core orchestrator).

---

## Cache Strategy

* **Fingerprint Mechanism:** Upon receiving a `video_id` and `source_image_path`, VRE computes a SHA-256 hash of the source image file bytes.
* **Manifest Inspection:** VRE checks whether `/data/visual_references/{video_id}/reference_manifest.json` exists on disk.
* **Integrity Verification:** If the manifest exists, it is loaded, and its `source_hash` field is compared against the newly computed hash.
* **Bypass on Match:** If hashes match and all referenced files exist on disk, VRE immediately returns the parsed manifest, bypassing all ML and CV inference pipelines.
* *Rationale:* Visual reference generation is computationally expensive. Caching eliminates redundant overhead during iterative prompt testing.

---

## Validation Strategy

* **Input Validation:** Source image paths are verified for existence and readability prior to processing. Images are checked for valid dimensions (minimum 256x256 pixels) and supported color spaces (RGB/BGR).
* **Output Validation:** Pydantic validators confirm that all generated asset files exist on disk, possess non-zero byte lengths, and match declared resolutions before saving the manifest.
* *Rationale:* Prevents downstream failure in Module 7 caused by corrupt or missing image conditioning assets.

---

## Logging Strategy

* **Framework Integration:** Utilizes Python's standard `logging` module configured with structured formatting.
* **Log Levels:**
* `INFO`: Emitted upon engine initialization, cache hits, successful manifest compilation, and pipeline completion.
* `DEBUG`: Emitted during intermediate processor steps, including bounding box coordinates, execution time per sub-processor, and file write actions.
* `WARNING`: Emitted when optional components (such as face detection) fail gracefully, allowing the pipeline to proceed without creator face data.
* `ERROR`: Emitted when mandatory files are missing, disk writes fail, or unhandled exceptions occur.


* *Rationale:* Ensures full observability in production environments without flooding log storage.

---

## Exception Hierarchy

```text
Exception
├── VREBaseError (Base VRE Exception)
    ├── SourceImageNotFoundError
    ├── FaceDetectionFailedWarning (Inherits from Warning/Exception policy)
    ├── SegmentationInferenceError
    ├── TopologyExtractionError
    ├── AssetWriteError
    └── ManifestValidationError

```

* **`VREBaseError`**: Catch-all base exception for any VRE-related failure.
* **`SourceImageNotFoundError`**: Raised when the requested thumbnail source file does not exist.
* **`SegmentationInferenceError`**: Raised when foreground extraction models fail to execute.
* **`TopologyExtractionError`**: Raised when depth or Canny operators encounter invalid array shapes.
* **`AssetWriteError`**: Raised when file persistence fails.
* **`ManifestValidationError`**: Raised when generated output dictionaries fail Pydantic model validation.

---

## Configuration

All configuration constants are declared within `modules/config.py`:

* **`VRE_STORAGE_ROOT`**: Path pointing to `/data/visual_references/`. *Rationale: Centralizes sharded asset locations.*
* **`VRE_CANNY_LOW_THRESHOLD`**: Integer (default: 100). *Rationale: Controls edge sensitivity for Canny map generation.*
* **`VRE_CANNY_HIGH_THRESHOLD`**: Integer (default: 200). *Rationale: Establishes strong edge hysteresis bounds.*
* **`VRE_FACE_DETECTION_CONFIDENCE`**: Float (default: 0.85). *Rationale: Filters out false-positive face bounding boxes.*
* **`VRE_CACHE_ENABLED`**: Boolean (default: `True`). *Rationale: Allows disabling cache checks during debugging or benchmark testing.*

---

## Testing Strategy

### Unit Tests (`tests/test_visual_reference_engine.py`)

* Test source hash computation for deterministic output.
* Test `BoundingBox` validation logic and coordinate clipping.
* Test `ManifestBuilder` behavior with valid and invalid payloads.

### Integration Tests

* Mock ML processor backends to return synthetic numpy arrays, verifying end-to-end orchestration from raw input to sharded disk persistence and valid manifest generation.
* Test cache-hit scenarios to verify that secondary calls bypass processor execution entirely.

### Contract Tests

* Verify that the output `VisualReferenceManifest` strictly satisfies the schema expected by Module 7's ComfyUI payload constructor (`image_generator.py`).

### Failure Tests

* Pass non-existent image paths to trigger `SourceImageNotFoundError`.
* Simulate read-only destination directories to verify `AssetWriteError` handling and exception propagation.

### Performance Tests

* Measure end-to-end execution latency on standard benchmark thumbnails, ensuring cached responses execute in under 15ms and cold processing remains within acceptable thresholds.

---

## Integration with Module 4

Module 4 provides raw metadata ingestion pipelines (creators, video titles, and timestamps). VRE leverages Module 4's ingested identifiers (`video_id`) to key its sharded folder structure under `data/visual_references/{video_id}/`.

---

## Integration with Module 5

Module 5 supplies thumbnail analysis metrics and aesthetic scoring data. While VRE does not directly consume analysis scores, it operates on the selected thumbnail asset paths determined during Module 5's curation phase.

---

## Integration with Module 6

Module 6 compiles prompt packages containing semantic instructions and asset pointers. VRE consumes the compiled package output, executes visual asset preparation, and injects reference paths back into the execution context.

---

## Integration with Module 7

Module 7 (`image_generator.py` and `comfyui_client.py`) consumes the `reference_manifest.json` generated by VRE. Module 7 reads the explicit file paths for depth maps, canny edges, face masks, and object crops to construct fully populated ComfyUI API JSON payloads for ControlNet and IP-Adapter nodes.

---

## Future Extensions

* **OpenPose Skeletal Estimation:** Integration of body and hand pose estimation models to generate skeleton JSON maps for advanced pose-locking ControlNets.
* **OCR Typography Masking:** Automated text detection to generate exclusion masks preventing generative corruption over existing thumbnail text elements.
* **Color Palette Extraction:** Automated k-means dominant color extraction stored within manifest metadata to guide color-matching nodes in downstream diffusion graphs.

---

## Risks

* **Model Weight Footprint:** Loading segmentation and monocular depth models into memory increases container RAM/VRAM utilization. *Mitigation:* Use lightweight ONNX runtimes and lazy-load models strictly upon processor invocation.
* **Disk Bloat:** Unmanaged sharded folders under `data/visual_references/` can accumulate large volumes of PNG artifacts over time. *Mitigation:* Implement a garbage collection utility utilizing `clean_assets(video_id)`.

---

## Open Questions

* Should VRE support dynamic resolution downscaling if source thumbnails exceed 4K dimensions, or should downscaling remain the responsibility of Module 7? *(Current decision: VRE operates on source resolution; downscaling is delegated to Module 7).*
* Should depth map generation support multiple alternative models (e.g., ZoeDepth vs. MiDaS) via runtime configuration flags? *(Current decision: Single default lightweight model with extension hooks planned for Phase 2).*

---

## Implementation Notes

* Ensure all file I/O uses atomic write patterns (writing to a temporary file in the target directory and renaming) to prevent partial JSON or image writes during worker thread preemption.
* Verify that numpy arrays passed to OpenCV writers are converted to contiguous memory blocks (`np.ascontiguousarray`) to avoid segmentation faults in C-bindings.

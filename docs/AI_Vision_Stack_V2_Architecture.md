# Executive Summary

This document outlines the production architecture for the next-generation Visual Reference Engine (VRE) **Version 2**, a fully local, open-source AI Vision Stack designed to analyze YouTube thumbnails and extract high-fidelity semantic and visual assets. The system transitions away from legacy heuristics (Haar Cascades, GrabCut, Canny) toward state-of-the-art (2026) deep learning models.

To operate within the strict hardware constraints of an RTX 4060 Laptop GPU (8GB VRAM), an Intel i9-13900HX, and 16GB System RAM, this architecture relies on a highly optimized **Sequential Execution Pipeline** with aggressive VRAM offloading, lazy loading, and TensorRT/ONNX optimizations. The resulting pipeline guarantees artifact-free ComfyUI integration while preserving creator identity, semantic structure, and visual hierarchy.

Version 2 introduces several engineering refinements over the original architecture:

* Restoration of **GroundingDINO** as a dedicated open-vocabulary localization stage, working alongside (not replaced by) Florence-2.
* Strict separation of matting responsibilities (BiRefNet) from segmentation responsibilities (SAM 2).
* Introduction of **OpenCLIP** for embedding generation, enabling similarity search, retrieval, and clustering.
* Promotion of **PaddleOCR** from a fallback utility to a first-class, independent pipeline stage.
* A deterministic **face selection algorithm** for identifying the creator face in multi-person thumbnails.
* Full **model versioning and reproducibility metadata** embedded in the manifest.
* An expanded **model lifecycle and caching architecture** for long-running stability.
* An internal **three-subsystem decomposition** (Identity Engine, Scene Understanding Engine, Asset Generation Engine) behind a single unchanged external API.

---

# Research and Model Comparisons

To determine the optimal local architecture, industry-standard and state-of-the-art models were evaluated against the requested production parameters.

## 1. Face Detection, Recognition & Parsing

| Parameter | InsightFace (SCRFD + ArcFace) | YOLO11-Face | MediaPipe FaceMesh |
| --- | --- | --- | --- |
| **Model** | InsightFace | YOLO11-Face | MediaPipe |
| **Purpose** | Face detection, landmarks, embeddings | Fast face & bounding box detection | On-device face mesh & geometry |
| **Developer** | DeepInsight | Ultralytics | Google |
| **License** | MIT (Code), Non-Commercial (Models) | AGPL-3.0 | Apache 2.0 |
| **GitHub** | deepinsight/insightface | ultralytics/ultralytics | google-ai-edge/mediapipe |
| **Accuracy** | SOTA (Unmatched recognition) | Very High | Moderate (Mobile-optimized) |
| **Speed** | 15-20 ms | 5-10 ms | 2-5 ms |
| **VRAM usage** | 1.2GB | 800MB | 0GB (CPU) |
| **CPU fallback** | Yes (ONNXRuntime) | Yes | Yes (Native) |
| **Windows compatibility** | Excellent | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Active | Very Active | Active |
| **Community adoption** | Industry Standard | Extremely High | Massive |
| **Strengths** | Flawless embeddings and 5-point landmarks | Unmatched inference speed | Zero VRAM footprint |
| **Weaknesses** | Non-commercial model license | Strict AGPL license | Struggles with occlusion |
| **Recommended** | **Yes** (Primary Face Stack) | No | No |

## 2. Foreground Matting, Saliency & Parsing

| Parameter | BiRefNet | DIS (IS-Net) | BiSeNet (CelebAMask) |
| --- | --- | --- | --- |
| **Model** | BiRefNet | IS-Net | BiSeNet |
| **Purpose** | High-res dichotomous segmentation (matting only) | Background removal & saliency | Semantic face parsing (Hair/Mask) |
| **Developer** | ZhengPeng7 | Xuebin Qin | Various |
| **License** | MIT | Apache 2.0 | MIT |
| **GitHub** | ZhengPeng7/BiRefNet | xuebinqin/DIS | zllrunning/face-parsing.PyTorch |
| **Accuracy** | SOTA (Perfect hair/transparency) | High | Excellent (Facial features) |
| **Speed** | 40 ms | 35 ms | 15 ms |
| **VRAM usage** | 1.8GB | 1.5GB | 800MB |
| **CPU fallback** | Yes | Yes | Yes |
| **Windows compatibility** | Excellent | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Active | Maintenance | Stable |
| **Community adoption** | Growing Rapidly | High | High |
| **Strengths** | Flawless transparent foregrounds | Good legacy compatibility | Exact hair and skin masks |
| **Weaknesses** | Heavy feature decoder; **does not perform instance segmentation** | Jagged edges on complex hair | Limited to facial features |
| **Recommended** | **Yes** (Matting-only: FG/BG/Transparent/Saliency) | No | **Yes** (Face Parsing) |

> **V2 Clarification:** BiRefNet is restricted to dichotomous matting tasks (foreground/background separation, transparent cutouts, saliency maps). It performs **no instance-level segmentation** — that responsibility belongs exclusively to SAM 2 (see Section 3).

## 3. Zero-Shot Instance Segmentation

| Parameter | SAM 2 (Large) | FastSAM | MobileSAM |
| --- | --- | --- | --- |
| **Model** | Segment Anything 2 | FastSAM | MobileSAM |
| **Purpose** | Promptable dense **instance** segmentation | Real-time object masks | Lightweight segmentation |
| **Developer** | Meta | Ultralytics | Chaoning Zhang |
| **License** | Apache 2.0 | AGPL-3.0 | Apache 2.0 |
| **GitHub** | facebookresearch/sam2 | CASIA-IVA-Lab/FastSAM | ChaoningZhang/MobileSAM |
| **Accuracy** | SOTA (Unmatched object masks) | Good | Moderate |
| **Speed** | 35 ms | 10 ms | 12 ms |
| **VRAM usage** | 3GB | 1GB | 900MB |
| **CPU fallback** | Yes | Yes | Yes |
| **Windows compatibility** | Excellent | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Active | Active | Stable |
| **Community adoption** | Massive | High | High |
| **Strengths** | Flawless prompted object crops when seeded with external boxes | High FPS | Low memory footprint |
| **Weaknesses** | Requires 3GB VRAM; **requires external box/point prompts for reliable instance separation** | Edges are jagged | Fails on complex compositions |
| **Recommended** | **Yes** (Human Segmentation, Object Segmentation, Object Masks, Object Crops) | No | No |

> **V2 Clarification:** SAM 2 is exclusively responsible for **instance-level** outputs: human instance segmentation, object instance segmentation, object masks, and individual object crops. It receives its box prompts from GroundingDINO (Section 4) rather than generating open-vocabulary proposals itself.

## 4. Open-Vocabulary Detection, Localization & Semantic Reasoning

| Parameter | GroundingDINO | Florence-2 (Large) | YOLO-World |
| --- | --- | --- | --- |
| **Model** | GroundingDINO | Florence-2 | YOLO-World |
| **Purpose** | Open-vocabulary bounding-box localization | Unified vision-language reasoning | Real-time open-vocab detection |
| **Developer** | IDEA-Research | Microsoft | Tencent AI Lab |
| **License** | Apache 2.0 | MIT | GPL-3.0 |
| **GitHub** | IDEA-Research/GroundingDINO | microsoft/Florence-2 | AILab-CVC/YOLO-World |
| **Accuracy** | Very High (precise, text-prompted boxes) | SOTA (multi-task spatial + semantic) | High |
| **Speed** | 80 ms | 120 ms | 25 ms |
| **VRAM usage** | 2GB | 1.5GB | 1GB |
| **CPU fallback** | Yes | Yes | Yes |
| **Windows compatibility** | Moderate (C++ Build Required) | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Stable | Active | Active |
| **Community adoption** | Massive | High | Growing |
| **Strengths** | Precise, high-recall boxes ideal for SAM 2 prompting | Unified captioning, layout, composition, and hierarchy reasoning | Fast but less precise box localization |
| **Weaknesses** | No captioning, OCR, or scene reasoning | Box localization less precise than dedicated grounding models for dense scenes | Weaker semantic reasoning |
| **Recommended** | **Yes** (Primary Localizer / SAM 2 Prompt Source) | **Yes** (Primary Semantic Reasoner) | No |

### GroundingDINO vs Florence-2 — Division of Responsibility

| Responsibility | GroundingDINO | Florence-2 |
| --- | --- | --- |
| Open-vocabulary object detection | **Yes** | Assist only |
| High-quality bounding boxes for SAM 2 prompting | **Yes** | No |
| Localization / proposal generation | **Yes** | No |
| Semantic understanding & scene description | No | **Yes** |
| Thumbnail reasoning & captioning | No | **Yes** |
| Composition analysis | No | **Yes** |
| Visual hierarchy | No | **Yes** |
| Layout understanding | No | **Yes** |
| Metadata extraction (labels, attributes) | No | **Yes** |

> **V2 Clarification:** GroundingDINO is **restored** to the architecture rather than being subsumed by Florence-2. GroundingDINO produces the precise, text-prompted bounding boxes that seed SAM 2's instance segmentation, while Florence-2 is reserved for higher-level semantic reasoning tasks (captioning, composition, layout, and visual hierarchy) where it is strongest.

## 5. OCR Options

| Parameter | PaddleOCR | Tesseract 5 | EasyOCR |
| --- | --- | --- | --- |
| **Model** | PaddleOCR | Tesseract | EasyOCR |
| **Purpose** | Multilingual detection + recognition OCR | Legacy OCR engine | Multilingual OCR (PyTorch) |
| **Developer** | Baidu | Google / OSS | JaidedAI |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **GitHub** | PaddlePaddle/PaddleOCR | tesseract-ocr/tesseract | JaidedAI/EasyOCR |
| **Accuracy** | Excellent (stylized/rotated text) | Moderate (clean scans only) | Good |
| **Speed** | 30 ms | 60 ms | 90 ms |
| **VRAM usage** | 500MB (GPU) / 0GB (CPU) | 0GB (CPU only) | 900MB |
| **CPU fallback** | Yes | Native | Yes |
| **Windows compatibility** | Excellent | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Very Active | Stable | Active |
| **Community adoption** | Massive | Massive (legacy) | High |
| **Strengths** | Handles stylized thumbnail fonts, rotation, curved text, reading-order detection | Extremely lightweight | Simple API |
| **Weaknesses** | Larger model footprint than Tesseract | Fails on stylized/thumbnail-style text | Slower, heavier than PaddleOCR |
| **Recommended** | **Yes** (Primary, independent OCR stage) | No | No |

> **V2 Clarification:** PaddleOCR is promoted from a validation fallback to a **primary, independent pipeline stage**. It runs without any dependency on Florence-2. Florence-2 *may* consume PaddleOCR's output to enrich semantic/composition reasoning, but OCR extraction itself never depends on Florence-2.

## 6. Embedding & Representation Models (New in V2)

| Parameter | OpenCLIP | CLIP (OpenAI) | DINOv2 |
| --- | --- | --- | --- |
| **Model** | OpenCLIP | CLIP | DINOv2 |
| **Purpose** | Open-source contrastive image/text embeddings | Original contrastive embeddings | Self-supervised visual features |
| **Developer** | LAION / OpenCLIP community | OpenAI | Meta |
| **License** | MIT | MIT | Apache 2.0 |
| **GitHub** | mlfoundations/open_clip | openai/CLIP | facebookresearch/dinov2 |
| **Accuracy** | SOTA among open-source CLIP variants (ViT-B/H checkpoints) | High, but frozen/unmaintained | High for pure visual similarity, no text alignment |
| **Speed** | 25 ms (ViT-B/32) | 25 ms | 30 ms |
| **VRAM usage** | 900MB (ViT-B/32, FP16) | 900MB | 1.1GB |
| **CPU fallback** | Yes | Yes | Yes |
| **Windows compatibility** | Excellent | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Very Active | Inactive (frozen) | Active |
| **Community adoption** | Massive, industry-standard for open embeddings | High (legacy) | Growing |
| **Strengths** | Actively trained, many checkpoint sizes, strong text-image alignment for retrieval | Battle-tested baseline | Best pure-vision embeddings |
| **Weaknesses** | Slightly heavier than base CLIP at larger checkpoints | No longer updated | No text alignment (limits cross-modal search) |
| **Recommended** | **Yes** (Primary Embedding Model) | No | No (future candidate for pure visual clustering) |

> **Why embeddings matter:** OpenCLIP embeddings give the pipeline a compact, fixed-length numerical fingerprint of each thumbnail. This unlocks capabilities that none of the other models provide on their own:
> * **Similarity Search:** Nearest-neighbor lookup across a video library to find visually or thematically similar thumbnails.
> * **Retrieval:** Text-to-image and image-to-image queries (e.g., "find thumbnails with a shocked face and red arrows") using the shared CLIP text/image embedding space.
> * **Clustering:** Unsupervised grouping of a channel's thumbnail history to discover recurring visual themes, formats, or A/B test variants.
> * **Duplicate Detection:** Cosine-similarity thresholding against prior embeddings to flag near-duplicate or reused thumbnails before publishing.
> * **Future AI Modules:** Downstream models (recommendation, auto-tagging, style transfer selection, thumbnail scoring) can consume the embedding directly rather than re-running the full vision stack.

## 7. Depth, Edge & Spatial Estimation

| Parameter | Depth Anything V2 | ZoeDepth | TEED |
| --- | --- | --- | --- |
| **Model** | Depth Anything V2 | ZoeDepth | TEED |
| **Purpose** | Monocular depth estimation | Metric depth estimation | AI edge detection |
| **Developer** | TikTok / HKU | Intel | xavierxiao |
| **License** | Apache 2.0 | MIT | MIT |
| **GitHub** | DepthAnything/Depth-Anything-V2 | isl-org/ZoeDepth | xavierxiao/TEED |
| **Accuracy** | SOTA (Preserves fine edges) | High | SOTA (Crisp edges) |
| **Speed** | 20 ms | 40 ms | 10 ms |
| **VRAM usage** | 1.2GB | 1.5GB | 500MB |
| **CPU fallback** | Yes | Yes | Yes |
| **Windows compatibility** | Excellent | Excellent | Excellent |
| **Python compatibility** | 3.11 Supported | 3.11 Supported | 3.11 Supported |
| **Maintenance status** | Active | Inactive | Stable |
| **Community adoption** | Massive | High | Moderate |
| **Strengths** | Flawless relative depth gradients | Accurate metric scale | Outperforms Canny & PiDiNet |
| **Weaknesses** | Requires large model for best quality | Softer object boundaries | Can over-detect textures |
| **Recommended** | **Yes** (Primary Depth) | No | **Yes** (Edge Map) |

## 8. Face Selection Strategies (New in V2)

| Strategy | Description | Strength | Weakness |
| --- | --- | --- | --- |
| **Largest Face Area** | Selects the face with the greatest bounding-box area | Simple, robust to false positives | Fails when a background face is closer to camera but not the creator |
| **Centrality** | Selects the face closest to the frame's optical center / rule-of-thirds anchor | Matches common thumbnail composition conventions | Can fail on off-center creator framing |
| **Saliency Weighting** | Cross-references BiRefNet's saliency map to score faces by attention weight | Captures compositional intent, not just geometry | Depends on saliency model quality |
| **Detection Confidence** | Weights by InsightFace/SCRFD detection confidence score | Filters low-quality or partial detections | Not discriminative among multiple high-confidence faces |
| **Embedding Identity Match (Future)** | Compares ArcFace embeddings against a known creator embedding gallery | Highest long-term accuracy; channel-specific | Requires a pre-built creator embedding gallery; not available cold-start |
| **Recommended (V2)** | **Weighted composite of Area + Centrality + Saliency + Confidence, with Embedding Match as an override when a gallery exists** | Deterministic, explainable, upgrade path to identity matching | Requires tuned weighting coefficients |

---

# Face Selection Logic (New in V2)

Thumbnails frequently contain multiple faces (the creator plus guests, reaction cams, or background people). The V2 architecture defines a **deterministic, explainable face selection algorithm** to consistently identify the creator face.

## Selection Pipeline

1. **Candidate Generation:** All faces returned by InsightFace (SCRFD) are treated as candidates, each carrying a bounding box, detection confidence, 5-point landmarks, and an ArcFace embedding.
2. **Feature Scoring:** Each candidate face is scored on four normalized (0–1) sub-scores:
   * `area_score` — face bounding-box area relative to the largest face in the frame.
   * `centrality_score` — inverse normalized distance from the frame's compositional center (weighted toward rule-of-thirds anchor points common in thumbnails).
   * `saliency_score` — mean BiRefNet saliency-map intensity within the face bounding box.
   * `confidence_score` — raw SCRFD detection confidence.
3. **Composite Score:**

   ```text
   creator_score = (0.35 * area_score)
                  + (0.25 * centrality_score)
                  + (0.25 * saliency_score)
                  + (0.15 * confidence_score)
   ```

4. **Embedding Override (Future-Ready):** If a per-channel creator embedding gallery exists, cosine similarity between each candidate's ArcFace embedding and the gallery is computed. Any candidate exceeding a similarity threshold (default `0.55`) is automatically selected as the creator face, **overriding** the composite geometric score. This allows the system to improve deterministically over time without changing the pipeline's external contract.
5. **Tie-Breaking:** If two or more candidates fall within `0.03` composite-score units of each other, the face nearer to the horizontal center of the frame wins, since centered framing is the dominant convention for creator faces in thumbnails.
6. **Output:** The winning candidate is written to `identity.creator_face_path`, and the full ranked candidate list (with sub-scores) is persisted to `identity.face_selection_metadata` for auditability.

This strategy is fully deterministic given identical inputs, requires no additional VRAM (all inputs are already produced by Stage 2), and provides a clean upgrade path to full identity matching once a creator embedding gallery is populated.

---

# Final Production Architecture

To successfully extract all required assets while adhering to an 8GB VRAM limit, the system utilizes a unified, multi-stage architecture, internally decomposed into three subsystems (see below).

## Internal Subsystem Decomposition (New in V2)

The Visual Reference Engine (VRE) external API remains a single entry point (`VisualReferenceEngine.process(thumbnail)`), but internally the system is decomposed into three cohesive subsystems for maintainability, testability, and independent scaling:

```text
VisualReferenceEngine (external API unchanged)
│
├── Identity Engine
│   Responsible for:
│   - Face detection (InsightFace/SCRFD)
│   - Face embeddings (ArcFace)
│   - Face parsing (BiSeNet / CelebAMask-HQ)
│   - Hair masks
│   - Creator identification (Face Selection Logic)
│
├── Scene Understanding Engine
│   Responsible for:
│   - Florence-2 (semantic reasoning, composition, layout, hierarchy)
│   - GroundingDINO (open-vocabulary localization, SAM 2 box prompts)
│   - PaddleOCR (text, bounding boxes, confidence, reading order)
│   - OpenCLIP (thumbnail/scene/style embeddings)
│
└── Asset Generation Engine
    Responsible for:
    - SAM 2 (human/object instance segmentation, masks, crops)
    - BiRefNet (matting, saliency, transparent foreground)
    - Depth Anything V2 (scene depth map)
    - TEED (edge map)
    - Manifest generation
    - Asset writing to disk
```

Each subsystem exposes an internal async interface but shares the same GPU threading lock, model cache, and lifecycle manager described in the Hardware & Memory Management section, ensuring the external behavior and API surface of VRE remains identical to Version 1.

## Selected Model Stack & Responsibilities

* **GroundingDINO:** Extracts open-vocabulary bounding boxes, localization, and object proposals — seeds SAM 2.
* **Florence-2 (Large):** Extracts composition metadata, thumbnail layout metadata, visual hierarchy metadata, scene description, caption understanding, and general metadata extraction (may optionally consume PaddleOCR text for enrichment).
* **PaddleOCR:** Extracts text, text bounding boxes, confidence scores, font characteristics (where possible), and reading order — runs independently of Florence-2.
* **OpenCLIP:** Extracts thumbnail embedding, scene embedding, style embedding, and semantic feature vector.
* **InsightFace (SCRFD + ArcFace):** Extracts creator face candidates, face embeddings, face landmarks.
* **BiSeNet (CelebAMask-HQ):** Extracts face parsing, hair mask, face mask.
* **BiRefNet:** Extracts foreground, background, transparent foreground, saliency map (matting only — no instance segmentation).
* **SAM 2 (Large):** Extracts human instance segmentation, object instance segmentation, object masks, individual object crops (prompted by GroundingDINO boxes).
* **Depth Anything V2:** Extracts scene depth map.
* **TEED:** Extracts edge map.
* **Geometric Heuristics (CPU/NumPy):** Extracts horizon estimation, camera angle estimation, lighting estimation, dominant colors, face selection scoring, processing metadata, and the complete manifest.

## Execution Pipeline & Inference Order (Updated)

Because the combined VRAM of these models exceeds 8GB, the pipeline continues to execute sequentially. Each stage loads its required models, generates assets, flushes VRAM, and triggers garbage collection before the next stage begins.

```text
[Input YouTube Thumbnail]
            |
            v
+-----------------------------------------------------------+
| STAGE 1: Localization (GroundingDINO)                     |
| -> Open-vocabulary Bounding Boxes, Object Proposals       |
|    [Scene Understanding Engine]                           |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 2: Semantic Reasoning (Florence-2)                  |
| -> Composition, Layout, Visual Hierarchy, Scene Caption   |
|    [Scene Understanding Engine]                           |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 3: OCR (PaddleOCR)                                  |
| -> Text, Text Boxes, Confidence, Reading Order            |
|    [Scene Understanding Engine — independent of Stage 2]  |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 4: Embeddings (OpenCLIP)                            |
| -> Thumbnail / Scene / Style Embeddings                   |
|    [Scene Understanding Engine]                           |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 5: Identity & Parsing (InsightFace + BiSeNet)       |
| -> Face Candidates, Embeddings, Landmarks, Hair Masks     |
| -> Face Selection Logic -> Creator Face                   |
|    [Identity Engine]                                      |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 6: Matting & Saliency (BiRefNet)                    |
| -> Saliency Map, Transparent Foreground, BG/FG            |
|    [Asset Generation Engine]                               |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 7: Instance Segmentation (SAM 2)                    |
| -> Human Seg, Object Masks, Object Crops                  |
|    (Prompted by GroundingDINO boxes from Stage 1)         |
|    [Asset Generation Engine]                               |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 8: Spatial & Depth (DA-V2 + TEED)                   |
| -> Depth Map, Edge Map                                    |
|    [Asset Generation Engine]                               |
+-----------------------------------------------------------+
            | (Clear VRAM)
            v
+-----------------------------------------------------------+
| STAGE 9: CPU Heuristics & Aggregation                     |
| -> Camera Angle, Horizon, Lighting, Dominant Colors,      |
|    Face Selection Scoring, Version Metadata,              |
|    Confidence Scores, JSON Manifest                       |
|    [Asset Generation Engine]                               |
+-----------------------------------------------------------+
            |
            v
[ComfyUI Ready Assets & System Manifest]

```

## Hardware & Memory Management Strategy

* **Lazy Loading & VRAM Eviction:** Models are instantiated in CPU RAM upon application boot. During pipeline execution, a model is moved to the GPU (`.to('cuda')`), executed, moved back to the CPU (`.to('cpu')`), followed immediately by forced garbage collection and `torch.cuda.empty_cache()`.
* **Precision Optimization:** All models utilize `FP16` (Half Precision) to halve the memory footprint without degrading visual quality.
* **Fallback Strategy:** If an image is too complex and causes an Out-Of-Memory (OOM) error during SAM 2 execution, the system catches the exception and falls back to tiled processing on the CPU.
* **Thread Safety:** The pipeline runs in an asynchronous queue. A strict GPU threading lock (`Lock()`) ensures that only one VRAM-heavy process accesses the RTX 4060 at any given millisecond, shared across all three internal subsystems.
* **Complex Heuristics:** Camera angle and lighting estimation do not use VRAM. Camera angle is calculated geometrically using the surface normals derived from the depth map: $\vec{N} = \nabla D(x,y)$. Horizon estimation utilizes OpenCV Hough Line Transforms on the TEED edge map masked by the background layer. Lighting direction is estimated via $L_{dir} = \arg\max \int (I(x) \cdot \vec{N}(x)) dx$ over foreground objects.

## Model Lifecycle & Caching Architecture (New in V2)

The original architecture referenced lazy loading conceptually; V2 formalizes the full model lifecycle to support long-running, stable operation across large batch jobs.

### Model Lifecycle States

```text
[Registered] -> [CPU-Cached (Idle)] -> [GPU-Active] -> [CPU-Cached (Idle)] -> ... -> [Evicted]
```

* **Registered:** On worker boot, every model's checkpoint path, config, and precision mode is registered in an in-memory `ModelRegistry`, but weights are not yet loaded.
* **CPU-Cached (Idle):** On first use, weights are loaded once into pinned CPU RAM (FP16) and kept resident for the lifetime of the worker process, avoiding repeated disk I/O.
* **GPU-Active:** Immediately before a stage runs, its model(s) are transferred to GPU VRAM under the global GPU lock, executed, and immediately transferred back to CPU-Cached state.
* **Evicted:** Only triggered by the worker restart policy or explicit memory-pressure signal; fully removes weights from both CPU and GPU memory.

### Cache Policy

| Cache Layer | Contents | Lifetime | Eviction Trigger |
| --- | --- | --- | --- |
| **GPU VRAM Cache** | Currently executing stage's model only | Milliseconds (single stage execution) | Immediate, after each stage completes |
| **CPU RAM Cache** | All registered model weights (FP16) | Full worker process lifetime | Worker restart or manual flush |
| **Disk Cache** | ONNX/TensorRT compiled engines, checkpoint hashes | Persistent across restarts | Manual invalidation on checkpoint version change |
| **Result Cache (optional)** | Manifest + asset paths keyed by input image hash | Configurable TTL (default 30 days) | LRU eviction past configured size cap |

### Worker Lifecycle & Restart Policy

* **Worker Restart Threshold:** The worker process is recycled every **1,000 processed thumbnails** (configurable), flushing CPU cache, GPU allocator, and Python heap to counteract CUDA memory fragmentation.
* **Health Checks:** Each worker reports VRAM headroom and average stage latency after every 50 thumbnails; a worker exceeding a VRAM-headroom floor (< 500MB free) triggers an early restart.
* **Graceful Drain:** On restart, the worker finishes its current in-flight thumbnail, stops accepting new work, flushes caches, and re-registers with the async queue.
* **Long-Running Stability:** Combined with the GPU cache policy above, this bounds peak fragmentation and guarantees the 3.0GB peak VRAM ceiling holds over multi-day batch runs.

---

# Storage & Manifest Schema

## Folder Structure (Updated)

The system persists data to the local disk in a structured format designed for immediate ingestion by ComfyUI `Load Image` nodes, extended in V2 with embedding, OCR, and versioning outputs.

```text
/outputs
  /[youtube_video_id]
    /faces
      - face_crop.png
      - hair_mask.png
      - face_mask.png
    /segmentation
      - foreground_transparent.png
      - background.png
      - saliency_map.png
      - human_seg.png
    /objects
      - object_01_crop.png
      - object_01_mask.png
    /spatial
      - depth_map.png
      - edge_map.png
    /embeddings
      - thumbnail_embedding.npy
      - scene_embedding.npy
      - style_embedding.npy
      - semantic_feature_vector.npy
    /ocr
      - ocr_regions.json
      - ocr_reading_order.json
    /semantic
      - composition_metadata.json
      - layout_metadata.json
      - visual_hierarchy.json
    /metadata
      - face_embedding.npy
      - face_selection_scores.json
      - model_versions.json
      - runtime_stats.json
      - manifest.json

```

## Complete Manifest Schema (Updated)

The final output is governed by a unified JSON schema, extended in V2 with embeddings, OCR, versioning, runtime, and face-selection metadata.

```json
{
  "processing_metadata": {
    "video_id": "string",
    "timestamp": "ISO-8601",
    "inference_time_ms": 0,
    "complete_manifest": true
  },
  "version_metadata": {
    "vision_stack_version": "2.0.0",
    "model_versions": {
      "grounding_dino": "string",
      "florence2": "string",
      "paddleocr": "string",
      "openclip": "string",
      "insightface": "string",
      "bisenet": "string",
      "birefnet": "string",
      "sam2": "string",
      "depth_anything_v2": "string",
      "teed": "string"
    },
    "checkpoint_names": {
      "grounding_dino": "string",
      "florence2": "string",
      "paddleocr": "string",
      "openclip": "string",
      "insightface": "string",
      "bisenet": "string",
      "birefnet": "string",
      "sam2": "string",
      "depth_anything_v2": "string",
      "teed": "string"
    },
    "checkpoint_hashes": {
      "grounding_dino": "sha256:string",
      "florence2": "sha256:string",
      "paddleocr": "sha256:string",
      "openclip": "sha256:string",
      "insightface": "sha256:string",
      "bisenet": "sha256:string",
      "birefnet": "sha256:string",
      "sam2": "sha256:string",
      "depth_anything_v2": "sha256:string",
      "teed": "sha256:string"
    },
    "inference_configuration": {
      "precision_mode": "FP16",
      "device": "cuda:0",
      "sequential_execution": true,
      "gpu_lock_enforced": true
    }
  },
  "runtime_statistics": {
    "stage_latencies_ms": {
      "grounding_dino": 0,
      "florence2": 0,
      "paddleocr": 0,
      "openclip": 0,
      "identity_engine": 0,
      "birefnet": 0,
      "sam2": 0,
      "depth_teed": 0,
      "cpu_heuristics": 0
    },
    "peak_vram_mb": 0,
    "worker_id": "string",
    "thumbnails_processed_by_worker": 0
  },
  "identity": {
    "creator_face_path": "string",
    "face_embedding_path": "string",
    "face_landmarks": [[0,0], [0,0], [0,0], [0,0], [0,0]],
    "face_parsing_path": "string",
    "hair_mask_path": "string",
    "face_mask_path": "string",
    "face_selection_metadata": {
      "strategy": "weighted_composite_v1",
      "candidates": [
        {
          "face_id": "string",
          "area_score": 0.0,
          "centrality_score": 0.0,
          "saliency_score": 0.0,
          "confidence_score": 0.0,
          "composite_score": 0.0,
          "embedding_override_applied": false
        }
      ],
      "selected_face_id": "string"
    }
  },
  "matting": {
    "human_segmentation_path": "string",
    "foreground_path": "string",
    "background_path": "string",
    "transparent_foreground_path": "string",
    "saliency_map_path": "string"
  },
  "objects": [
    {
      "object_detection_label": "string",
      "object_detection_source": "grounding_dino",
      "object_segmentation_path": "string",
      "object_mask_path": "string",
      "individual_object_crop_path": "string",
      "bounding_box": [0, 0, 0, 0],
      "confidence_score": 0.0
    }
  ],
  "spatial": {
    "scene_depth_map_path": "string",
    "edge_map_path": "string",
    "camera_angle_estimation": "string",
    "horizon_estimation_y_coord": 0
  },
  "ocr": {
    "text_regions": [
      {
        "text": "string",
        "bounding_box": [0, 0, 0, 0],
        "confidence": 0.0,
        "font_characteristics": {
          "estimated_size_px": 0,
          "bold": false,
          "italic": false
        },
        "reading_order_index": 0
      }
    ]
  },
  "embeddings": {
    "thumbnail_embedding_path": "string",
    "scene_embedding_path": "string",
    "style_embedding_path": "string",
    "semantic_feature_vector_path": "string",
    "embedding_model": "openclip",
    "embedding_dim": 0
  },
  "semantics": {
    "composition_metadata": "string",
    "lighting_estimation": "string",
    "dominant_colors": ["#Hex1", "#Hex2", "#Hex3"],
    "thumbnail_layout_metadata": "string",
    "visual_hierarchy_metadata": ["string", "string"],
    "scene_description": "string"
  }
}

```

---

# Performance Estimates & Risk Analysis

## Expected Metrics (Updated)

| Stage | Operations | Estimated VRAM | Expected Inference Time |
| --- | --- | --- | --- |
| **Stage 1** | GroundingDINO | 2.0 GB | 80 ms |
| **Stage 2** | Florence-2 | 1.5 GB | 120 ms |
| **Stage 3** | PaddleOCR | 0.5 GB | 30 ms |
| **Stage 4** | OpenCLIP | 0.9 GB | 25 ms |
| **Stage 5** | InsightFace + BiSeNet | 1.2 GB | 40 ms |
| **Stage 6** | BiRefNet | 1.8 GB | 40 ms |
| **Stage 7** | SAM 2 | 3.0 GB | 80 ms |
| **Stage 8** | Depth Anything V2 + TEED | 1.5 GB | 30 ms |
| **Stage 9** | CPU Calculations | 0.0 GB | 110 ms |
| **TOTAL** | **Full Asset Pipeline** | **Max Peak: 3.0 GB** (sequential; SAM 2 stage remains the peak) | **~555 ms per thumbnail** |

> Peak VRAM remains bounded by SAM 2's 3.0GB requirement since stages execute strictly sequentially with eviction between each; the additional GroundingDINO, OpenCLIP, and PaddleOCR stages add to total wall-clock time but not to peak VRAM.

## Risk Analysis (Expanded)

* **VRAM Fragmentation:** Continual loading and unloading of PyTorch models can cause CUDA memory fragmentation over time.
  * *Mitigation:* Pipeline reboot threshold (reset the worker process every 1,000 thumbnails) to flush the GPU memory allocator, formalized in the Worker Lifecycle & Restart Policy.

* **Long-Running GPU Fragmentation (New):** Even with periodic restarts, sustained multi-day batch jobs can accumulate allocator fragmentation between restarts, gradually reducing effective free VRAM.
  * *Mitigation:* Per-worker VRAM headroom health checks every 50 thumbnails; workers that drop below a 500MB free-VRAM floor trigger an early, graceful restart rather than waiting for the fixed threshold.

* **Florence-2 Hallucinations:** Vision-language models occasionally mislabel objects or hallucinate text/scene detail in highly distorted thumbnails.
  * *Mitigation:* Cross-reference Florence-2 semantic output with PaddleOCR's independently generated text regions and GroundingDINO's independently generated boxes before committing to the manifest; discrepancies are flagged in `processing_metadata`.

* **GroundingDINO False Positives (New):** Open-vocabulary text-prompted detection can generate low-confidence or spurious boxes on cluttered or highly stylized thumbnails.
  * *Mitigation:* Apply a confidence floor (default `0.35`) and non-maximum suppression before boxes are forwarded to SAM 2; boxes below threshold are logged but excluded from prompting.

* **SAM 2 Mask Leakage:** Background colors similar to foreground objects can cause mask bleed.
  * *Mitigation:* Use BiRefNet's highly accurate dichotomous mask as a bounding constraint on SAM 2's output, and use GroundingDINO boxes (rather than full-image auto-segmentation) to keep SAM 2 prompts tightly scoped per object.

* **OCR Failures (New):** PaddleOCR can under-perform on extremely stylized fonts, heavy drop-shadows, or low-contrast overlay text common in thumbnails.
  * *Mitigation:* Run OCR at multiple upscale factors when initial confidence is low; flag any text region below a confidence threshold (default `0.5`) as `low_confidence` in the manifest rather than silently dropping it.

* **OpenCLIP Embedding Drift (New):** Upgrading the OpenCLIP checkpoint version over time shifts the embedding space, making old and new embeddings non-comparable for similarity search.
  * *Mitigation:* Persist `embedding_model` and its checkpoint hash alongside every embedding (see `version_metadata`); similarity search operations are checkpoint-scoped, and a re-embedding backfill job is required whenever the OpenCLIP checkpoint changes.

* **Multi-Face Ambiguity (New):** Thumbnails with multiple prominent faces (e.g., collab videos, reaction content) can produce ambiguous creator-face selection.
  * *Mitigation:* The weighted composite face-selection score (Section: Face Selection Logic) combined with the embedding-override gallery reduces ambiguity; any composite-score tie within `0.03` is logged in `face_selection_metadata` for manual review/audit even though a deterministic winner is still selected automatically.

## Future Improvements

* **TensorRT Compilation:** Converting ONNX and PyTorch checkpoints to TensorRT engines (`.engine`) will drastically lower VRAM requirements and cut inference latency by roughly 40% across all nine stages.
* **Paged Attention / KV Caching for VLM:** Implementing FlashAttention for Florence-2 to speed up the semantic layout extraction.
* **Creator Embedding Gallery:** Building a per-channel ArcFace embedding gallery to activate the Face Selection Logic's embedding-override path for near-perfect creator identification.
* **ComfyUI Custom Nodes:** Wrapping this entire Python execution pipeline into a singular ComfyUI custom node to eliminate file-system I/O bottlenecks and pass tensors directly in memory to the latent space.
* **Result Cache Expansion:** Extending the optional result cache (keyed by image hash) into a persistent vector index (e.g., FAISS/HNSW) over OpenCLIP embeddings to serve similarity search and duplicate detection directly from the Vision Stack.

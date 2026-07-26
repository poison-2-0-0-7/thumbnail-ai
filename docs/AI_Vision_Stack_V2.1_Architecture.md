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

* **GroundingDINO** (`groundingdino_swint_ogc.pth`): Extracts open-vocabulary bounding boxes, localization, and object proposals — seeds SAM 2.
* **Florence-2 (Large)** (`Florence-2-large`): Extracts composition metadata, thumbnail layout metadata, visual hierarchy metadata, scene description, caption understanding, and general metadata extraction (may optionally consume PaddleOCR text for enrichment).
* **PaddleOCR** (`PP-OCRv5_server_det` + `PP-OCRv5_server_rec`): Extracts text, text bounding boxes, confidence scores, font characteristics (where possible), and reading order — runs independently of Florence-2.
* **OpenCLIP** (`ViT-B-32 / laion2b_s34b_b79k`): Extracts thumbnail embedding, scene embedding, style embedding, and semantic feature vector.
* **InsightFace (SCRFD + ArcFace)** (`buffalo_l`): Extracts creator face candidates, face embeddings, face landmarks.
* **BiSeNet (CelebAMask-HQ)** (`79999_iter.pth`): Extracts face parsing, hair mask, face mask.
* **BiRefNet** (`BiRefNet-general-epoch_244.pth`): Extracts foreground, background, transparent foreground, saliency map (matting only — no instance segmentation).
* **SAM 2 (Large)** (`sam2.1_hiera_large.pt`): Extracts human instance segmentation, object instance segmentation, object masks, individual object crops (prompted by GroundingDINO boxes).
* **Depth Anything V2** (`depth_anything_v2_vitb.pth`, Base): Extracts scene depth map.
* **TEED** (`7_model.pth`, BIPED+BRIND): Extracts edge map.
* **Geometric Heuristics (CPU/NumPy):** Extracts horizon estimation, camera angle estimation, lighting estimation, dominant colors, face selection scoring, processing metadata, and the complete manifest.

> Exact checkpoint names, sizes, licenses, and backend recommendations for every model above are specified in full in the **Model Checkpoint Specifications** section.

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

# Model Checkpoint Specifications

This section pins every model in the stack to exactly one recommended production checkpoint, so implementation teams have zero ambiguity about what to download and load. Unless noted otherwise, all checkpoints are loaded in **FP16** on `cuda:0` per the Hardware & Memory Management Strategy.

## GroundingDINO

| Field | Value |
| --- | --- |
| **Checkpoint** | `groundingdino_swint_ogc.pth` |
| **Variant** | Swin-T (Tiny) backbone, OGC training mix (Objects365 + GoldG + Cap4M) |
| **Config** | `GroundingDINO_SwinT_OGC.py` |
| **Model size** | ~694 MB (~172M params) |
| **Download source** | `IDEA-Research/GroundingDINO` GitHub releases, mirrored at `ShilongLiu/GroundingDINO` on Hugging Face |
| **License** | Apache 2.0 |
| **Preferred inference backend** | Native PyTorch (`groundingdino-py`); community ONNX exports exist but are not officially maintained — do not use for production without independent validation |
| **Expected VRAM** | ~2.0 GB (FP16) |
| **Expected inference speed** | ~80 ms per thumbnail on an RTX 4060-class GPU |

## SAM 2

| Field | Value |
| --- | --- |
| **Checkpoint** | `sam2.1_hiera_large.pt` |
| **Variant** | SAM 2.1, Hiera-Large image encoder |
| **Config** | `sam2.1_hiera_l.yaml` |
| **Model size** | ~897 MB (~224M params) |
| **Download source** | `facebookresearch/sam2` GitHub releases, mirrored at `facebook/sam2.1-hiera-large` on Hugging Face |
| **License** | Apache 2.0 |
| **Preferred inference backend** | Native PyTorch (`SAM2ImagePredictor`); TensorRT export path is experimental upstream — track for the V2 TensorRT future-improvement item before promoting to production |
| **Expected VRAM** | ~3.0 GB (FP16) — this is the architecture's peak-VRAM stage |
| **Expected inference speed** | ~35 ms per prompted object on an RTX 4060-class GPU |

## Florence-2

| Field | Value |
| --- | --- |
| **Checkpoint** | `Florence-2-large` |
| **Variant** | 0.77B parameter unified vision-language checkpoint |
| **Model size** | ~1.5 GB (FP16 safetensors) |
| **Download source** | `microsoft/Florence-2-large` on Hugging Face |
| **License** | MIT |
| **Preferred inference backend** | Hugging Face `transformers` (`AutoModelForCausalLM`, `trust_remote_code=True`); community ONNX/OpenVINO exports exist for future optimization |
| **Expected VRAM** | ~1.5 GB (FP16) |
| **Expected inference speed** | ~120 ms per thumbnail on an RTX 4060-class GPU |

## OpenCLIP

| Field | Value |
| --- | --- |
| **Checkpoint** | `ViT-B-32`, pretrained tag `laion2b_s34b_b79k` |
| **Variant** | ViT-B/32 image/text encoder trained on LAION-2B |
| **Model size** | ~600 MB (FP32 source weights; ~300 MB when loaded FP16) |
| **Download source** | `open_clip` package auto-download, mirrored at `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` on Hugging Face |
| **License** | MIT (code and weights) |
| **Preferred inference backend** | `open_clip` PyTorch; ONNX export supported for future TensorRT conversion |
| **Expected VRAM** | ~0.9 GB (FP16) |
| **Expected inference speed** | ~25 ms per thumbnail on an RTX 4060-class GPU |

## InsightFace

| Field | Value |
| --- | --- |
| **Checkpoint** | `buffalo_l` model pack (`det_10g.onnx` detector + `w600k_r50.onnx` recognition, plus landmark/gender-age sub-models) |
| **Variant** | SCRFD-10GF detector + ResNet50 ArcFace recognition head |
| **Model size** | ~326 MB total pack |
| **Download source** | InsightFace model zoo via the `insightface` Python package's automatic downloader (backed by GitHub/OSS mirrors) |
| **License** | MIT (code); **non-commercial** for the pretrained weights — flag for legal review if this product is ever monetized |
| **Preferred inference backend** | ONNX Runtime (`onnxruntime-gpu`, CUDA execution provider) |
| **Expected VRAM** | ~1.2 GB |
| **Expected inference speed** | ~15–20 ms per thumbnail on an RTX 4060-class GPU |

## BiSeNet (Face Parsing)

| Field | Value |
| --- | --- |
| **Checkpoint** | `79999_iter.pth` |
| **Variant** | BiSeNet with ResNet-18 backbone, trained on CelebAMask-HQ (19-class face parsing) |
| **Model size** | ~50 MB |
| **Download source** | `zllrunning/face-parsing.PyTorch` GitHub releases |
| **License** | MIT |
| **Preferred inference backend** | Native PyTorch |
| **Expected VRAM** | ~0.8 GB (shares the Identity Engine stage with InsightFace; combined stage VRAM ~1.2 GB per the Performance Estimates table) |
| **Expected inference speed** | ~15 ms per thumbnail on an RTX 4060-class GPU |

## BiRefNet

| Field | Value |
| --- | --- |
| **Checkpoint** | `BiRefNet-general-epoch_244.pth` |
| **Variant** | General-purpose dichotomous image segmentation (matting) checkpoint, Swin-L backbone |
| **Model size** | ~885 MB (~200M params) |
| **Download source** | `ZhengPeng7/BiRefNet` on Hugging Face / GitHub releases |
| **License** | MIT |
| **Preferred inference backend** | Native PyTorch; FP16 supported |
| **Expected VRAM** | ~1.8 GB (FP16) |
| **Expected inference speed** | ~40 ms per thumbnail on an RTX 4060-class GPU |

## Depth Anything V2

| Field | Value |
| --- | --- |
| **Checkpoint** | `depth_anything_v2_vitb.pth` |
| **Variant** | **Base** (ViT-B, ~97.5M params) — deliberately chosen over the Large variant |
| **Model size** | ~390 MB |
| **Download source** | `depth-anything/Depth-Anything-V2-Base` on Hugging Face |
| **License** | Apache 2.0 |
| **Preferred inference backend** | Native PyTorch |
| **Expected VRAM** | ~1.2 GB (FP16) |
| **Expected inference speed** | ~20 ms per thumbnail on an RTX 4060-class GPU |

> **Licensing note:** the Depth Anything V2 **Large** checkpoint is released under **CC-BY-NC-4.0** (non-commercial), which conflicts with the Apache 2.0 license shown for this model in the Research and Model Comparisons table. The **Base** and **Small** checkpoints are Apache 2.0. This architecture standardizes on **Base** specifically to keep the model's license consistent with the rest of the stack — implementation teams must not silently upgrade to the Large checkpoint without a licensing review.

## TEED

| Field | Value |
| --- | --- |
| **Checkpoint** | `7_model.pth` |
| **Variant** | Trained on BIPED + BRIND edge-detection datasets |
| **Model size** | <5 MB (TEED is intentionally a tiny architecture) |
| **Download source** | TEED GitHub repository releases (see the `checkpoints/` directory) |
| **License** | MIT |
| **Preferred inference backend** | Native PyTorch |
| **Expected VRAM** | ~0.5 GB (mostly activation memory; weights themselves are negligible) |
| **Expected inference speed** | ~10 ms per thumbnail on an RTX 4060-class GPU |

## PaddleOCR

| Field | Value |
| --- | --- |
| **Checkpoint** | `PP-OCRv5_server_det` (detection) + `PP-OCRv5_server_rec` (recognition) |
| **Variant** | PP-OCRv5 server tier — chosen over the mobile tier for accuracy on stylized thumbnail text; chosen over the newer PP-OCRv6 tiers pending a dedicated validation pass (see note below) |
| **Model size** | ~90 MB combined (det + rec) |
| **Download source** | `PaddlePaddle/PP-OCRv5_server_det` and `PaddlePaddle/PP-OCRv5_server_rec` on Hugging Face, or the `paddleocr` package's built-in model downloader |
| **License** | Apache 2.0 |
| **Preferred inference backend** | PaddlePaddle native inference (`paddleocr` package) with GPU inference enabled; ONNX export available via `paddle2onnx` for a future ONNX Runtime migration |
| **Expected VRAM** | ~0.5 GB (GPU) / 0 GB (CPU fallback) |
| **Expected inference speed** | ~30 ms per thumbnail on an RTX 4060-class GPU |

> **Upgrade note:** PP-OCRv6 (medium tier) has since been released as the PaddleOCR team's new default, reporting meaningful accuracy gains over PP-OCRv5_server. It is not selected as the V2.1 production checkpoint because it was not part of the original Research and Model Comparisons evaluation; it is a strong candidate for the next architecture revision once independently benchmarked against this pipeline's stylized-thumbnail text distribution.

---

# Inter-Module Contracts

This section defines the input/output contract for every internal subsystem stage, so implementation teams can build and test each module independently against a fixed interface. All data structures are conceptual (language-agnostic); the canonical on-disk representation is the Complete Manifest Schema below.

### GroundingDINO

**Input:**
- `Thumbnail Image` (RGB, original resolution)
- `TextPrompt` (open-vocabulary label list, e.g. `"person . face . logo . arrow . text"`)

**Output:**
- `List[DetectedObject]`

**Fields (`DetectedObject`):**
- `label: string`
- `confidence: float`
- `bounding_box: [x0, y0, x1, y1]`
- `source: "grounding_dino"`

**Responsibilities:** Produce high-recall, text-prompted bounding boxes for downstream SAM 2 prompting. Does not perform captioning, OCR, or scene reasoning.

**Failure conditions:** No boxes above the confidence floor (`0.35`) for a given prompt term; malformed or empty `TextPrompt`; CUDA OOM during inference.

**Expected guarantees:** Every returned box lies within image bounds; `confidence` is a raw sigmoid score in `[0, 1]`; boxes below the confidence floor are excluded from the returned list (not merely flagged).

------------------------------------------------

### SAM 2

**Input:**
- `Image` (RGB, original resolution)
- `List[DetectedObject]` (box prompts from GroundingDINO)

**Output:**
- `List[SegmentedObject]`

**Fields (`SegmentedObject`):**
- `source_label: string` (copied from the prompting `DetectedObject.label`)
- `object_mask_path: string`
- `individual_object_crop_path: string`
- `bounding_box: [x0, y0, x1, y1]`
- `confidence_score: float`

**Responsibilities:** Convert each GroundingDINO box prompt into a precise instance mask and a cropped asset. Also produces the standalone `human_segmentation_path` when a `"person"`-labeled box is present.

**Failure conditions:** Empty `List[DetectedObject]` input (no prompts to segment); CUDA OOM (triggers the documented CPU tiled-processing fallback); mask bleed on low-contrast foreground/background boundaries (mitigated upstream by BiRefNet constraint, see Risk Analysis).

**Expected guarantees:** One `SegmentedObject` is returned per input `DetectedObject` that clears SAM 2's internal mask-quality threshold; masks are single-channel, same resolution as the input image.

------------------------------------------------

### InsightFace

**Input:**
- `Image` (RGB, original resolution)

**Output:**
- `List[FaceCandidate]`

**Fields (`FaceCandidate`):**
- `face_id: string`
- `bounding_box: [x0, y0, x1, y1]`
- `landmarks_5pt: [[x, y], ...]` (5 points)
- `embedding: float[512]` (ArcFace)
- `detection_confidence: float`

**Responsibilities:** Detect all face candidates in the frame and produce embeddings/landmarks for the Face Selection Logic. Does not itself select the creator face — that is a downstream CPU heuristic.

**Failure conditions:** Zero faces detected (valid, non-error outcome — downstream stages must handle an empty candidate list); low-resolution or heavily occluded faces falling below detector confidence.

**Expected guarantees:** `embedding` is always a fixed 512-dimension ArcFace vector when a candidate is returned; `landmarks_5pt` always contains exactly 5 points.

------------------------------------------------

### BiSeNet (Face Parsing)

**Input:**
- `Image` (RGB)
- `FaceCandidate` (the selected creator face, from Face Selection Logic)

**Output:**
- `FaceParsingResult`

**Fields:**
- `face_parsing_path: string` (19-class segmentation map)
- `hair_mask_path: string`
- `face_mask_path: string`

**Responsibilities:** Produce pixel-level face/hair segmentation for the already-selected creator face only (not run against every candidate, to bound latency).

**Failure conditions:** No creator face was selected upstream (stage is skipped, not errored); face crop too small for reliable parsing (below an internally configured minimum resolution).

**Expected guarantees:** Output masks are always the same resolution as the input face crop.

------------------------------------------------

### BiRefNet

**Input:**
- `Image` (RGB, original resolution)

**Output:**
- `MattingResult`

**Fields:**
- `foreground_path: string`
- `background_path: string`
- `transparent_foreground_path: string` (RGBA)
- `saliency_map_path: string` (single-channel, normalized `[0, 1]`)

**Responsibilities:** Dichotomous foreground/background matting and saliency estimation only. Explicitly does **not** produce instance-level masks (see SAM 2 for that).

**Failure conditions:** CUDA OOM on very high-resolution source thumbnails (falls back to tiled processing).

**Expected guarantees:** `saliency_map_path` is always full-image resolution and is consumed as an input to the Face Selection Logic's `saliency_score`.

------------------------------------------------

### OpenCLIP

**Input:**
- `Image` (RGB, original resolution)

**Output:**
- `EmbeddingResult`

**Fields:**
- `thumbnail_embedding_path: string` (float32 `.npy`, fixed dimension)
- `scene_embedding_path: string`
- `style_embedding_path: string`
- `semantic_feature_vector_path: string`
- `embedding_model: "openclip"`
- `embedding_dim: int`
- `checkpoint_hash: string` (see Risk Analysis — Embedding Drift)

**Responsibilities:** Produce fixed-length embeddings for downstream similarity search, retrieval, and clustering. Never mutates or depends on any other stage's output.

**Failure conditions:** None expected under normal operation beyond generic CUDA OOM; this is the lowest-risk stage in the pipeline.

**Expected guarantees:** `embedding_dim` and `checkpoint_hash` are always persisted alongside the vector so embeddings remain comparable (or are correctly flagged as incomparable) across checkpoint upgrades.

------------------------------------------------

### PaddleOCR

**Input:**
- `Image` (RGB, original resolution)

**Output:**
- `OCRResult`

**Fields:**
- `text_regions: List[TextRegion]`
  - `text: string`
  - `bounding_box: [x0, y0, x1, y1]`
  - `confidence: float`
  - `font_characteristics: { estimated_size_px, bold, italic }`
  - `reading_order_index: int`
  - `low_confidence: bool` (see Risk Analysis — OCR Failures)

**Responsibilities:** Independent text detection, recognition, and reading-order estimation. Runs with no dependency on Florence-2 or any other stage.

**Failure conditions:** No text present in the thumbnail (valid empty result); low-contrast or heavily stylized fonts driving confidence below `0.5` (flagged via `low_confidence`, not dropped).

**Expected guarantees:** `text_regions` is always returned as a list (possibly empty), never `null`; `reading_order_index` values are unique and monotonically ordered.

------------------------------------------------

### Florence-2

**Input:**
- `Image` (RGB, original resolution)
- `OCR Output` (`OCRResult.text_regions`, optional enrichment input)

**Output:**
- `SemanticAnalysis`

**Fields:**
- `composition_metadata: string`
- `lighting_estimation: string`
- `thumbnail_layout_metadata: string`
- `visual_hierarchy_metadata: List[string]`
- `scene_description: string`

**Responsibilities:** Unified semantic reasoning — composition, layout, visual hierarchy, and scene captioning. May cross-reference `OCR Output` to improve grounding of text-related semantic claims, but must degrade gracefully (produce output, possibly with lower confidence) if `OCR Output` is empty.

**Failure conditions:** Hallucinated scene detail on highly distorted or abstract thumbnails (mitigated via cross-referencing against PaddleOCR and GroundingDINO outputs per the Risk Analysis).

**Expected guarantees:** All string fields are always populated (never `null`) even on low-confidence inputs; the stage never blocks pipeline completion.

------------------------------------------------

### Asset Generation Engine

**Input:**
- All previous stage outputs (`DetectedObject` list, `SegmentedObject` list, `FaceCandidate` list + selection result, `FaceParsingResult`, `MattingResult`, `EmbeddingResult`, `OCRResult`, `SemanticAnalysis`, plus Depth Anything V2 / TEED spatial outputs and CPU geometric heuristics)

**Output:**
- `ReferenceManifest`

**Responsibilities:** Aggregate every upstream stage's output into the single Complete Manifest Schema (see Storage & Manifest Schema), write all binary assets to the documented folder structure, and compute the CPU-only geometric heuristics (camera angle, horizon, lighting direction, dominant colors) that have no dedicated model stage.

**Failure conditions:** Any required upstream field missing at aggregation time is written as `null`/empty rather than causing the whole manifest write to fail — partial manifests are preferred over silent pipeline crashes; `processing_metadata.complete_manifest` is set to `false` when any stage was skipped or failed, so downstream consumers can detect partial results.

**Expected guarantees:** `ReferenceManifest` always validates against the Complete Manifest Schema's field set, even when individual values are `null`; the manifest is the single source of truth handed to ComfyUI.

---

# Configuration Architecture

The original architecture referenced per-model settings implicitly; this section formalizes a **centralized configuration system** so every model can be swapped, retuned, or re-pointed to a new checkpoint without touching pipeline source code.

## Configuration File Structure

The pipeline loads a single root configuration (YAML, though the schema below is format-agnostic) at worker boot:

```text
vision_stack:

  grounding_dino:
    checkpoint: "groundingdino_swint_ogc.pth"
    precision: "fp16"
    device: "cuda:0"
    backend: "pytorch"
    batch_size: 1
    cache_enabled: true
    timeout: 5000
    fallback: "skip_stage"

  sam2:
    checkpoint: "sam2.1_hiera_large.pt"
    precision: "fp16"
    device: "cuda:0"
    backend: "pytorch"
    batch_size: 1
    cache_enabled: true
    timeout: 8000
    fallback: "cpu_tiled_processing"

  florence2:
    checkpoint: "Florence-2-large"
    precision: "fp16"
    device: "cuda:0"
    backend: "transformers"
    batch_size: 1
    cache_enabled: true
    timeout: 6000
    fallback: "skip_stage"

  openclip:
    checkpoint: "ViT-B-32/laion2b_s34b_b79k"
    precision: "fp16"
    device: "cuda:0"
    backend: "open_clip"
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: "skip_stage"

  insightface:
    checkpoint: "buffalo_l"
    precision: "fp16"
    device: "cuda:0"
    backend: "onnxruntime"
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: "skip_stage"

  birefnet:
    checkpoint: "BiRefNet-general-epoch_244.pth"
    precision: "fp16"
    device: "cuda:0"
    backend: "pytorch"
    batch_size: 1
    cache_enabled: true
    timeout: 5000
    fallback: "skip_stage"

  bisenet:
    checkpoint: "79999_iter.pth"
    precision: "fp16"
    device: "cuda:0"
    backend: "pytorch"
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: "skip_stage"

  paddleocr:
    checkpoint: "PP-OCRv5_server_det+PP-OCRv5_server_rec"
    precision: "fp16"
    device: "cuda:0"
    backend: "paddle"
    batch_size: 1
    cache_enabled: true
    timeout: 4000
    fallback: "cpu_fallback"

  depth_anything:
    checkpoint: "depth_anything_v2_vitb.pth"
    precision: "fp16"
    device: "cuda:0"
    backend: "pytorch"
    batch_size: 1
    cache_enabled: true
    timeout: 3000
    fallback: "skip_stage"
```

## Per-Model Configuration Fields

| Field | Type | Description |
| --- | --- | --- |
| `checkpoint` | string | Exact checkpoint identifier, resolved against the `ModelRegistry` (see Model Lifecycle & Caching Architecture) at worker boot. Changing this value alone is sufficient to swap a model's weights. |
| `precision` | enum (`fp16`, `fp32`) | Inference precision. `fp16` is the V2 default across the stack per the Hardware & Memory Management Strategy. |
| `device` | string | Target device (`cuda:0`, `cpu`). Allows individual stages to be pinned to CPU for debugging without a code change. |
| `backend` | enum (`pytorch`, `onnxruntime`, `paddle`, `transformers`, `tensorrt`) | Inference backend, matching the "Preferred inference backend" recommendation in Model Checkpoint Specifications. `tensorrt` is reserved for the Future Improvements TensorRT compilation path. |
| `batch_size` | int | Always `1` in the current sequential, single-thumbnail pipeline; present for forward compatibility with a future batched execution mode. |
| `cache_enabled` | bool | Whether the model participates in the CPU-RAM Cache Policy (weights kept resident in CPU RAM between GPU-Active periods) or is reloaded from disk on every use. |
| `timeout` | int (ms) | Per-invocation timeout. On expiry, the stage's `fallback` policy is triggered rather than blocking the worker indefinitely. |
| `fallback` | enum (`skip_stage`, `cpu_fallback`, `cpu_tiled_processing`, `retry_once`) | Behavior when a stage times out, OOMs, or otherwise fails. `skip_stage` writes `null` fields for that stage into the manifest and sets `processing_metadata.complete_manifest = false`, consistent with the Asset Generation Engine's partial-manifest guarantee. |

## Design Principles

* **No source-code coupling:** Every value that could plausibly change between deployments — checkpoint path, precision, device, backend, timeout — lives in configuration, never hardcoded in pipeline modules.
* **Single source of truth:** The `ModelRegistry` described in the Model Lifecycle & Caching Architecture section is populated directly from this file at boot; there is no second place where checkpoint paths are declared.
* **Environment overrides:** Standard practice (environment variables or a layered config loader) should allow a deployment-specific override file to patch individual fields (e.g. forcing `device: "cpu"` for a CI test runner) without editing the base `vision_stack` file.
* **Validated at boot, not at call time:** All `checkpoint` paths and `backend` values are validated for existence/compatibility once, during the `[Registered]` model-lifecycle state, so a misconfiguration fails fast at worker startup rather than mid-batch.

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

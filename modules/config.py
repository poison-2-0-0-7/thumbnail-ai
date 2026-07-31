"""
config.py
=========

Centralized configuration for Module 1 (CSV Reader).

This module holds constants that are shared across the CSV reader
implementation: schema definitions, validation patterns, and logging
configuration. Keeping these values in one place avoids magic strings
scattered throughout the codebase and gives future modules a single,
predictable place to look for shared configuration.

This module has zero dependencies on other project modules, in keeping
with the project's requirement that every module be independently
testable and loosely coupled.
"""

from __future__ import annotations

from pathlib import Path

from models import GenerationProfile
from vision_stack.config import (
    VISION_STACK_CONFIG_ENV,
    VISION_STACK_CONFIG_PATH,
    VISION_STACK_MODEL_ORDER,
    VISION_STACK_STAGE_LATENCY_KEYS,
    VISION_STACK_VERSION,
    load_vision_stack_config,
)

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

#: Root of the project. config.py lives in <root>/src/, so the parent of
#: the parent directory is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default location of the creators CSV file.
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "data" / "creators.csv"

#: Directory where log files are written.
LOG_DIR: Path = PROJECT_ROOT / "logs"

#: Log file used by Module 1.
MODULE1_LOG_PATH: Path = LOG_DIR / "module1.log"

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

#: Canonical column order for the creators CSV. This is the single source
#: of truth for the schema; any CSV that does not match this header is
#: considered malformed.
CSV_COLUMNS: tuple[str, ...] = ("email", "video_url")

#: Encoding used for all CSV reads/writes.
CSV_ENCODING: str = "utf-8"

# ---------------------------------------------------------------------------
# Validation patterns
# ---------------------------------------------------------------------------

#: A pragmatic (not fully RFC 5322 compliant, but production-sane) email
#: validation pattern. Rejects obviously malformed addresses while
#: avoiding the complexity/false-negative tradeoffs of a fully compliant
#: regex.
EMAIL_PATTERN: str = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

#: Accepted YouTube URL formats:
#:   - https://www.youtube.com/watch?v=VIDEO_ID
#:   - https://youtube.com/watch?v=VIDEO_ID
#:   - https://youtu.be/VIDEO_ID
#: VIDEO_ID is exactly 11 characters of [A-Za-z0-9_-], which matches
#: YouTube's actual video ID format.
YOUTUBE_URL_PATTERN: str = (
    r"^https://(?:www\.)?(?:"
    r"youtube\.com/watch\?v=(?P<id_long>[A-Za-z0-9_-]{11})(?:&\S*)?"
    r"|"
    r"youtu\.be/(?P<id_short>[A-Za-z0-9_-]{11})(?:\?\S*)?"
    r")$"
)

# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------

#: Maximum number of seconds to wait for a file lock before giving up.
LOCK_TIMEOUT_SECONDS: float = 10.0

#: Polling interval (seconds) while waiting to acquire a lock.
LOCK_CHECK_INTERVAL_SECONDS: float = 0.1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Module 2 — YouTube Metadata Extractor (log path only)
# ---------------------------------------------------------------------------

#: Log file used by Module 2.
MODULE2_LOG_PATH: Path = LOG_DIR / "module2.log"

# ---------------------------------------------------------------------------
# Module 3 — Thumbnail Downloader
# ---------------------------------------------------------------------------

#: Log file used by Module 3.
MODULE3_LOG_PATH: Path = LOG_DIR / "module3.log"

#: Directory where downloaded thumbnails are stored.
DEFAULT_THUMBNAIL_DIR: Path = PROJECT_ROOT / "data" / "thumbnails"

#: Filename template for a saved thumbnail; formatted with ``video_id``.
THUMBNAIL_FILENAME_TEMPLATE: str = "{video_id}.jpg"

#: Total seconds to wait for a thumbnail HTTP response before giving up.
THUMBNAIL_REQUEST_TIMEOUT_SECONDS: float = 30.0

#: Maximum retry attempts for transient download failures.
THUMBNAIL_MAX_RETRY_ATTEMPTS: int = 3

#: Minimum seconds to wait between retry attempts (exponential back-off base).
THUMBNAIL_RETRY_WAIT_MIN_SECONDS: float = 1.0

#: Maximum seconds to wait between retry attempts.
THUMBNAIL_RETRY_WAIT_MAX_SECONDS: float = 8.0

#: Minimum acceptable file size in bytes.  Files smaller than this are
#: rejected as empty or truncated even if Pillow can open them.
THUMBNAIL_MIN_FILE_SIZE_BYTES: int = 1_024  # 1 KB

#: Image format strings (as returned by ``PIL.Image.format``) that are
#: accepted as valid thumbnails.  YouTube serves JPEG for maxresdefault
#: but may also serve WEBP or PNG depending on the CDN edge node.
THUMBNAIL_ACCEPTED_IMAGE_FORMATS: frozenset[str] = frozenset(
    {"JPEG", "PNG", "WEBP", "GIF"}
)

#: HTTP status codes that indicate a permanent failure and should NOT be
#: retried.  All other non-2xx codes are considered transient.
THUMBNAIL_PERMANENT_HTTP_ERRORS: frozenset[int] = frozenset({403, 404, 410})

# ---------------------------------------------------------------------------
# Module 4 — Thumbnail Intelligence Engine
# ---------------------------------------------------------------------------

#: Log file used by Module 4.
MODULE4_LOG_PATH: Path = LOG_DIR / "module4.log"

#: Directory where structured intelligence reports are stored as JSON.
DEFAULT_ANALYSIS_DIR: Path = PROJECT_ROOT / "data" / "analysis"

#: Filename template for a saved intelligence report; formatted with
#: ``video_id``.
ANALYSIS_FILENAME_TEMPLATE: str = "{video_id}.json"

#: Device string passed to CV/ML models. Resolved once per process by
#: ``thumbnail_intelligence`` via ``torch.cuda.is_available()`` — this
#: constant is the fallback used when that resolution is unavailable.
DEFAULT_DEVICE: str = "cpu"

#: EasyOCR language list. English is sufficient for the current
#: creator base; additional languages can be appended without any
#: other code changes.
OCR_LANGUAGES: list[str] = ["en"]

#: Minimum per-detection confidence for an OCR text region to be kept.
#: Regions below this threshold are dropped as noise but still counted
#: toward ``average_confidence`` bookkeeping in the raw engine output.
OCR_MIN_CONFIDENCE: float = 0.35

#: InsightFace model pack name.
FACE_MODEL_NAME: str = "buffalo_l"

#: Minimum detector confidence for a face to be kept.
FACE_MIN_CONFIDENCE: float = 0.5

#: YOLO model checkpoint. Ultralytics resolves this name to a cached
#: weights file (downloading it once on first use).
YOLO_MODEL_NAME: str = "yolo11n.pt"

#: Minimum per-detection confidence for a YOLO object to be kept.
YOLO_MIN_CONFIDENCE: float = 0.4

#: Maximum number of dominant colors to extract per thumbnail.
COLOR_PALETTE_SIZE: int = 5

#: Base URL of the local Ollama server used for the reasoning stage.
OLLAMA_BASE_URL: str = "http://localhost:11434"

#: Ollama model used for the reasoning stage.
OLLAMA_MODEL: str = "qwen3:8b"

#: Maximum seconds to wait for an Ollama response before giving up.
OLLAMA_TIMEOUT_SECONDS: float = 60.0

#: Maximum retry attempts for transient Ollama failures.
OLLAMA_MAX_RETRY_ATTEMPTS: int = 3

#: Minimum seconds to wait between Ollama retry attempts.
OLLAMA_RETRY_WAIT_MIN_SECONDS: float = 2.0

#: Maximum seconds to wait between Ollama retry attempts.
OLLAMA_RETRY_WAIT_MAX_SECONDS: float = 20.0

#: Maximum number of transcript characters forwarded to the reasoning
#: stage. Long transcripts are truncated (keeping the head, where
#: creators usually state the video's premise) to keep prompt size and
#: local-inference latency bounded.
REASONING_TRANSCRIPT_CHAR_LIMIT: int = 6_000

# ---------------------------------------------------------------------------
# Module 5 — Redesign Specification Engine (fully deterministic — no
# network calls or AI/LLM dependency of any kind)
# ---------------------------------------------------------------------------

#: Log file used by Module 5.
MODULE5_LOG_PATH: Path = LOG_DIR / "module5.log"

#: Directory where structured redesign specifications are stored as JSON.
DEFAULT_REDESIGN_SPEC_DIR: Path = PROJECT_ROOT / "data" / "redesign_specs"

#: Filename template for a saved redesign specification; formatted with
#: ``video_id``.
REDESIGN_SPEC_FILENAME_TEMPLATE: str = "{video_id}.json"

# --- Composition thresholds ---

CLUTTER_HIGH_THRESHOLD: float = 0.6
CLUTTER_REDUCTION_FACTOR: float = 0.7
MIN_NEGATIVE_SPACE_RATIO: float = 0.25
RULE_OF_THIRDS_LOW_THRESHOLD: float = 0.4
MIN_SUBJECT_AREA_RATIO: float = 0.15

# --- Color thresholds ---

BRIGHTNESS_TARGET_RANGE: tuple[float, float] = (0.35, 0.75)
CONTRAST_TARGET_RANGE: tuple[float, float] = (0.4, 0.8)
SATURATION_TARGET_RANGE: tuple[float, float] = (0.3, 0.7)

# --- Weakness-keyword matching (rule-based, not generative) ---

COLOR_TEMPERATURE_FLIP_KEYWORDS: frozenset[str] = frozenset(
    {"too warm", "too cool", "color temperature", "washed out"}
)

# ---------------------------------------------------------------------------
# Module 6 — Prompt Compiler (fully deterministic — no AI/LLM dependency,
# no image generation, and no network calls)
# ---------------------------------------------------------------------------

#: Log file used by Module 6.
MODULE6_LOG_PATH: Path = LOG_DIR / "module6.log"

#: Directory where compiled prompt packages are stored as JSON.
DEFAULT_PROMPT_PACKAGE_DIR: Path = PROJECT_ROOT / "data" / "prompt_packages"

#: Filename template for a saved prompt package; formatted with ``video_id``.
PROMPT_PACKAGE_FILENAME_TEMPLATE: str = "{video_id}.json"

# --- Zone-label thresholds ---

#: Fraction of frame width/height below which a bounding-box centre is in
#: the left/top third. Above ``1 - ZONE_THIRD_THRESHOLD`` is right/bottom.
ZONE_THIRD_THRESHOLD: float = 1 / 3

# --- Generation parameters (fixed defaults; seed is a stable video-ID hash) ---

DEFAULT_GENERATION_WIDTH: int = 1280
DEFAULT_GENERATION_HEIGHT: int = 720
DEFAULT_ASPECT_RATIO: str = "16:9"
DEFAULT_GUIDANCE_SCALE: float = 7.5
DEFAULT_INFERENCE_STEPS: int = 30
DEFAULT_SAMPLER: str = "deterministic"
SEED_HASH_MODULUS: int = 2**32

# --- Quality parameters ---

BASE_QUALITY_TAGS: tuple[str, ...] = (
    "sharp focus",
    "high detail",
    "professional thumbnail quality",
)
DEFAULT_MIN_RESOLUTION_PX: int = 1280

# --- Backend-neutral model settings ---

#: A stable placeholder for Module 7 to map to its selected generator.
DEFAULT_MODEL_NAME: str = "thumbnail-diffusion-v1"
DEFAULT_STYLE_PRESET: str = "photographic"
DEFAULT_NEGATIVE_PROMPT_WEIGHT: float = 1.0

# --- Fixed rendering and safety constraints ---

BASE_NEGATIVE_PROMPT_TERMS: tuple[str, ...] = (
    "blurry",
    "low resolution",
    "watermark",
    "distorted anatomy",
    "extra limbs",
    "text artifacts",
    "jpeg compression artifacts",
)

BASE_RENDERING_CONSTRAINTS: tuple[str, ...] = (
    "Render at the exact target resolution; do not crop after generation.",
    "Do not add any watermark, logo, or signature.",
)

SAFETY_CONSTRAINTS: tuple[str, ...] = (
    "Do not depict real, identifiable people; use a generic or anonymized figure.",
    "Do not reproduce copyrighted characters, logos, or branded imagery.",
    "Do not include graphic violence, gore, or sexual content.",
    "Do not include misleading medical, financial, or safety claims in any rendered text.",
)

# ---------------------------------------------------------------------------
# Module 5.5 — Thumbnail Copywriter & Layout Planner Engine
# ---------------------------------------------------------------------------

#: Log file used by Module 5.5.
MODULE55_LOG_PATH: Path = LOG_DIR / "module5_5.log"

#: Directory where structured design blueprints are stored as JSON.
DEFAULT_DESIGN_BLUEPRINT_DIR: Path = PROJECT_ROOT / "data" / "design_blueprints"

#: Filename template for a saved design blueprint; formatted with ``video_id``.
DESIGN_BLUEPRINT_FILENAME_TEMPLATE: str = "{video_id}.json"

MODULE55_MOBILE_CHAR_SOFT_LIMIT: int = 40
MODULE55_MOBILE_CHAR_HARD_LIMIT: int = 60
MODULE55_MAX_ZONE_OVERLAP: float = 0.15
MODULE55_SAFE_MARGIN_RATIO: float = 0.05

MODULE55_HOOK_TYPE_THRESHOLDS: dict[str, float] = {
    "high_curiosity": 0.7,
    "mismatch": 0.5,
}

MODULE55_CURIOSITY_LEXICON: frozenset[str] = frozenset(
    {
        "why", "how", "secret", "never", "stop", "dont", "don't", "truth", "hidden",
        "reason", "mistake", "fix", "top", "best", "worst", "i tried", "vs", "what",
        "this", "shocking", "banned", "illegal", "changed", "finally", "revealed",
        "exposed", "proven", "simple", "hack", "trick"
    }
)

MODULE55_HEADLINE_SCORE_WEIGHTS: dict[str, float] = {
    "curiosity": 0.25,
    "emotional_impact": 0.20,
    "readability": 0.15,
    "ctr_potential": 0.20,
    "mobile_readability": 0.10,
    "brand_consistency": 0.10,
}

# ---------------------------------------------------------------------------
# Module 7 — Local Image Generation Engine
# ---------------------------------------------------------------------------

COMFYUI_HOST: str = "127.0.0.1"
COMFYUI_PORT: int = 8188
COMFYUI_STARTUP_TIMEOUT_SECONDS: float = 60.0
COMFYUI_REQUEST_TIMEOUT_SECONDS: float = 120.0
COMFYUI_WS_PATH: str = "/ws"
COMFYUI_CONNECT_RETRY_ATTEMPTS: int = 3
COMFYUI_CONNECT_RETRY_WAIT_MIN_SECONDS: float = 2.0
COMFYUI_CONNECT_RETRY_WAIT_MAX_SECONDS: float = 10.0
COMFYUI_WEBSOCKET_TIMEOUT_SECONDS: float = 5.0
COMFYUI_EXECUTION_TIMEOUT_SECONDS: float = 300.0
COMFYUI_POLL_INTERVAL_SECONDS: float = 3.0
MODULE7_STILL_QUEUED_WARNING_SECONDS: float = 30.0
MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT: int = 25
COMFYUI_WS_RECONNECT_POLL_CYCLES: int = 3
COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS: float = 10.0
COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS: int = 3
COMFYUI_HISTORY_CONFIRMATION_RETRY_DELAY_SECONDS: float = 0.5
COMFYUI_OUTPUT_SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = ("png", "jpg", "jpeg", "webp")
COMFYUI_OUTPUT_PREFERRED_NODES: tuple[str, ...] = ()
COMFYUI_OUTPUT_DOWNLOAD_TIMEOUT_SECONDS: float = 30.0
COMFYUI_OUTPUT_DOWNLOAD_MAX_RETRIES: int = 3
COMFYUI_OUTPUT_DOWNLOAD_RETRY_BACKOFF_SECONDS: float = 1.0

MODULE7_PROFILE: str = "auto"
MODULE7_VRAM_HEADROOM_GB: float = 0.5
MODULE7_WORKFLOW_LIBRARY_DIR: Path = PROJECT_ROOT / "workflows"
MODULE7_WORKFLOW_VERSION: str = "workflow_v1"
MODULE7_NICHE_WORKFLOW_MAP: dict[str, str] = {
    "gaming": "gaming.json", "finance": "finance.json", "education": "education.json",
    "podcast": "podcast.json", "tech": "tech.json", "lifestyle": "lifestyle.json",
    "vlog": "vlog.json", "fitness": "fitness.json", "reaction": "reaction.json",
    "documentary": "documentary.json",
}

MODULE7_GENERATION_PROFILES: dict[str, GenerationProfile] = {
    "PROFILE_STANDARD": GenerationProfile(name="PROFILE_STANDARD", checkpoint="juggernautXL.safetensors", checkpoint_family="sdxl", sampler="dpmpp_2m", scheduler="karras", steps=30, cfg=6.5, controlnet_enabled=True, ipadapter_enabled=True, restoration="codeformer", restoration_fidelity=0.35, upscaler="real_esrgan_x4", expected_vram_gb=7.5, expected_generation_seconds=25.0),
    "PROFILE_FAST": GenerationProfile(name="PROFILE_FAST", checkpoint="juggernautXL.safetensors", checkpoint_family="sdxl", sampler="dpmpp_2m", scheduler="karras", steps=16, cfg=6.0, controlnet_enabled=True, ipadapter_enabled=True, restoration="codeformer", restoration_fidelity=0.35, upscaler="lanczos_only", expected_vram_gb=7.0, expected_generation_seconds=9.0),
    "PROFILE_PREMIUM": GenerationProfile(name="PROFILE_PREMIUM", checkpoint="flux1-schnell-q5_k_m.gguf", checkpoint_family="flux", sampler="euler", scheduler="simple", steps=20, cfg=1.0, controlnet_enabled=False, ipadapter_enabled=True, restoration="both", restoration_fidelity=0.35, upscaler="real_esrgan_x4", expected_vram_gb=7.8, expected_generation_seconds=55.0),
    "PROFILE_LOW_VRAM": GenerationProfile(name="PROFILE_LOW_VRAM", checkpoint="juggernautXL.safetensors", checkpoint_family="sdxl", sampler="dpmpp_2m", scheduler="karras", steps=20, cfg=6.0, controlnet_enabled=False, ipadapter_enabled=True, restoration="codeformer", restoration_fidelity=0.4, upscaler="lanczos_only", expected_vram_gb=5.0, expected_generation_seconds=32.0),
}
MODULE7_PROFILE_PREFERENCE: tuple[str, ...] = (
    "PROFILE_PREMIUM", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM",
)
MODULE7_QA_WEIGHTS: dict[str, float] = {
    "identity_score": 0.30, "face_quality_score": 0.15,
    "composition_score": 0.15, "text_safe_zone_score": 0.15,
    "object_preservation_score": 0.15, "color_compliance_score": 0.10,
}
MODULE7_IDENTITY_SIMILARITY_THRESHOLD: float = 0.5
MODULE7_CODEFORMER_FIDELITY: float = 0.35
MAX_IDENTITY_RETRIES: int = 2
MAX_GENERATION_RETRIES: int = 3
MODULE7_SAVE_CANDIDATES: bool = False
MODULE7_LOG_PATH: Path = LOG_DIR / "module7.log"
MODULE7_METRICS_PATH: Path = LOG_DIR / "module7_metrics.jsonl"
MODULE7_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "generated_thumbnails"
MODULE7_NSFW_THRESHOLD: float = 0.15
MODULE7_MAX_CONCURRENT_GENERATIONS: int = 1
MODULE7_DRAFT_STEPS: int = 12
MODULE7_DRAFT_UPSCALE_SKIP: bool = True
MODULE7_FRAGMENT_LIBRARY_DIR: Path = PROJECT_ROOT / "workflows" / "fragments"
MODULE7_CONTROLNET_STRENGTH_DEFAULTS: dict[str, float] = {
    "depth": 0.55,
    "canny": 0.45,
    "segmentation": 0.5,
}
MODULE7_IPADAPTER_WEIGHT_DEFAULT: float = 0.6
MODULE7_CAPABILITY_PROBE_ENABLED: bool = True
MODULE7_CAPABILITY_PROBE_CACHE_SECONDS: float = 300.0


# ---------------------------------------------------------------------------
# Module 6.5 - Visual Reference Engine
# ---------------------------------------------------------------------------

VRE_STORAGE_ROOT: Path = PROJECT_ROOT / "data" / "visual_references"
VRE_MANIFEST_FILENAME: str = "reference_manifest.json"
VRE_ENGINE_VERSION: str = "1.0.0"
VRE_CACHE_ENABLED: bool = True
VRE_MIN_IMAGE_DIMENSION_PX: int = 256
VRE_CANNY_LOW_THRESHOLD: int = 100
VRE_CANNY_HIGH_THRESHOLD: int = 200
VRE_FACE_DETECTION_CONFIDENCE: float = 0.85
MODULE65_LOG_PATH: Path = LOG_DIR / "module6_5.log"

# ---------------------------------------------------------------------------
# AI Vision Stack V2.1 - Configuration Architecture
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# AI Vision Stack V2.1 - GroundingDINO Wrapper
# ---------------------------------------------------------------------------

#: Minimum detection confidence to keep a box, per the architecture document's
#: documented confidence floor for GroundingDINO.
GROUNDING_DINO_BOX_THRESHOLD: float = 0.35

#: Minimum per-token text-grounding confidence (GroundingDINO's own internal
#: text-matching threshold, distinct from the output box_threshold above).
GROUNDING_DINO_TEXT_THRESHOLD: float = 0.25

#: Default open-vocabulary prompt when a caller does not supply one.
GROUNDING_DINO_DEFAULT_PROMPT: str = "person . face . logo . text . arrow"

#: Log file for this wrapper, following the one-log-file-per-component
#: convention already used by every other module.
VISION_STACK_GROUNDING_DINO_LOG_PATH: Path = LOG_DIR / "vision_stack_grounding_dino.log"


# ---------------------------------------------------------------------------
# Module 8 — Asset Extraction Engine
# ---------------------------------------------------------------------------

MODULE8_LOG_PATH: Path = LOG_DIR / "module8.log"

DEFAULT_ASSET_EXTRACTION_DIR: Path = PROJECT_ROOT / "data" / "asset_extraction"
ASSET_MANIFEST_FILENAME: str = "asset_manifest.json"
ASSET_EXTRACTION_ENGINE_VERSION: str = "1.0.0"
ASSET_EXTRACTION_CACHE_ENABLED: bool = True

# --- Family execution order (fixed, deterministic; drives sequential GPU use) ---
ASSET_EXTRACTION_FAMILY_ORDER: tuple[str, ...] = (
    "typography",        # cheapest, no model — run first for fast partial results
    "visual_properties",  # cheapest, no model
    "composition",        # cheapest, no model
    "objects",            # SAM2
    "people",             # InsightFace + BiSeNet
    "scene",              # BiRefNet + DepthAnything + SAM2 + GroundingDINO
    "effects",            # cheapest, no model — run last, lowest priority
)

# --- Per-family thresholds ---
ASSET_MIN_FACE_CROP_CONFIDENCE: float = 0.5          # reuse Module 4's FACE_MIN_CONFIDENCE value
ASSET_MIN_OBJECT_MASK_CONFIDENCE: float = 0.35        # SAM2 box-prompted mask acceptance floor
ASSET_OBJECT_HIERARCHY_CONTAINMENT_RATIO: float = 0.9  # child-bbox-inside-parent threshold
ASSET_SKY_GROUND_PROMPT: str = "sky . ground . horizon"  # GroundingDINO open-vocab query
ASSET_EXTENDED_PALETTE_K: int = 8                       # vs Module 4's k=5 in ColorProfile
ASSET_BLUR_LAPLACIAN_SHARP_THRESHOLD: float = 100.0
ASSET_BLUR_LAPLACIAN_SOFT_THRESHOLD: float = 30.0
ASSET_EFFECTS_MIN_CONFIDENCE_TO_FLAG: float = 0.4

# --- Resource budget (RTX 4060 laptop, 8GB VRAM / 16GB system RAM) ---
ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX: int = 2048     # downscale ceiling before any model call
ASSET_EXTRACTION_SAM2_TILE_SIZE_PX: int = 1024           # cpu_tiled_processing tile size
ASSET_EXTRACTION_MODEL_TIMEOUT_SECONDS: float = 30.0
ASSET_EXTRACTION_MAX_RETRY_ATTEMPTS: int = 2


# ---------------------------------------------------------------------------
# Module 9 — AI Decision Engine
# ---------------------------------------------------------------------------

MODULE9_LOG_PATH: Path = LOG_DIR / "module9.log"
MODULE9_METRICS_PATH: Path = LOG_DIR / "module9_metrics.jsonl"

DEFAULT_DECISION_DIR: Path = PROJECT_ROOT / "data" / "decisions"
DECISION_MANIFEST_FILENAME: str = "decision_manifest.json"
DECISION_ENGINE_VERSION: str = "1.0.0"
DECISION_CACHE_ENABLED: bool = True

#: Ambiguity confidence threshold for LLM routing.
AMBIGUITY_CONFIDENCE_THRESHOLD: float = 0.65

#: Recalibration ceiling for self-reported LLM confidence.
LLM_CONFIDENCE_CEILING: float = 0.9

#: Priority hierarchy for resolving conflicts among candidate decisions.
DECISION_PRIORITY_ORDER: tuple[str, ...] = (
    "keep",
    "remove",
    "replace",
    "enhance",
    "add",
)

#: Bounding box IoU deduplication threshold for ADD recommendations.
ADD_DEDUP_IOU_THRESHOLD: float = 0.7

#: Dedicated or default Ollama model used for Module 9 adjudication.
MODULE9_OLLAMA_MODEL: str = "qwen3:8b"
MODULE9_OLLAMA_TIMEOUT_SECONDS: float = 60.0
MODULE9_OLLAMA_MAX_RETRY_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Module 10 — Asset Composer
# ---------------------------------------------------------------------------

MODULE10_LOG_PATH: Path = LOG_DIR / "module10.log"
COMPOSITION_WORKSPACE_ROOT: Path = PROJECT_ROOT / "data" / "composition_workspaces"
COMPOSITION_ENGINE_VERSION: str = "1.0.0"
COMPOSITION_CACHE_ENABLED: bool = True
COMPOSITION_SAFE_MARGIN_PX: int = 24
COMPOSITION_TEXT_FEATHER_PX: int = 6
COMPOSITION_MANIFEST_FILENAME: str = "workspace_manifest.json"
MODULE7_COMPOSITION_WORKSPACE_ROOT: Path = COMPOSITION_WORKSPACE_ROOT
COMPOSITION_RESOLVE_CANNY_ASSET_KEY: str = "canny_map"

# ---------------------------------------------------------------------------
# Module 10.5 — Thumbnail Planner
# ---------------------------------------------------------------------------

THUMBNAIL_PLANNER_ENABLED: bool = False

MODULE10_5_LOG_PATH: Path = LOG_DIR / "module10_5.log"
DEFAULT_GENERATION_PLAN_DIR: Path = PROJECT_ROOT / "data" / "generation_plans"
GENERATION_PLAN_FILENAME: str = "generation_plan.json"
PLANNER_ENGINE_VERSION: str = "1.0.0"
PLANNER_CACHE_ENABLED: bool = True

ASSET_EXTRACTION_ENGINE_ENABLED: bool = False
DECISION_ENGINE_ENABLED: bool = False

# ---------------------------------------------------------------------------

# Evaluation Framework (PVQEF)
# ---------------------------------------------------------------------------

EVAL_LOG_PATH: Path = LOG_DIR / "evaluation.log"
EVAL_RUNS_DIR: Path = PROJECT_ROOT / "data" / "evaluation" / "runs"
EVAL_GOLDEN_DIR: Path = PROJECT_ROOT / "data" / "evaluation" / "golden"
EVAL_HISTORY_PATH: Path = PROJECT_ROOT / "data" / "evaluation" / "history" / "benchmark_history.jsonl"

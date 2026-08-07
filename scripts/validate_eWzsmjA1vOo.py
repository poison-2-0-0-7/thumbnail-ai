"""
validate_eWzsmjA1vOo.py
=======================

Production Validation Script for Video eWzsmjA1vOo (Phase 33).
Executes the complete V2 pipeline for eWzsmjA1vOo and validates all produced artifacts,
scene understanding, layer decomposition, prompt compilation, decisions, and similarity metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_MODULES_DIR = _PROJECT_ROOT / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from main import run_pipeline
from thumbnail_understanding import ThumbnailUnderstandingEngine
from similarity_gate import SimilarityGate


def validate():
    print("======================================================================")
    print("PRODUCTION VALIDATION RUN: Video eWzsmjA1vOo")
    print("======================================================================")

    csv_path = _PROJECT_ROOT / "data" / "creators_eWzsmjA1vOo.csv"
    video_id = "eWzsmjA1vOo"

    # 1. Run complete pipeline
    run_pipeline(csv_path=csv_path)

    # 2. Verify Thumbnail Understanding V2 artifact
    und_file = _PROJECT_ROOT / "data" / "thumbnail_understanding" / f"{video_id}.json"
    if und_file.is_file():
        print(f"\n[SUCCESS] ThumbnailUnderstanding V2 artifact found: {und_file}")
        data = json.loads(und_file.read_text(encoding="utf-8"))
        elements_count = len(data.get("scene_graph", {}).get("elements", []))
        hero_id = data.get("scene_graph", {}).get("hero_element_id")
        layers_count = data.get("decomposed_scene", {}).get("layer_count", 0)
        actions_count = len(data.get("improvement_plan", {}).get("actions", []))
        ctr_potential = data.get("psychology", {}).get("ctr_potential_score", 0.0)

        print(f"  - Grounded Elements: {elements_count}")
        print(f"  - Hero Subject ID: {hero_id}")
        print(f"  - Decomposed Layers: {layers_count}")
        print(f"  - Planned Actions: {actions_count}")
        print(f"  - Source CTR Potential Score: {ctr_potential}")

    else:
        print(f"\n[WARNING] ThumbnailUnderstanding V2 artifact not found at {und_file}")

    # 3. Verify Redesign Specification artifact
    spec_file = _PROJECT_ROOT / "data" / "redesign_specs" / f"spec_{video_id}.json"
    if spec_file.is_file():
        print(f"\n[SUCCESS] RedesignSpecification artifact found: {spec_file}")
    else:
        print(f"\n[INFO] RedesignSpecification path check: {spec_file}")

    # 4. Verify Decision Manifest artifact
    dec_file = _PROJECT_ROOT / "data" / "decisions" / f"decisions_{video_id}.json"
    if dec_file.is_file():
        print(f"\n[SUCCESS] DecisionManifest artifact found: {dec_file}")
        dec_data = json.loads(dec_file.read_text(encoding="utf-8"))
        print(f"  - Decisions Count: {len(dec_data.get('decisions', []))}")

    # 5. Verify Similarity Gate metrics against generated output
    gen_dir = _PROJECT_ROOT / "data" / "generated_thumbnails"
    gen_images = list(gen_dir.glob(f"*{video_id}*"))
    if not gen_images:
        gen_images = list(gen_dir.glob("*.jpg")) + list(gen_dir.glob("*.png"))

    source_thumb = _PROJECT_ROOT / "data" / "thumbnails" / f"{video_id}.jpg"

    if source_thumb.is_file() and gen_images:
        gen_thumb = gen_images[0]
        print(f"\n[SIMILARITY EVALUATION] Comparing {source_thumb.name} vs {gen_thumb.name}")
        sim_res = SimilarityGate.evaluate(str(source_thumb), str(gen_thumb))
        print(f"  - Passed Gate: {sim_res.passed}")
        print(f"  - SSIM Score: {sim_res.ssim_score:.4f}")
        print(f"  - Difference Score: {sim_res.difference_score:.4f}")
    else:
        print(f"\n[INFO] Source thumbnail exists: {source_thumb.is_file()}, Generated images count: {len(gen_images)}")

    print("\n======================================================================")
    print("PRODUCTION VALIDATION COMPLETE")
    print("======================================================================")


if __name__ == "__main__":
    validate()

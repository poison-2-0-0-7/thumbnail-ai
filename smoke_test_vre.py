from pathlib import Path
import sys

sys.path.insert(0, "modules")

from visual_reference_engine import VisualReferenceEngine

engine = VisualReferenceEngine()

manifest = engine.prepare_assets(
    video_id="smoke_test",
    source_image_path="data/thumbnails/eWzsmjA1vOo.jpg",
)

print("SUCCESS")
print(manifest)
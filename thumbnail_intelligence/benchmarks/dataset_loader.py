"""
dataset_loader.py
=================

DatasetLoader Implementation for Phase 6.1 Benchmark Framework.
Loads benchmark datasets from JSON, CSV, image directories, or creates synthetic datasets.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from thumbnail_intelligence.benchmarks.models import DatasetItem
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief


class DatasetLoaderError(RuntimeError):
    """Exception raised for dataset loader errors."""
    pass


class DatasetLoader:
    """Loads and constructs benchmark DatasetItem collections."""

    @staticmethod
    def load_from_json(json_path: Union[str, Path]) -> List[DatasetItem]:
        """Load DatasetItem list from a JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise DatasetLoaderError(f"JSON dataset file not found at '{path}'")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return [DatasetItem.model_validate(item) for item in data]
            elif isinstance(data, dict) and "items" in data:
                return [DatasetItem.model_validate(item) for item in data["items"]]
            else:
                raise DatasetLoaderError(f"Unexpected JSON format in '{path}'; expected list or dict with 'items'.")
        except Exception as e:
            raise DatasetLoaderError(f"Failed to parse JSON dataset from '{path}': {str(e)}") from e

    @staticmethod
    def load_from_csv(csv_path: Union[str, Path]) -> List[DatasetItem]:
        """Load DatasetItem list from a CSV file."""
        path = Path(csv_path)
        if not path.exists():
            raise DatasetLoaderError(f"CSV dataset file not found at '{path}'")

        items: List[DatasetItem] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item_id = row.get("item_id") or row.get("id") or f"video_{len(items)+1:03d}"
                    title = row.get("title") or "Sample Video Title"
                    category = row.get("category") or "Tech"
                    url = row.get("video_url") or row.get("url")
                    orig_path = row.get("original_thumbnail_path") or row.get("image_path")

                    brief = DesignBrief()
                    items.append(
                        DatasetItem(
                            item_id=item_id,
                            title=title,
                            category=category,
                            video_url=url,
                            original_thumbnail_path=orig_path,
                            brief=brief,
                        )
                    )
            return items
        except Exception as e:
            raise DatasetLoaderError(f"Failed to parse CSV dataset from '{path}': {str(e)}") from e

    @staticmethod
    def load_from_directory(dir_path: Union[str, Path]) -> List[DatasetItem]:
        """Load DatasetItem list from a directory containing thumbnail images."""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise DatasetLoaderError(f"Directory not found at '{path}'")

        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        image_files = [f for f in path.iterdir() if f.suffix.lower() in valid_exts]

        items: List[DatasetItem] = []
        for img_file in image_files:
            item_id = f"item_{img_file.stem}"
            title = img_file.stem.replace("_", " ").replace("-", " ").title()
            brief = DesignBrief()
            items.append(
                DatasetItem(
                    item_id=item_id,
                    title=title,
                    category="Benchmark",
                    original_thumbnail_path=str(img_file),
                    brief=brief,
                )
            )

        return items

    @staticmethod
    def create_synthetic_dataset(count: int = 5, temp_dir: Optional[str] = None) -> List[DatasetItem]:
        """Create synthetic benchmark DatasetItem list with generated baseline thumbnail images for testing."""
        out_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="synth_dataset_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        categories = ["Gaming", "Tech", "Education", "Vlogs", "Finance"]
        items: List[DatasetItem] = []

        for i in range(count):
            item_id = f"synth_video_{i+1:03d}"
            cat = categories[i % len(categories)]
            title = f"How To Master {cat} In 2026 Part {i+1}"

            # Create synthetic original thumbnail image
            orig_path = str(out_dir / f"orig_{item_id}.png")
            img = np.full((720, 1280, 3), 120 + i * 20, dtype=np.uint8)
            cv2.putText(img, title, (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.imwrite(orig_path, img)

            brief = DesignBrief()
            items.append(
                DatasetItem(
                    item_id=item_id,
                    title=title,
                    category=cat,
                    original_thumbnail_path=orig_path,
                    brief=brief,
                )
            )

        return items

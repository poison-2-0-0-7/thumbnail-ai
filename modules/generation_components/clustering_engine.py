"""CandidateClusteringEngine: Perceptual hashing, duplicate detection, and cluster formation for candidate thumbnails."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from typing import Any, Sequence
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class CandidateCluster(BaseModel):
    """Cluster grouping duplicate/similar candidate images."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    survivor_index: int
    candidate_indices: list[int]
    duplicate_indices: list[int] = Field(default_factory=list)


class ClusteringResult(BaseModel):
    """Output manifest of candidate clustering pass."""

    model_config = ConfigDict(frozen=True)

    clusters: list[CandidateCluster]
    survivor_indices: list[int]
    excluded_duplicates: dict[int, str] = Field(default_factory=dict)  # candidate_idx -> exclusion_reason
    candidate_cluster_map: dict[int, str] = Field(default_factory=dict)  # candidate_idx -> cluster_id
    perceptual_hashes: dict[int, str] = Field(default_factory=dict)  # candidate_idx -> hash_str


def compute_dhash(image_path: Path | str, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for an image file."""
    path = Path(image_path)
    if not path.is_file():
        # Return fallback hash if file does not exist
        return "0" * (hash_size * hash_size // 4)

    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())
            
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left_pixel = pixels[row * (hash_size + 1) + col]
                    right_pixel = pixels[row * (hash_size + 1) + col + 1]
                    difference.append(left_pixel > right_pixel)
            
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 1 << (index % 4)
                if index % 4 == 3:
                    hex_string.append(hex(decimal_value)[2:])
                    decimal_value = 0
            return "".join(hex_string)
    except Exception as exc:
        logger.warning(f"Failed to compute perceptual dHash for {path}: {exc}")
        return "0" * (hash_size * hash_size // 4)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings of equal length."""
    if len(hash1) != len(hash2) or not hash1 or not hash2:
        return 64
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 64


class CandidateClusteringEngine:
    """Engine for perceptual hashing, duplicate detection, and survivor selection among candidates."""

    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold

    def cluster_candidates(
        self,
        candidates: Sequence[tuple[int, Path, Any, Any, Any, Any, str, dict[str, float]]],
    ) -> ClusteringResult:
        """
        Cluster candidate thumbnails using perceptual dHash.

        Args:
            candidates: Sequence of tuples matching candidate loop item structure:
                        (cand_idx, image_path, qa_report, face_match, strategy, prompt_pkg, wf_hash, durations)

        Returns:
            ClusteringResult detailing clusters, survivors, and duplicate exclusions.
        """
        if not candidates:
            return ClusteringResult(
                clusters=[],
                survivor_indices=[],
                excluded_duplicates={},
                candidate_cluster_map={},
                perceptual_hashes={},
            )

        hashes: dict[int, str] = {}
        scores: dict[int, float] = {}

        for cand in candidates:
            cand_idx = cand[0]
            img_path = cand[1]
            qa_report = cand[2]
            hashes[cand_idx] = compute_dhash(img_path)
            score = getattr(qa_report, "overall_score", 0.0) if qa_report else 0.0
            scores[cand_idx] = score

        clusters: list[CandidateCluster] = []
        visited: set[int] = set()
        survivor_indices: list[int] = []
        excluded_duplicates: dict[int, str] = {}
        candidate_cluster_map: dict[int, str] = {}

        cluster_counter = 1
        sorted_candidates = sorted(candidates, key=lambda c: (c[0]))

        for cand in sorted_candidates:
            cand_idx = cand[0]
            if cand_idx in visited:
                continue

            cluster_id = f"cluster_{cluster_counter}"
            cluster_members = [cand_idx]
            visited.add(cand_idx)

            # Find duplicates using Hamming distance threshold
            for other_cand in sorted_candidates:
                other_idx = other_cand[0]
                if other_idx in visited:
                    continue
                dist = hamming_distance(hashes[cand_idx], hashes[other_idx])
                if dist <= self.threshold:
                    cluster_members.append(other_idx)
                    visited.add(other_idx)

            # Select survivor in cluster (highest overall score, fallback to lowest index)
            survivor_idx = max(cluster_members, key=lambda idx: (scores[idx], -idx))
            duplicates = [idx for idx in cluster_members if idx != survivor_idx]

            survivor_indices.append(survivor_idx)
            for idx in cluster_members:
                candidate_cluster_map[idx] = cluster_id

            for dup_idx in duplicates:
                excluded_duplicates[dup_idx] = f"duplicate_cluster_{cluster_id}_survivor_{survivor_idx}"

            cluster_obj = CandidateCluster(
                cluster_id=cluster_id,
                survivor_index=survivor_idx,
                candidate_indices=cluster_members,
                duplicate_indices=duplicates,
            )
            clusters.append(cluster_obj)
            cluster_counter += 1

        logger.info(
            "CandidateClusteringEngine formed {n_clusters} clusters across {n_cands} candidates; {n_surv} survivors.",
            n_clusters=len(clusters),
            n_cands=len(candidates),
            n_surv=len(survivor_indices),
        )

        return ClusteringResult(
            clusters=clusters,
            survivor_indices=survivor_indices,
            excluded_duplicates=excluded_duplicates,
            candidate_cluster_map=candidate_cluster_map,
            perceptual_hashes=hashes,
        )

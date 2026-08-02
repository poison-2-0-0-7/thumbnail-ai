"""
StyleProfileStore component for Phase 2 of Module 10 Creator Style Learning.

Manages channel_id sharded creator profiles with incremental centroid vector updates
and atomic temp-file-then-replace persistence.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.config import MODULE10_CREATOR_PROFILES_DIR, MODULE10_STYLE_MIN_SAMPLES
from modules.models import CreatorStyleEmbedding, StyleProfileManifest, ThumbnailStyleSignature


class StyleProfileStore:
    """
    Persistent store for creator style signatures, running embeddings, and manifests.
    Sharded by channel_id under data/creator_style_profiles/{channel_id}/.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or MODULE10_CREATOR_PROFILES_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_channel_dir(self, channel_id: str) -> Path:
        # Sanitize channel_id for path safety
        safe_channel_id = "".join(c for c in channel_id if c.isalnum() or c in ("-", "_")).strip()
        if not safe_channel_id:
            safe_channel_id = "default_channel"
        channel_dir = self.base_dir / safe_channel_id
        channel_dir.mkdir(parents=True, exist_ok=True)
        (channel_dir / "signatures").mkdir(parents=True, exist_ok=True)
        return channel_dir

    def _atomic_write_json(self, target_path: Path, data_dict: dict) -> None:
        """Atomically write dictionary to target_path using temporary file replacement."""
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, target_path)

    def get_manifest(self, channel_id: str) -> Optional[StyleProfileManifest]:
        """Retrieve manifest for channel_id if it exists."""
        channel_dir = self._get_channel_dir(channel_id)
        manifest_path = channel_dir / "profile_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StyleProfileManifest(**data)
        except Exception:
            return None

    def get_embedding(self, channel_id: str) -> Optional[CreatorStyleEmbedding]:
        """Retrieve running centroid embedding for channel_id if it exists."""
        channel_dir = self._get_channel_dir(channel_id)
        emb_path = channel_dir / "style_embedding.json"
        if not emb_path.exists():
            return None
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CreatorStyleEmbedding(**data)
        except Exception:
            return None

    def get_signature(self, channel_id: str, video_id: str) -> Optional[ThumbnailStyleSignature]:
        """Retrieve stored signature for a specific video_id."""
        channel_dir = self._get_channel_dir(channel_id)
        sig_path = channel_dir / "signatures" / f"{video_id}.json"
        if not sig_path.exists():
            return None
        try:
            with open(sig_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ThumbnailStyleSignature(**data)
        except Exception:
            return None

    def update_profile(
        self,
        video_id: str,
        channel_id: str,
        signature: ThumbnailStyleSignature,
        embedding_vector: list[float],
        min_samples: int = MODULE10_STYLE_MIN_SAMPLES,
    ) -> tuple[StyleProfileManifest, CreatorStyleEmbedding]:
        """
        Store a new signature and incrementally update the creator's running centroid embedding.
        """
        channel_dir = self._get_channel_dir(channel_id)
        now_str = datetime.now(timezone.utc).isoformat()

        # 1. Save signature
        sig_path = channel_dir / "signatures" / f"{video_id}.json"
        self._atomic_write_json(sig_path, signature.model_dump())

        # 2. Update running centroid embedding
        existing_emb = self.get_embedding(channel_id)
        if existing_emb is None or not existing_emb.embedding or existing_emb.sample_count == 0:
            new_sample_count = 1
            new_centroid = list(embedding_vector)
            source_videos = [video_id]
            first_seen = now_str
        else:
            old_count = existing_emb.sample_count
            new_sample_count = old_count + 1
            old_centroid = existing_emb.embedding

            # Incremental mean vector formula: C_new = C_old + (E_new - C_old) / N
            new_centroid = [
                old_val + (new_val - old_val) / float(new_sample_count)
                for old_val, new_val in zip(old_centroid, embedding_vector)
            ]
            source_videos = list(existing_emb.source_video_ids)
            if video_id not in source_videos:
                source_videos.append(video_id)
            existing_manifest = self.get_manifest(channel_id)
            first_seen = existing_manifest.first_seen_at if existing_manifest else now_str

        updated_emb = CreatorStyleEmbedding(
            channel_id=channel_id,
            embedding=new_centroid,
            embedding_model="OpenCLIP-ViT-B-32",
            source_video_ids=source_videos,
            sample_count=new_sample_count,
            computed_at=now_str,
        )
        self._atomic_write_json(channel_dir / "style_embedding.json", updated_emb.model_dump())

        # 3. Update profile manifest
        is_established = new_sample_count >= min_samples
        updated_manifest = StyleProfileManifest(
            channel_id=channel_id,
            sample_count=new_sample_count,
            profile_established=is_established,
            first_seen_at=first_seen,
            last_updated_at=now_str,
            video_ids=source_videos,
            schema_version="1.0.0",
        )
        self._atomic_write_json(channel_dir / "profile_manifest.json", updated_manifest.model_dump())

        return updated_manifest, updated_emb

    def reset_centroid(
        self,
        channel_id: str,
        from_video_ids: list[str],
        new_embeddings: list[list[float]],
    ) -> CreatorStyleEmbedding:
        """
        Reset/re-seed a creator's centroid embedding (e.g. following confirmed style drift).
        """
        channel_dir = self._get_channel_dir(channel_id)
        now_str = datetime.now(timezone.utc).isoformat()

        if not new_embeddings:
            raise ValueError("new_embeddings list must not be empty for centroid reset")

        dim = len(new_embeddings[0])
        n = float(len(new_embeddings))
        new_centroid = [
            sum(emb[i] for emb in new_embeddings) / n for i in range(dim)
        ]

        updated_emb = CreatorStyleEmbedding(
            channel_id=channel_id,
            embedding=new_centroid,
            embedding_model="OpenCLIP-ViT-B-32",
            source_video_ids=list(from_video_ids),
            sample_count=len(new_embeddings),
            computed_at=now_str,
        )
        self._atomic_write_json(channel_dir / "style_embedding.json", updated_emb.model_dump())

        existing_manifest = self.get_manifest(channel_id)
        first_seen = existing_manifest.first_seen_at if existing_manifest else now_str
        updated_manifest = StyleProfileManifest(
            channel_id=channel_id,
            sample_count=len(new_embeddings),
            profile_established=len(new_embeddings) >= MODULE10_STYLE_MIN_SAMPLES,
            first_seen_at=first_seen,
            last_updated_at=now_str,
            video_ids=list(from_video_ids),
            schema_version="1.0.0",
        )
        self._atomic_write_json(channel_dir / "profile_manifest.json", updated_manifest.model_dump())

        return updated_emb

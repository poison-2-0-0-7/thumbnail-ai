"""
observability/trace/artifact_index_builder.py
=============================================

Discovers, checks existence, calculates hashes, and indexes all module artifacts for a given video_id.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules import config
from observability.interfaces import IArtifactCollector
from observability.models import ArtifactIndex, ArtifactRef


def compute_sha256(path: Path) -> Optional[str]:
    """Compute the SHA-256 hash of a file on disk in chunks."""
    if not path.is_file():
        return None
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


class ArtifactIndexBuilder(IArtifactCollector):
    """
    Builds an ArtifactIndex for a specific video_id by inspecting known artifact paths.
    """

    def __init__(
        self,
        thumbnail_dir: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
        asset_extraction_dir: Optional[Path] = None,
        redesign_spec_dir: Optional[Path] = None,
        design_blueprint_dir: Optional[Path] = None,
        prompt_package_dir: Optional[Path] = None,
        decision_dir: Optional[Path] = None,
        composition_workspace_dir: Optional[Path] = None,
        generation_plan_dir: Optional[Path] = None,
        strategy_pack_dir: Optional[Path] = None,
        generated_thumbnail_dir: Optional[Path] = None,
    ) -> None:
        self.thumbnail_dir = thumbnail_dir or config.DEFAULT_THUMBNAIL_DIR
        self.analysis_dir = analysis_dir or config.DEFAULT_ANALYSIS_DIR
        self.asset_extraction_dir = asset_extraction_dir or config.DEFAULT_ASSET_EXTRACTION_DIR
        self.redesign_spec_dir = redesign_spec_dir or config.DEFAULT_REDESIGN_SPEC_DIR
        self.design_blueprint_dir = design_blueprint_dir or config.DEFAULT_DESIGN_BLUEPRINT_DIR
        self.prompt_package_dir = prompt_package_dir or config.DEFAULT_PROMPT_PACKAGE_DIR
        self.decision_dir = decision_dir or config.DEFAULT_DECISION_DIR
        self.composition_workspace_dir = (
            composition_workspace_dir or (config.PROJECT_ROOT / "data" / "composition_workspaces")
        )
        self.generation_plan_dir = generation_plan_dir or getattr(
            config, "DEFAULT_GENERATION_PLAN_DIR", config.PROJECT_ROOT / "data" / "generation_plans"
        )
        self.strategy_pack_dir = strategy_pack_dir or getattr(
            config, "MODULE7_STRATEGY_PACK_DIR", config.PROJECT_ROOT / "data" / "strategy_packs"
        )
        self.generated_thumbnail_dir = generated_thumbnail_dir or config.MODULE7_OUTPUT_DIR

    def _build_ref(self, module: str, artifact_type: str, path: Path) -> ArtifactRef:
        """Construct an ArtifactRef for a target file path."""
        exists = path.is_file()
        if exists:
            sha256_hash = compute_sha256(path)
            try:
                size_bytes = path.stat().st_size
            except Exception:
                size_bytes = None
            return ArtifactRef(
                module=module,
                artifact_type=artifact_type,
                path=str(path.resolve()),
                exists=True,
                sha256=sha256_hash,
                size_bytes=size_bytes,
            )
        else:
            return ArtifactRef(
                module=module,
                artifact_type=artifact_type,
                path=str(path),
                exists=False,
                sha256=None,
                size_bytes=None,
            )

    def collect(self, video_id: str) -> ArtifactIndex:
        """
        Walk all expected artifact locations for video_id and produce an ArtifactIndex.
        """
        refs: list[ArtifactRef] = []

        # Module 3 — Thumbnail Downloader
        refs.append(
            self._build_ref("module3", "thumbnail_image", self.thumbnail_dir / f"{video_id}.jpg")
        )

        # Module 4 — Thumbnail Intelligence
        refs.append(
            self._build_ref("module4", "thumbnail_intelligence", self.analysis_dir / f"{video_id}.json")
        )

        # Module 8 — Asset Extraction
        refs.append(
            self._build_ref(
                "module8",
                "asset_extraction_manifest",
                self.asset_extraction_dir / video_id / "asset_manifest.json",
            )
        )

        # Module 5 — Redesign Spec
        refs.append(
            self._build_ref("module5", "redesign_specification", self.redesign_spec_dir / f"{video_id}.json")
        )

        # Module 5.5 — Design Blueprint
        refs.append(
            self._build_ref(
                "module5.5", "design_blueprint", self.design_blueprint_dir / f"{video_id}.json"
            )
        )

        # Module 6 — Prompt Compiler
        refs.append(
            self._build_ref("module6", "prompt_package", self.prompt_package_dir / f"{video_id}.json")
        )

        # Module 9 — Decision Engine
        refs.append(
            self._build_ref(
                "module9", "decision_manifest", self.decision_dir / video_id / "decision_manifest.json"
            )
        )

        # Module 10 — Asset Composer
        refs.append(
            self._build_ref(
                "module10",
                "composition_workspace",
                self.composition_workspace_dir / video_id / "workspace_manifest.json",
            )
        )
        refs.append(
            self._build_ref(
                "module10",
                "generation_bundle",
                self.composition_workspace_dir / video_id / "generation_bundle.json",
            )
        )

        # Module 10.5 — Thumbnail Planner
        refs.append(
            self._build_ref(
                "module10.5", "generation_plan", self.generation_plan_dir / f"{video_id}.json"
            )
        )
        refs.append(
            self._build_ref(
                "module10.5", "strategy_pack", self.strategy_pack_dir / f"{video_id}.json"
            )
        )

        # Module 7 — Image Generation
        refs.append(
            self._build_ref(
                "module7", "generated_thumbnail", self.generated_thumbnail_dir / video_id / f"{video_id}.png"
            )
        )
        refs.append(
            self._build_ref(
                "module7",
                "image_generation_result",
                self.generated_thumbnail_dir / video_id / "manifest.json",
            )
        )

        built_at = datetime.now(timezone.utc).isoformat()
        return ArtifactIndex(video_id=video_id, refs=refs, built_at=built_at)

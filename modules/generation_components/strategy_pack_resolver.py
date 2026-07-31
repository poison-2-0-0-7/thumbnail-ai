"""Strategy Pack library discovery, validation, and candidate strategy resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger

from config import PROJECT_ROOT
from models import CandidateStrategy, StrategyPack
from module7_exceptions import StrategyPackError

DEFAULT_STRATEGY_PACK_DIR: Path = PROJECT_ROOT / "data" / "strategy_packs"


class StrategyPackLibrary:
    """Discover, load, and validate strategy packs from local disk."""

    def __init__(self, library_dir: Path = DEFAULT_STRATEGY_PACK_DIR) -> None:
        self.library_dir = Path(library_dir).resolve()

    def discover(self) -> list[Path]:
        """Return strategy pack files in deterministic filename order."""
        if not self.library_dir.is_dir():
            return []
        return sorted(self.library_dir.glob("*.json"), key=lambda path: path.name)

    def load(self, pack_ref: str | Path) -> StrategyPack:
        """Load and validate a strategy pack by name or file path."""
        ref_path = Path(pack_ref)
        if not ref_path.is_absolute():
            if not str(pack_ref).endswith(".json"):
                path = self.library_dir / f"{pack_ref}.json"
            else:
                path = self.library_dir / pack_ref
        else:
            path = ref_path

        path = path.resolve()
        if not path.is_file():
            raise StrategyPackError(f"Strategy pack file not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            raise StrategyPackError(f"Could not read strategy pack {path}: {exc}") from exc

        self.validate(data, source=path)

        try:
            pack = StrategyPack.model_validate(data)
        except Exception as exc:
            raise StrategyPackError(f"Strategy pack validation failed for {path}: {exc}") from exc

        logger.debug("Loaded StrategyPack '{name}' ({count} strategies) from {path}", name=pack.name, count=len(pack.strategies), path=path)
        return pack

    def validate(self, data: object, source: Path | None = None) -> None:
        """Validate JSON structure contract required for a strategy pack."""
        label = str(source) if source is not None else "strategy pack"
        if not isinstance(data, dict):
            raise StrategyPackError(f"{label} must be a JSON object")

        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise StrategyPackError(f"{label} requires a non-empty 'name' string")

        strategies = data.get("strategies")
        if not isinstance(strategies, list) or not strategies:
            raise StrategyPackError(f"{label} requires a non-empty 'strategies' array")

        for idx, strat in enumerate(strategies):
            if not isinstance(strat, dict):
                raise StrategyPackError(f"{label} strategy at index {idx} must be a JSON object")
            if not isinstance(strat.get("name"), str) or not strat["name"].strip():
                raise StrategyPackError(f"{label} strategy at index {idx} missing valid 'name'")


class StrategyPackResolver:
    """Resolve configured strategy pack into ordered list[CandidateStrategy]."""

    def __init__(self, library: StrategyPackLibrary | None = None) -> None:
        self.library = library or StrategyPackLibrary()

    def resolve(
        self,
        requested_pack: Optional[str] = None,
        max_candidates: int = 1,
    ) -> list[CandidateStrategy]:
        """Resolve requested pack or fallback to single faithful strategy."""
        if not requested_pack:
            logger.info("No strategy_pack requested; using single faithful default strategy.")
            return [CandidateStrategy.faithful_default()]

        pack = self.library.load(requested_pack)
        strategies = list(pack.strategies)

        if len(strategies) > max_candidates:
            logger.warning(
                "Strategy pack '{pack}' contains {count} strategies, but max_candidates={max_cand}. Truncating.",
                pack=pack.name,
                count=len(strategies),
                max_cand=max_candidates,
            )
            strategies = strategies[:max_candidates]

        names = [s.name for s in strategies]
        logger.info(
            "Resolved strategy_pack={pack} -> {n} candidate(s): {names}",
            pack=pack.name,
            n=len(strategies),
            names=", ".join(names),
        )
        return strategies

"""
llm_reasoner.py
===============

LLM reasoning stage using local Ollama for ambiguous candidate decision adjudication.
Implements ILLMReasoner with tenacity retries and JSON schema enforcement.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from modules.config import (
    MODULE9_OLLAMA_MODEL,
    MODULE9_OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)
from modules.decision_components.confidence import recalibrate_llm_confidence
from modules.decision_components.interfaces import ILLMReasoner
from modules.decision_exceptions import (
    OllamaConnectionError,
    OllamaResponseParseError,
    OllamaTimeoutError,
)
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement

_MODULE9_OLLAMA_SYSTEM_PROMPT = """You are an expert AI thumbnail redesign arbiter.
Your task is to adjudicate candidate decisions for a YouTube thumbnail redesign.
For each candidate provided, evaluate whether the proposed action (keep, remove, replace, enhance, add) is visually optimal for high click-through-rate (CTR).

Respond ONLY with valid JSON in the following exact format:
{
  "adjudications": [
    {
      "candidate_id": "cand_1",
      "action": "keep",
      "confidence": 0.85,
      "rationale": "High visual contrast creator face drives engagement"
    }
  ]
}
Do not output any text or explanation outside the JSON object.
"""


class LLMReasoner(ILLMReasoner):
    """Local Ollama LLM reasoner for adjudicating ambiguous candidate decisions."""

    def __init__(
        self,
        model_name: str = MODULE9_OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout_seconds: float = MODULE9_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def adjudicate(
        self, candidates: list[CandidateDecision], bundle: Any
    ) -> list[CandidateDecision]:
        """Adjudicate ambiguous candidate decisions using local Ollama."""
        if not candidates:
            return []

        user_prompt = self._build_user_prompt(candidates, bundle)

        try:
            raw_response = self._call_ollama_with_retry(user_prompt)
            adjudicated = self._parse_adjudication_response(raw_response, candidates)
            return adjudicated
        except Exception as exc:
            logger.warning(
                "LLM adjudication degraded for video_id={id}: {exc}. Retaining original candidates.",
                id=bundle.video_id if hasattr(bundle, "video_id") else "unknown",
                exc=str(exc),
            )
            return candidates

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((OllamaConnectionError, OllamaTimeoutError)),
        reraise=True,
    )
    def _call_ollama_with_retry(self, user_prompt: str) -> str:
        """Call Ollama chat API with tenacity retries."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _MODULE9_OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(f"Ollama request timed out: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"Ollama connection error: {exc}") from exc
        except Exception as exc:
            raise OllamaResponseParseError(f"Unexpected error calling Ollama: {exc}") from exc

    def _build_user_prompt(self, candidates: list[CandidateDecision], bundle: Any) -> str:
        cand_descriptions = []
        for c in candidates:
            cand_descriptions.append(
                f"- Candidate ID: {c.candidate_id}, Target: {c.target.label} ({c.target.element_type}), "
                f"Proposed Action: {c.action.value}, Current Confidence: {c.confidence:.2f}, Rationale: {c.rationale}"
            )
        cand_str = "\n".join(cand_descriptions)
        return f"Candidates requiring adjudication:\n{cand_str}"

    def _parse_adjudication_response(
        self, raw_text: str, original_candidates: list[CandidateDecision]
    ) -> list[CandidateDecision]:
        try:
            data = json.loads(raw_text)
            adjudications = data.get("adjudications", [])
            adj_map = {item["candidate_id"]: item for item in adjudications if "candidate_id" in item}

            result: list[CandidateDecision] = []
            for orig in original_candidates:
                if orig.candidate_id in adj_map:
                    item = adj_map[orig.candidate_id]
                    revised_action_str = item.get("action", orig.action.value).lower()
                    try:
                        revised_action = DecisionAction(revised_action_str)
                    except ValueError:
                        revised_action = orig.action

                    raw_conf = float(item.get("confidence", orig.confidence))
                    recalibrated_conf = recalibrate_llm_confidence(raw_conf)
                    rationale = str(item.get("rationale", orig.rationale))

                    source = (
                        DecisionSource.RULE_LLM_AGREEMENT
                        if revised_action == orig.action
                        else DecisionSource.LLM
                    )

                    revised = CandidateDecision(
                        candidate_id=orig.candidate_id,
                        target=orig.target,
                        action=revised_action,
                        confidence=recalibrated_conf,
                        source=source,
                        rationale=f"LLM adjudicated: {rationale}",
                        rule_ids=orig.rule_ids,
                        llm_raw_response_ref=raw_text[:100],
                    )
                    result.append(revised)
                else:
                    result.append(orig)
            return result
        except Exception as exc:
            raise OllamaResponseParseError(f"Failed to parse LLM adjudication JSON: {exc}") from exc

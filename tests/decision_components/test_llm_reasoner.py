"""
test_llm_reasoner.py
====================

Unit tests for LLMReasoner (Phase 4).
"""

from unittest.mock import MagicMock, patch
import json
import pytest
import requests

from modules.decision_components.llm_reasoner import LLMReasoner
from modules.decision_exceptions import OllamaConnectionError, OllamaTimeoutError
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


@pytest.fixture
def sample_candidate() -> CandidateDecision:
    target = TargetElement(element_id="elem_1", element_type="object", label="car")
    return CandidateDecision(
        candidate_id="cand_1",
        target=target,
        action=DecisionAction.REPLACE,
        confidence=0.50,
        source=DecisionSource.RULE,
        rationale="Low confidence candidate",
    )


@patch("requests.post")
def test_llm_reasoner_success(mock_post: MagicMock, sample_candidate: CandidateDecision):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "adjudications": [
                        {
                            "candidate_id": "cand_1",
                            "action": "replace",
                            "confidence": 0.95,  # Should be capped at 0.9 ceiling
                            "rationale": "Better color contrast",
                        }
                    ]
                }
            )
        }
    }
    mock_post.return_value = mock_response

    reasoner = LLMReasoner()
    bundle = MagicMock()
    bundle.video_id = "v_test"

    results = reasoner.adjudicate([sample_candidate], bundle)

    assert len(results) == 1
    assert results[0].action == DecisionAction.REPLACE
    assert results[0].confidence == 0.9  # Recalibrated to ceiling 0.9
    assert results[0].source == DecisionSource.RULE_LLM_AGREEMENT


@patch("requests.post")
def test_llm_reasoner_degraded_fallback(mock_post: MagicMock, sample_candidate: CandidateDecision):
    # Simulate connection error
    mock_post.side_effect = requests.exceptions.ConnectionError("Ollama offline")

    reasoner = LLMReasoner()
    bundle = MagicMock()
    bundle.video_id = "v_test"

    # Should degrade gracefully to original candidates
    results = reasoner.adjudicate([sample_candidate], bundle)
    assert len(results) == 1
    assert results[0].candidate_id == "cand_1"

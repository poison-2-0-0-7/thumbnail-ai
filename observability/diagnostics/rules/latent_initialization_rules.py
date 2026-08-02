"""
observability/diagnostics/rules/latent_initialization_rules.py
================================================================

Diagnostic rules for latent initialization mode observations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class SourceNeverEncodedRule(IDiagnosticRule):
    """
    RULE-LAT-01: Verifies whether source thumbnail was VAE encoded or latent was generated from noise.
    On current txt2img renderer, latent_initialization_mode is always EmptyLatentImage.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-LAT-01"

    @property
    def rule_name(self) -> str:
        return "Source Image Latent Encoding State"

    @property
    def category(self) -> str:
        return "latent_initialization"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        mode = facts.latent_initialization_mode or "EmptyLatentImage"

        if mode == "EmptyLatentImage":
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="INFO",
                confidence=1.0,
                affected_module="module7",
                root_cause="Renderer initialized latent space using EmptyLatentImage (noise) rather than VAE encoding a source image.",
                recommended_action="If image-to-image or inpainting was desired, ensure VAEEncode and Module 7 V2 editing engine are enabled.",
                supporting_facts=["latent_initialization_mode=EmptyLatentImage", "denoise=1.0"],
                evaluation_timestamp=now_str,
            )
        return None

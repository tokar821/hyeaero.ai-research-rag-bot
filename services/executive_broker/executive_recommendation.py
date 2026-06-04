"""Single authoritative broker recommendation — one primary, ranked alternatives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExecutiveRecommendation:
    primary_recommendation: str
    confidence: str  # HIGH | MODERATE | LOW
    rationale: str
    alternatives: List[Dict[str, str]] = field(default_factory=list)
    rejected_options: List[Dict[str, str]] = field(default_factory=list)
    direct_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_recommendation": self.primary_recommendation,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "alternatives": list(self.alternatives),
            "rejected_options": list(self.rejected_options),
            "direct_answer": self.direct_answer,
        }


__all__ = ["ExecutiveRecommendation"]

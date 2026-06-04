"""Unified broker truth object from pipeline metadata (read-only assembly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from services.intent_collapse.canonical_intent_frame import CanonicalIntentFrame


@dataclass
class BrokerTruthState:
    intent: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": dict(self.intent) if self.intent else None,
            "evaluation": dict(self.evaluation) if self.evaluation else None,
            "recommendation": dict(self.recommendation) if self.recommendation else None,
            "constraints": dict(self.constraints),
            "confidence": self.confidence,
        }

    @property
    def primary_model(self) -> Optional[str]:
        rec = self.recommendation or {}
        return rec.get("primary_recommendation") or None

    @property
    def has_executive_recommendation(self) -> bool:
        return bool(self.primary_model)


def _confidence_from_parts(
    intent_conf: float,
    exec_conf: Optional[str],
    ambiguity_count: int,
) -> float:
    base = intent_conf
    if exec_conf == "HIGH":
        base = min(0.99, base + 0.08)
    elif exec_conf == "LOW":
        base = max(0.2, base - 0.15)
    base -= min(0.2, ambiguity_count * 0.03)
    return max(0.15, min(0.99, base))


def synthesize_truth_state(data_used: Optional[Dict[str, Any]] = None) -> BrokerTruthState:
    """Assemble truth from stamped layer outputs — does not re-decide."""
    du = data_used if isinstance(data_used, dict) else {}

    intent_raw = du.get("canonical_intent_frame")
    intent_dict = intent_raw if isinstance(intent_raw, dict) else None
    frame = CanonicalIntentFrame.from_dict(intent_dict) if intent_dict else None

    evaluation = du.get("broker_decision")
    if not isinstance(evaluation, dict):
        evaluation = None

    recommendation = du.get("executive_recommendation")
    if not isinstance(recommendation, dict):
        recommendation = None

    constraints: Dict[str, Any] = {}
    adv = du.get("adversarial")
    if isinstance(adv, dict):
        constraints["adversarial"] = adv
    mr = du.get("market_reality")
    if isinstance(mr, dict):
        constraints["market_reality"] = mr

    intent_conf = float(frame.confidence) if frame else 0.5
    amb = len(frame.ambiguity_flags) if frame else 0
    exec_conf = (recommendation or {}).get("confidence") if recommendation else None
    confidence = _confidence_from_parts(intent_conf, exec_conf, amb)

    return BrokerTruthState(
        intent=intent_dict,
        evaluation=evaluation,
        recommendation=recommendation,
        constraints=constraints,
        confidence=round(confidence, 3),
    )


__all__ = ["BrokerTruthState", "synthesize_truth_state"]

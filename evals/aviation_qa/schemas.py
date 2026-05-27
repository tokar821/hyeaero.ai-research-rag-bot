"""
Structured types for aviation QA scenarios and evaluator verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RealismExpectations:
    """What “trusted advisor” realism means for a scenario."""

    min_aircraft_class: str = ""  # light | super-midsize | large | ultra-long
    requires_elimination_language: bool = False
    allows_tech_stop_suggestion: bool = True
    must_not_claim_brochure_nonstop: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RealismExpectations":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            min_aircraft_class=str(raw.get("min_aircraft_class") or ""),
            requires_elimination_language=bool(raw.get("requires_elimination_language")),
            allows_tech_stop_suggestion=bool(raw.get("allows_tech_stop_suggestion", True)),
            must_not_claim_brochure_nonstop=bool(raw.get("must_not_claim_brochure_nonstop")),
            notes=str(raw.get("notes") or ""),
        )


@dataclass
class ScenarioQA:
    """Per-scenario QA expectations (extends golden block)."""

    forbidden_phrases: List[str] = field(default_factory=list)
    required_phrases_any: List[str] = field(default_factory=list)
    realism: RealismExpectations = field(default_factory=RealismExpectations)
    max_recommendations: int = 5
    prefer_short_answer: bool = False

    @classmethod
    def from_case(cls, case: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "ScenarioQA":
        qa = case.get("qa") if isinstance(case.get("qa"), dict) else {}
        d = defaults if isinstance(defaults, dict) else {}
        forbidden = list(qa.get("forbidden_phrases") or d.get("forbidden_phrases") or [])
        required = list(qa.get("required_phrases_any") or [])
        return cls(
            forbidden_phrases=forbidden,
            required_phrases_any=required,
            realism=RealismExpectations.from_dict(qa.get("realism_expectations")),
            max_recommendations=int(qa.get("max_recommendations") or d.get("max_recommendations") or 5),
            prefer_short_answer=bool(qa.get("prefer_short_answer")),
        )


@dataclass
class EvaluatorVerdict:
    """
    Structured evaluator output (critique only — does not answer the user).
    """

    route_realism: str  # PASS | WARN | FAIL
    aircraft_realism: str
    hallucination_risk: float  # 0..1 higher = worse
    repetition_score: float  # 0..1 higher = worse
    humanness_score: float  # 0..1 higher = better
    operational_realism: float  # 0..1 higher = better
    tone_broker_score: float  # 0..1 higher = more broker-like
    fake_confidence_risk: float  # 0..1 higher = worse
    brochure_language_risk: float  # 0..1 higher = worse
    missing_tradeoffs: bool
    main_failure: str
    sub_failures: List[str] = field(default_factory=list)
    passed: bool = False
    trust_score: float = 0.0  # 0..1 composite — trust over completeness

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_realism": self.route_realism,
            "aircraft_realism": self.aircraft_realism,
            "hallucination_risk": round(self.hallucination_risk, 4),
            "repetition_score": round(self.repetition_score, 4),
            "humanness_score": round(self.humanness_score, 4),
            "operational_realism": round(self.operational_realism, 4),
            "tone_broker_score": round(self.tone_broker_score, 4),
            "fake_confidence_risk": round(self.fake_confidence_risk, 4),
            "brochure_language_risk": round(self.brochure_language_risk, 4),
            "missing_tradeoffs": self.missing_tradeoffs,
            "main_failure": self.main_failure,
            "sub_failures": list(self.sub_failures),
            "passed": self.passed,
            "trust_score": round(self.trust_score, 4),
        }

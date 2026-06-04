"""
Recommendation Confidence & Evidence Engine (RCEE) — Phase 22.

Deterministic confidence and evidence scoring only. Does not alter routing or recommendations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CONFIDENCE_ENV = "ENABLE_RECOMMENDATION_CONFIDENCE"

_EVIDENCE_AKAL = 40.0
_EVIDENCE_MARKET = 25.0
_EVIDENCE_MISSION = 20.0
_EVIDENCE_DEAL_KILLER = 10.0
_EVIDENCE_COMPARISON = 5.0

_PENALTY_MISSING_BUDGET = 15.0
_PENALTY_MISSING_PAX = 15.0
_PENALTY_MISSING_ROUTE = 15.0
_PENALTY_NO_COMPS = 20.0
_PENALTY_ALIAS_AMBIGUITY = 10.0
_PENALTY_CONFLICTING_RECORDS = 25.0

_ALIAS_AMBIGUITY_TOKENS = frozenset(
    {
        "longitude",
        "3500",
        "g280",
        "g 280",
        "g-280",
        "cj3+",
        "falcon 8x",
    }
)


@dataclass
class EvidenceItem:
    source_type: str
    source_name: str
    confidence: float
    contribution: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "confidence": round(float(self.confidence), 3),
            "contribution": round(float(self.contribution), 2),
        }


@dataclass
class RecommendationConfidence:
    aircraft: str
    overall_confidence: float
    evidence_score: float
    data_completeness_score: float
    market_confidence: float
    mission_confidence: float
    authority_confidence: float
    confidence_band: str
    warnings: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft": self.aircraft,
            "overall_confidence": round(float(self.overall_confidence), 2),
            "evidence_score": round(float(self.evidence_score), 2),
            "data_completeness_score": round(float(self.data_completeness_score), 2),
            "market_confidence": round(float(self.market_confidence), 2),
            "mission_confidence": round(float(self.mission_confidence), 2),
            "authority_confidence": round(float(self.authority_confidence), 2),
            "confidence_band": self.confidence_band,
            "warnings": list(self.warnings),
            "missing_inputs": list(self.missing_inputs),
            "evidence": [e.to_dict() for e in self.evidence],
        }


def recommendation_confidence_enabled() -> bool:
    return (os.getenv(_CONFIDENCE_ENV) or "").strip().lower() in ("1", "true", "yes")


def confidence_band(score: float) -> str:
    value = float(score)
    if value >= 90:
        return "VERY_HIGH"
    if value >= 70:
        return "HIGH"
    if value >= 40:
        return "MODERATE"
    return "LOW"


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_alias_key(token: str) -> str:
    return re.sub(r"\s+", " ", (token or "").strip().lower())


def _detect_alias_ambiguity(aircraft: str) -> bool:
    key = _normalize_alias_key(aircraft)
    if key in _ALIAS_AMBIGUITY_TOKENS:
        return True
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    canonical = resolve_aircraft_alias(aircraft)
    raw = (aircraft or "").strip()
    if canonical and raw and canonical.lower() != raw.lower():
        return True
    return False


def _mission_confidence_score(mission: Any) -> Tuple[float, List[str], List[str]]:
    """Score mission input completeness (0–100)."""
    warnings: List[str] = []
    missing: List[str] = []

    has_pax = bool(getattr(mission, "passenger_count", None))
    routes = getattr(mission, "routes", None) or []
    has_route = bool(routes)
    has_budget = bool(getattr(mission, "budget_usd", None))

    if not has_pax:
        missing.append("passenger_count")
    if not has_route:
        missing.append("route")
    if not has_budget:
        missing.append("budget")

    if has_pax and has_route and has_budget:
        return 88.0, warnings, missing
    if has_pax and has_route and not has_budget:
        warnings.append("budget not specified; acquisition fit inferred without price ceiling")
        return 58.0, warnings, missing
    if not has_route:
        warnings.append("route missing; range validation uses inferred defaults")
        return 32.0, warnings, missing
    if not has_pax:
        warnings.append("passenger count missing; cabin fit uses class defaults")
        return 45.0, warnings, missing
    return 50.0, warnings, missing


def _buy_decision_confidence(
    market: Optional[Dict[str, Any]],
    deal_killer: Optional[Dict[str, Any]],
) -> Tuple[float, List[str], List[str]]:
    """Score buy-decision evidence (0–100)."""
    warnings: List[str] = []
    missing: List[str] = []
    ctx = market if isinstance(market, dict) else {}
    dk = deal_killer if isinstance(deal_killer, dict) else {}

    has_comps = ctx.get("status") == "OK" and isinstance(ctx.get("expected_market_band_usd"), dict)
    ask_usd = dk.get("ask_usd") or ctx.get("ask_usd")
    tail = dk.get("tail") or dk.get("tail_number") or ctx.get("tail")
    hypothetical = bool(dk.get("hypothetical") or ctx.get("hypothetical_ask"))

    if not has_comps:
        missing.append("market_comps")
        warnings.append("no verified market comparables available")
        return 28.0, warnings, missing

    if tail and ask_usd and has_comps:
        return 88.0, warnings, missing
    if ask_usd and has_comps and not tail:
        warnings.append("ask price present without verified tail number")
        return 62.0, warnings, missing
    if hypothetical:
        warnings.append("hypothetical ask; valuation not tied to a specific listing")
        return 55.0, warnings, missing
    return 45.0, warnings, missing


def evaluate_data_completeness(
    *,
    aircraft: str,
    mission: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Percentage completeness across aircraft specs, market context, mission fields, listing details.
    """
    du = data_used if isinstance(data_used, dict) else {}
    checks = 0
    passed = 0

    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    checks += 1
    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    if rec and rec.nbaa_range_nm > 0 and rec.passenger_capacity_max > 0:
        passed += 1

    checks += 1
    market = du.get("aircraft_authority_market")
    if isinstance(market, dict) and market.get("status") == "OK":
        passed += 1

    checks += 1
    if mission is not None:
        has_mission = bool(
            getattr(mission, "passenger_count", None)
            or getattr(mission, "routes", None)
            or getattr(mission, "budget_usd", None)
        )
        if has_mission:
            passed += 1
    elif isinstance(du.get("mission_state"), dict):
        passed += 1

    checks += 1
    dk = du.get("deal_killer")
    listing_ok = False
    if isinstance(dk, dict):
        listing_ok = bool(dk.get("tail") or dk.get("tail_number") or dk.get("ask_usd"))
    if isinstance(market, dict) and market.get("ask_usd"):
        listing_ok = True
    if listing_ok:
        passed += 1

    if checks == 0:
        return 0.0
    return _clamp((passed / checks) * 100.0)


def _collect_evidence(
    *,
    aircraft: str,
    mission: Any,
    data_used: Dict[str, Any],
    is_buy: bool,
) -> Tuple[List[EvidenceItem], float, float, List[str]]:
    """Build evidence registry and authority confidence."""
    evidence: List[EvidenceItem] = []
    warnings: List[str] = []

    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    authority_confidence = 0.0
    if rec is not None:
        authority_confidence = _clamp(rec.confidence * 100.0)
        evidence.append(
            EvidenceItem(
                source_type="aircraft_authority",
                source_name="Aircraft Authority",
                confidence=rec.confidence,
                contribution=_EVIDENCE_AKAL,
            )
        )

    market = data_used.get("aircraft_authority_market")
    if isinstance(market, dict) and market.get("status") == "OK":
        evidence.append(
            EvidenceItem(
                source_type="market_database",
                source_name="Market Database",
                confidence=0.9,
                contribution=_EVIDENCE_MARKET,
            )
        )

    mission_score, mission_warnings, _ = _mission_confidence_score(mission)
    if mission_score >= 70:
        evidence.append(
            EvidenceItem(
                source_type="mission_validation",
                source_name="Mission Validation",
                confidence=mission_score / 100.0,
                contribution=_EVIDENCE_MISSION,
            )
        )
    warnings.extend(mission_warnings)

    deal_killer = data_used.get("deal_killer")
    if isinstance(deal_killer, dict) and deal_killer.get("verdict"):
        evidence.append(
            EvidenceItem(
                source_type="deal_killer",
                source_name="Deal Killer",
                confidence=0.85,
                contribution=_EVIDENCE_DEAL_KILLER,
            )
        )

    comp_ds = data_used.get("authoritative_comparison_dataset")
    if isinstance(comp_ds, dict) and comp_ds.get("status") == "OK":
        evidence.append(
            EvidenceItem(
                source_type="comparison_dataset",
                source_name="Comparison Dataset",
                confidence=0.8,
                contribution=_EVIDENCE_COMPARISON,
            )
        )

    evidence_score = _clamp(sum(e.contribution for e in evidence))
    return evidence, evidence_score, authority_confidence, warnings


def _compute_penalties(
    *,
    aircraft: str,
    mission: Any,
    data_used: Dict[str, Any],
    is_buy: bool,
) -> Tuple[float, List[str], List[str]]:
    penalty = 0.0
    warnings: List[str] = []
    missing: List[str] = []

    if not getattr(mission, "budget_usd", None):
        penalty += _PENALTY_MISSING_BUDGET
        missing.append("budget")
        warnings.append("missing mission budget reduces acquisition confidence")
    if not getattr(mission, "passenger_count", None):
        penalty += _PENALTY_MISSING_PAX
        missing.append("passenger_count")
        warnings.append("missing passenger count reduces cabin-fit confidence")
    routes = getattr(mission, "routes", None) or []
    if not routes:
        penalty += _PENALTY_MISSING_ROUTE
        missing.append("route")
        warnings.append("missing route reduces range-fit confidence")

    market = data_used.get("aircraft_authority_market")
    if is_buy:
        has_comps = isinstance(market, dict) and market.get("status") == "OK"
        if not has_comps:
            penalty += _PENALTY_NO_COMPS
            missing.append("market_comps")
            warnings.append("no market comparables for buy-decision confidence")

    if _detect_alias_ambiguity(aircraft):
        penalty += _PENALTY_ALIAS_AMBIGUITY
        warnings.append(f"alias ambiguity detected for '{aircraft}'")

    comp_ds = data_used.get("authoritative_comparison_dataset")
    if isinstance(comp_ds, dict):
        if comp_ds.get("status") != "OK" or comp_ds.get("missing"):
            penalty += _PENALTY_CONFLICTING_RECORDS
            warnings.append("conflicting or incomplete aircraft records in comparison dataset")

    return penalty, warnings, missing


def _confidence_for_aircraft(
    aircraft: str,
    *,
    mission: Any,
    data_used: Dict[str, Any],
    is_buy: bool,
) -> RecommendationConfidence:
    evidence, evidence_score, authority_confidence, ev_warnings = _collect_evidence(
        aircraft=aircraft,
        mission=mission,
        data_used=data_used,
        is_buy=is_buy,
    )
    data_completeness = evaluate_data_completeness(
        aircraft=aircraft, mission=mission, data_used=data_used
    )
    mission_confidence, mission_warnings, mission_missing = _mission_confidence_score(mission)
    market_confidence, market_warnings, market_missing = _buy_decision_confidence(
        data_used.get("aircraft_authority_market"),
        data_used.get("deal_killer"),
    )
    if not is_buy:
        market_confidence = max(market_confidence, 50.0) if data_used.get("aircraft_authority_market") else 40.0

    penalty, pen_warnings, pen_missing = _compute_penalties(
        aircraft=aircraft,
        mission=mission,
        data_used=data_used,
        is_buy=is_buy,
    )

    composite = (
        evidence_score * 0.45
        + data_completeness * 0.25
        + mission_confidence * 0.15
        + (market_confidence if is_buy else authority_confidence) * 0.15
    )
    overall = _clamp(composite - penalty * 0.5)

    warnings = list(dict.fromkeys(ev_warnings + mission_warnings + market_warnings + pen_warnings))
    missing_inputs = list(dict.fromkeys(mission_missing + market_missing + pen_missing))

    return RecommendationConfidence(
        aircraft=aircraft,
        overall_confidence=overall,
        evidence_score=evidence_score,
        data_completeness_score=data_completeness,
        market_confidence=market_confidence,
        mission_confidence=mission_confidence,
        authority_confidence=authority_confidence,
        confidence_band=confidence_band(overall),
        warnings=warnings,
        missing_inputs=missing_inputs,
        evidence=evidence,
    )


def _extract_aircraft_list(data_used: Dict[str, Any]) -> List[str]:
    aircraft: List[str] = []
    rec_rows = data_used.get("consultant_recommendations")
    if isinstance(rec_rows, list):
        for row in rec_rows:
            if isinstance(row, dict):
                name = str(row.get("model") or row.get("aircraft") or "").strip()
                if name:
                    aircraft.append(name)

    alt = data_used.get("alternative_execution")
    if isinstance(alt, dict):
        target = str(alt.get("target") or "").strip()
        if target:
            aircraft.insert(0, target)

    comp_ds = data_used.get("authoritative_comparison_dataset")
    if isinstance(comp_ds, dict):
        for row in comp_ds.get("aircraft") or []:
            if isinstance(row, dict):
                name = str(row.get("canonical_name") or "").strip()
                if name and name not in aircraft:
                    aircraft.append(name)

    market = data_used.get("aircraft_authority_market")
    if isinstance(market, dict):
        name = str(market.get("canonical_name") or "").strip()
        if name and name not in aircraft:
            aircraft.append(name)

    return aircraft[:8]


def build_recommendation_confidence(
    query: str,
    response: Any,
) -> Dict[str, Any]:
    """Build confidence bundle for all recommended aircraft in a response payload."""
    payload = response if isinstance(response, dict) else {}
    du = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}

    from services.consultant.mission_state import MissionState, build_mission_from_current_turn

    mission = build_mission_from_current_turn(query or "")
    if isinstance(du.get("mission_state"), dict):
        msd = du["mission_state"]
        mission = MissionState(
            passenger_count=msd.get("passenger_count"),
            routes=list(msd.get("routes") or []),
            budget_usd=msd.get("budget_usd"),
            nonstop_requirement=msd.get("nonstop_requirement"),
        )

    is_buy = bool(du.get("deal_killer") or du.get("aircraft_authority_market"))
    aircraft_list = _extract_aircraft_list(du)
    if not aircraft_list:
        aircraft_list = ["unknown"]

    scores = [
        _confidence_for_aircraft(ac, mission=mission, data_used=du, is_buy=is_buy)
        for ac in aircraft_list
    ]

    panel = {
        "confidence_score": scores[0].overall_confidence if scores else 0.0,
        "confidence_band": scores[0].confidence_band if scores else "LOW",
        "evidence_sources": [e.source_name for s in scores for e in s.evidence],
        "missing_information": list(
            dict.fromkeys(m for s in scores for m in s.missing_inputs)
        ),
    }

    return {
        "aircraft_confidence": [s.to_dict() for s in scores],
        "confidence_panel": panel,
    }


def attach_recommendation_confidence_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.recommendation_confidence`` when env flag enabled."""
    if not recommendation_confidence_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["recommendation_confidence"] = build_recommendation_confidence(query, out)
    out["data_used"] = du
    return out


def evaluate_recommendation_confidence_hooks(response: Any) -> List[str]:
    """
    Optional evaluation hooks — confidence consistency, inflation, unsupported high confidence.

    Returns failure tokens for consultant_evaluator integration. Does not mutate *response*.
    """
    if not isinstance(response, dict):
        return []
    du = response.get("data_used")
    if not isinstance(du, dict):
        return []
    bundle = du.get("recommendation_confidence")
    if not isinstance(bundle, dict):
        return []

    failures: List[str] = []
    rows = bundle.get("aircraft_confidence")
    if not isinstance(rows, list):
        return failures

    for row in rows:
        if not isinstance(row, dict):
            continue
        overall = float(row.get("overall_confidence") or 0)
        evidence = float(row.get("evidence_score") or 0)
        completeness = float(row.get("data_completeness_score") or 0)
        band = str(row.get("confidence_band") or "")
        missing = row.get("missing_inputs") or []

        if overall > evidence + 25:
            failures.append("confidence_inflation")
        if band in ("HIGH", "VERY_HIGH") and completeness < 50:
            failures.append("unsupported_high_confidence")
        if band == "VERY_HIGH" and missing:
            failures.append("confidence_consistency")
        if overall >= 70 and evidence < 40:
            failures.append("confidence_consistency")

    return list(dict.fromkeys(failures))


__all__ = [
    "EvidenceItem",
    "RecommendationConfidence",
    "attach_recommendation_confidence_if_enabled",
    "build_recommendation_confidence",
    "confidence_band",
    "evaluate_data_completeness",
    "evaluate_recommendation_confidence_hooks",
    "recommendation_confidence_enabled",
]

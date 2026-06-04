"""
Recommendation Justification Engine (RJE) — deterministic, auditable recommendation reasoning.

Advisory transparency only. Does not alter routing or response shaping.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_JUSTIFY_ENV = "ENABLE_RECOMMENDATION_JUSTIFICATION"

_FORBIDDEN_REASON_RE = re.compile(
    r"\b(?:great\s+aircraft|excellent\s+choice|popular\s+option|best\s+in\s+class|"
    r"world[-\s]?class|amazing|perfect\s+choice)\b",
    re.I,
)


@dataclass
class RecommendationJustification:
    recommendation_id: str
    aircraft: str
    recommendation_rank: int
    recommendation_reason: str
    mission_alignment_score: float
    budget_alignment_score: float
    range_alignment_score: float
    passenger_alignment_score: float
    operating_cost_alignment_score: float
    rejection_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "aircraft": self.aircraft,
            "recommendation_rank": self.recommendation_rank,
            "recommendation_reason": self.recommendation_reason,
            "mission_alignment_score": round(float(self.mission_alignment_score), 2),
            "budget_alignment_score": round(float(self.budget_alignment_score), 2),
            "range_alignment_score": round(float(self.range_alignment_score), 2),
            "passenger_alignment_score": round(float(self.passenger_alignment_score), 2),
            "operating_cost_alignment_score": round(float(self.operating_cost_alignment_score), 2),
            "rejection_reasons": list(self.rejection_reasons),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class RejectedAircraft:
    aircraft: str
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"aircraft": self.aircraft, "rejection_reason": self.rejection_reason}


@dataclass
class DecisionFactors:
    winner: str
    factors: List[str]
    per_aircraft: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "factors": list(self.factors),
            "per_aircraft": {k: list(v) for k, v in self.per_aircraft.items()},
        }


def recommendation_justification_enabled() -> bool:
    return (os.getenv(_JUSTIFY_ENV) or "").strip().lower() in ("1", "true", "yes")


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _sanitize_reason(text: str) -> str:
    cleaned = _FORBIDDEN_REASON_RE.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;")


def _score_range(required_nm: Optional[float], aircraft_nm: float) -> Tuple[float, str]:
    if not required_nm or required_nm <= 0:
        margin = aircraft_nm
        score = _clamp_score(min(100.0, (aircraft_nm / 4000.0) * 85.0))
        return score, f"catalog practical range {int(aircraft_nm)} nm available"
    margin = aircraft_nm - required_nm
    if margin < 0:
        return 0.0, f"insufficient range by {int(abs(margin))} nm versus {int(required_nm)} nm requirement"
    score = _clamp_score(70.0 + min(30.0, (margin / max(required_nm, 1)) * 100.0))
    return score, f"exceeds required range by {int(margin)} nm"


def _score_passengers(required: Optional[int], pax_min: int, pax_max: int) -> Tuple[float, str]:
    if not required or required <= 0:
        return 85.0, f"seating capacity up to {pax_max} passengers"
    if required > pax_max:
        short = required - pax_max
        return 0.0, f"passenger shortfall of {short} versus {pax_max}-seat envelope"
    margin = pax_max - required
    score = _clamp_score(75.0 + min(25.0, margin * 4.0))
    return score, f"fits {required} passengers with {margin}-seat margin"


def _score_budget(budget_usd: Optional[float], typical_price: Optional[float]) -> Tuple[float, str]:
    if not budget_usd or budget_usd <= 0:
        return 80.0, "no budget constraint supplied"
    if not typical_price or typical_price <= 0:
        return 70.0, "acquisition band not verified in catalog"
    ratio = typical_price / budget_usd
    if ratio > 1.18:
        pct = int((ratio - 1.0) * 100)
        return 0.0, f"exceeds budget by {pct}%"
    if ratio <= 0.85:
        return 95.0, "acquisition cost within budget band with margin"
    score = _clamp_score(100.0 - (ratio - 0.85) * 200.0)
    return score, "acquisition cost within stated budget band"


def _score_operating_cost(operating_index: float) -> Tuple[float, str]:
    oi = float(operating_index or 0.65)
    score = _clamp_score(100.0 - oi * 70.0)
    return score, f"operating index {oi:.2f} versus class baseline"


def _alignment_scores(
    *,
    aircraft: str,
    mission: Any = None,
    budget_usd: Optional[float] = None,
) -> Tuple[Dict[str, float], List[str], float]:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    if rec is None:
        return (
            {
                "mission_alignment_score": 0.0,
                "budget_alignment_score": 0.0,
                "range_alignment_score": 0.0,
                "passenger_alignment_score": 0.0,
                "operating_cost_alignment_score": 0.0,
            },
            ["insufficient verified aircraft record"],
            0.0,
        )

    required_nm = None
    required_pax = None
    if mission is not None:
        required_pax = getattr(mission, "passenger_count", None)
        routes = getattr(mission, "routes", None) or []
        if routes:
            required_nm = 2500.0
        if getattr(mission, "nonstop_requirement", None):
            required_nm = max(required_nm or 0, 3000.0)
        budget_usd = budget_usd or getattr(mission, "budget_usd", None)

    range_score, range_reason = _score_range(required_nm, rec.nbaa_range_nm)
    pax_score, pax_reason = _score_passengers(required_pax, rec.passenger_capacity_min, rec.passenger_capacity_max)

    typical_price = None
    try:
        from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

        rows = find_catalog_matches([rec.canonical_name])
        if rows:
            typical_price = typical_market_price_usd(rows[0])
    except Exception:
        pass

    budget_score, budget_reason = _score_budget(budget_usd, typical_price)
    op_score, op_reason = _score_operating_cost(
        float((rec.to_profile_dict() or {}).get("operating_index") or 0.65)
    )

    mission_score = _clamp_score(
        range_score * 0.35 + pax_score * 0.25 + budget_score * 0.25 + op_score * 0.15
    )
    reasons = [_sanitize_reason(r) for r in (range_reason, pax_reason, budget_reason, op_reason) if r]
    confidence = rec.confidence * (mission_score / 100.0)

    return (
        {
            "mission_alignment_score": mission_score,
            "budget_alignment_score": budget_score,
            "range_alignment_score": range_score,
            "passenger_alignment_score": pax_score,
            "operating_cost_alignment_score": op_score,
        },
        reasons,
        confidence,
    )


def _primary_rejection_reason(scores: Dict[str, float], reasons: Sequence[str]) -> str:
    """Pick the most specific measurable rejection reason (hard failures first)."""
    if scores["range_alignment_score"] <= 0:
        return _sanitize_reason(
            next(
                (r for r in reasons if "range" in r.lower() or "insufficient" in r.lower()),
                "insufficient range",
            )
        )
    if scores["budget_alignment_score"] <= 0:
        return _sanitize_reason(
            next(
                (r for r in reasons if "budget" in r.lower() or "exceeds" in r.lower()),
                "acquisition cost outside target band",
            )
        )
    if scores["passenger_alignment_score"] <= 0:
        return _sanitize_reason(
            next(
                (r for r in reasons if "passenger" in r.lower()),
                "insufficient passenger capacity",
            )
        )
    budget_reason = next((r for r in reasons if "exceeds budget" in r.lower()), None)
    if budget_reason:
        return _sanitize_reason(budget_reason)
    if scores["mission_alignment_score"] >= 70:
        return (
            f"lower composite mission alignment ({scores['mission_alignment_score']:.0f}) "
            "versus selected option"
        )
    return _sanitize_reason(
        reasons[0] if reasons else "lower mission alignment versus selected aircraft"
    )


def build_rejection_analysis(
    candidates: Sequence[str],
    winner: str,
    *,
    mission: Any = None,
    budget_usd: Optional[float] = None,
) -> List[RejectedAircraft]:
    """Explain why non-winning candidates were rejected."""
    rejections: List[RejectedAircraft] = []
    winner_key = (winner or "").strip().lower()
    for cand in candidates:
        name = str(cand or "").strip()
        if not name or name.lower() == winner_key:
            continue
        scores, reasons, _ = _alignment_scores(aircraft=name, mission=mission, budget_usd=budget_usd)
        reason = _primary_rejection_reason(scores, reasons)
        rejections.append(RejectedAircraft(aircraft=name, rejection_reason=reason))
    return rejections


def build_comparison_justification(
    comparison_dataset: Dict[str, Any],
) -> Optional[DecisionFactors]:
    """Build winner factors from authoritative comparison dataset."""
    rows = comparison_dataset.get("aircraft") if isinstance(comparison_dataset, dict) else None
    if not isinstance(rows, list) or len(rows) < 2:
        return None

    parsed: List[Dict[str, Any]] = [r for r in rows if isinstance(r, dict)]
    if len(parsed) < 2:
        return None

    def _range(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("range_nm") or 0)
        except (TypeError, ValueError):
            return 0.0

    def _speed(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("speed_ktas") or 0)
        except (TypeError, ValueError):
            return 0.0

    winner_row = max(parsed, key=_range)
    winner = str(winner_row.get("canonical_name") or "")
    factors: List[str] = []
    per: Dict[str, List[str]] = {}

    max_range = _range(winner_row)
    max_speed_row = max(parsed, key=_speed)
    min_baggage = min(parsed, key=lambda r: float(r.get("baggage") or 1.0))

    factors.append(f"{winner} leads on verified range at {int(max_range)} nm")
    if _speed(max_speed_row) > 0:
        factors.append(
            f"{max_speed_row.get('canonical_name')} leads on catalog cruise speed at {int(_speed(max_speed_row))} ktas"
        )

    for row in parsed:
        name = str(row.get("canonical_name") or "")
        notes: List[str] = []
        if _range(row) == max_range:
            notes.append("greater range")
        if _speed(row) == _speed(max_speed_row) and _speed(row) > 0:
            notes.append("higher speed")
        if row is min_baggage:
            notes.append("lower operating cost band proxy via baggage/economics index")
        if name != winner and _range(row) < max_range:
            notes.append("shorter verified range versus leader")
        per[name] = notes

    return DecisionFactors(winner=winner, factors=[_sanitize_reason(f) for f in factors if f], per_aircraft=per)


def build_buy_decision_justification(
    *,
    market_context: Dict[str, Any],
    deal_killer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic buy-decision reasoning from market context and deal killer."""
    ctx = market_context if isinstance(market_context, dict) else {}
    dk = deal_killer if isinstance(deal_killer, dict) else {}
    verdict = str(dk.get("verdict") or dk.get("deal_verdict") or "").strip()
    band = ctx.get("expected_market_band_usd") if isinstance(ctx.get("expected_market_band_usd"), dict) else {}

    why_good: List[str] = []
    why_risky: List[str] = []
    valuation_drivers: List[str] = []

    ask_pos = str(ctx.get("ask_position") or "")
    if ask_pos == "below_market":
        why_good.append("ask price sits below verified market band midpoint")
    elif ask_pos == "in_band":
        why_good.append("ask price aligns with catalog-derived market band")
    elif ask_pos == "above_market":
        why_risky.append("ask price sits above verified market band")

    dep = str(ctx.get("depreciation_band") or "")
    if dep == "elevated":
        why_risky.append("airframe age implies elevated depreciation exposure")
    elif dep == "moderate":
        valuation_drivers.append("moderate depreciation curve assumed for age class")

    age_pos = str(ctx.get("age_position") or "")
    if age_pos == "young":
        valuation_drivers.append("young airframe age supports residual value")
    elif age_pos == "mature":
        why_risky.append("mature airframe age increases lifecycle cost risk")

    if band:
        valuation_drivers.append(
            f"market band reference ${band.get('low', 0)/1e6:.1f}M–${band.get('high', 0)/1e6:.1f}M"
        )

    if verdict:
        vlow = verdict.lower()
        if "good" in vlow or "fair" in vlow:
            why_good.append(f"deal killer verdict: {verdict}")
        elif "over" in vlow or "risk" in vlow or "pass" in vlow:
            why_risky.append(f"deal killer verdict: {verdict}")

    return {
        "why_deal_is_good": [_sanitize_reason(x) for x in why_good if x],
        "why_deal_is_risky": [_sanitize_reason(x) for x in why_risky if x],
        "valuation_drivers": [_sanitize_reason(x) for x in valuation_drivers if x],
    }


def build_mission_justification(
    mission: Any,
    *,
    selected_aircraft: Optional[str] = None,
) -> Dict[str, Any]:
    """Explain mission-fit reasoning for a selected or top-ranked aircraft."""
    from services.consultant.mission_state import MissionState

    ms = mission if isinstance(mission, MissionState) else mission
    constraints: List[str] = []
    assumptions: List[str] = []

    if getattr(ms, "passenger_count", None):
        constraints.append(f"passengers: {ms.passenger_count}")
    if getattr(ms, "routes", None):
        constraints.append(f"routes: {', '.join(ms.routes[:3])}")
    if getattr(ms, "budget_usd", None):
        constraints.append(f"budget: ${ms.budget_usd/1e6:.1f}M")
    if getattr(ms, "nonstop_requirement", None):
        constraints.append("nonstop operation required")

    if not constraints:
        assumptions.append("mission profile partially unspecified; range requirement inferred at 2500 nm when routes absent")

    why_fits: List[str] = []
    if selected_aircraft:
        _, reasons, _ = _alignment_scores(aircraft=selected_aircraft, mission=ms)
        why_fits.extend(reasons[:4])

    return {
        "why_aircraft_fits": [_sanitize_reason(r) for r in why_fits if r],
        "mission_constraints": constraints,
        "assumptions": assumptions,
        "primary_constraint": constraints[0] if constraints else "range",
    }


def _stable_recommendation_id(aircraft: str, rank: int, *, mission: Any = None) -> str:
    parts = [aircraft.strip().lower(), str(rank)]
    if mission is not None:
        parts.extend(
            [
                str(getattr(mission, "passenger_count", "") or ""),
                str(getattr(mission, "budget_usd", "") or ""),
                ",".join(getattr(mission, "routes", None) or []),
                str(bool(getattr(mission, "nonstop_requirement", False))),
            ]
        )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def _build_recommendation_justification_row(
    aircraft: str,
    rank: int,
    *,
    mission: Any = None,
    budget_usd: Optional[float] = None,
    rejection_reasons: Optional[List[str]] = None,
) -> RecommendationJustification:
    scores, reasons, confidence = _alignment_scores(
        aircraft=aircraft, mission=mission, budget_usd=budget_usd
    )
    primary = _sanitize_reason(reasons[0] if reasons else "verified catalog alignment")
    return RecommendationJustification(
        recommendation_id=_stable_recommendation_id(aircraft, rank, mission=mission),
        aircraft=aircraft,
        recommendation_rank=rank,
        recommendation_reason=primary,
        mission_alignment_score=scores["mission_alignment_score"],
        budget_alignment_score=scores["budget_alignment_score"],
        range_alignment_score=scores["range_alignment_score"],
        passenger_alignment_score=scores["passenger_alignment_score"],
        operating_cost_alignment_score=scores["operating_cost_alignment_score"],
        rejection_reasons=list(rejection_reasons or []),
        confidence=confidence,
    )


def build_recommendation_justification(
    query: str,
    response: Any,
) -> Dict[str, Any]:
    """
    Build full justification bundle from a consultant response payload.

    Deterministic — no LLM reasoning.
    """
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

    recommendations: List[RecommendationJustification] = []
    rejections: List[RejectedAircraft] = []
    comparison: Optional[DecisionFactors] = None
    buy: Optional[Dict[str, Any]] = None
    mission_block: Optional[Dict[str, Any]] = None

    rec_rows = du.get("consultant_recommendations")
    if isinstance(rec_rows, list) and rec_rows:
        ranked = [str(r.get("model") or r.get("aircraft") or "") for r in rec_rows if isinstance(r, dict)]
        ranked = [m for m in ranked if m]
        winner = ranked[0] if ranked else ""
        for idx, model in enumerate(ranked[:5], start=1):
            recommendations.append(
                _build_recommendation_justification_row(model, idx, mission=mission)
            )
        if winner and len(ranked) > 1:
            rejections = build_rejection_analysis(ranked[1:], winner, mission=mission)
        mission_block = build_mission_justification(mission, selected_aircraft=winner)

    alt = du.get("alternative_execution")
    if isinstance(alt, dict):
        target = str(alt.get("target") or "")
        candidates = [str(c) for c in (alt.get("candidates") or []) if c]
        if target and candidates:
            recommendations.insert(
                0,
                _build_recommendation_justification_row(target, 0, mission=mission),
            )
            rejections.extend(build_rejection_analysis(candidates, target, mission=mission))

    comp_ds = du.get("authoritative_comparison_dataset")
    if not isinstance(comp_ds, dict):
        models = []
        icrl = du.get("intent_conflict_resolution") or {}
        if isinstance(icrl, dict):
            graph = icrl.get("graph") or {}
            if isinstance(graph, dict):
                models = list(graph.get("entities") or [])
        if len(models) >= 2:
            from services.aircraft.aircraft_authority_service import build_authoritative_comparison_dataset

            comp_ds = build_authoritative_comparison_dataset(models)
            du = {**du, "authoritative_comparison_dataset": comp_ds}
    if isinstance(comp_ds, dict):
        comparison = build_comparison_justification(comp_ds)

    market = du.get("aircraft_authority_market")
    deal_killer = du.get("deal_killer")
    if isinstance(market, dict):
        buy = build_buy_decision_justification(market_context=market, deal_killer=deal_killer)

    if not mission_block and recommendations:
        mission_block = build_mission_justification(
            mission, selected_aircraft=recommendations[0].aircraft
        )
    elif not mission_block:
        mission_block = build_mission_justification(mission)

    panel = {
        "recommendation_reasons": [r.recommendation_reason for r in recommendations],
        "rejection_reasons": [r.rejection_reason for r in rejections],
        "scoring_factors": [
            {
                "aircraft": r.aircraft,
                "mission_alignment_score": r.mission_alignment_score,
                "range_alignment_score": r.range_alignment_score,
                "budget_alignment_score": r.budget_alignment_score,
            }
            for r in recommendations
        ],
    }

    return {
        "recommendations": [r.to_dict() for r in recommendations],
        "rejections": [r.to_dict() for r in rejections],
        "comparison": comparison.to_dict() if comparison else None,
        "buy_decision": buy,
        "mission": mission_block,
        "justification_panel": panel,
    }


def attach_recommendation_justification_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.recommendation_justification`` when env flag enabled."""
    if not recommendation_justification_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["recommendation_justification"] = build_recommendation_justification(query, out)
    out["data_used"] = du
    return out


__all__ = [
    "DecisionFactors",
    "RecommendationJustification",
    "RejectedAircraft",
    "attach_recommendation_justification_if_enabled",
    "build_buy_decision_justification",
    "build_comparison_justification",
    "build_mission_justification",
    "build_recommendation_justification",
    "build_rejection_analysis",
    "recommendation_justification_enabled",
]

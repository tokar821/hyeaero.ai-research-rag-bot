"""
Multi-Criteria Decision Optimization Engine (MCDO) — Phase 23.

Deterministic ranking optimization only. Does not alter routing or execution flow.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_OPT_ENV = "ENABLE_DECISION_OPTIMIZATION"

_CATEGORY_LIQUIDITY = {
    "light": 0.72,
    "super-midsize": 0.82,
    "large-cabin": 0.88,
    "ultra-long": 0.85,
    "turboprop": 0.70,
}


@dataclass(frozen=True)
class DecisionProfile:
    name: str
    acquisition_cost_weight: float
    operating_cost_weight: float
    range_weight: float
    cabin_weight: float
    speed_weight: float
    resale_weight: float
    market_liquidity_weight: float
    ownership_risk_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "acquisition_cost_weight": self.acquisition_cost_weight,
            "operating_cost_weight": self.operating_cost_weight,
            "range_weight": self.range_weight,
            "cabin_weight": self.cabin_weight,
            "speed_weight": self.speed_weight,
            "resale_weight": self.resale_weight,
            "market_liquidity_weight": self.market_liquidity_weight,
            "ownership_risk_weight": self.ownership_risk_weight,
        }

    @property
    def weight_total(self) -> float:
        return (
            self.acquisition_cost_weight
            + self.operating_cost_weight
            + self.range_weight
            + self.cabin_weight
            + self.speed_weight
            + self.resale_weight
            + self.market_liquidity_weight
            + self.ownership_risk_weight
        )


STANDARD_BUYER = DecisionProfile(
    name="STANDARD_BUYER",
    acquisition_cost_weight=15,
    operating_cost_weight=15,
    range_weight=20,
    cabin_weight=15,
    speed_weight=10,
    resale_weight=10,
    market_liquidity_weight=10,
    ownership_risk_weight=5,
)

COST_FOCUSED = DecisionProfile(
    name="COST_FOCUSED",
    acquisition_cost_weight=40,
    operating_cost_weight=25,
    range_weight=10,
    cabin_weight=5,
    speed_weight=0,
    resale_weight=20,
    market_liquidity_weight=0,
    ownership_risk_weight=0,
)

RANGE_FOCUSED = DecisionProfile(
    name="RANGE_FOCUSED",
    acquisition_cost_weight=0,
    operating_cost_weight=15,
    range_weight=50,
    cabin_weight=10,
    speed_weight=15,
    resale_weight=10,
    market_liquidity_weight=0,
    ownership_risk_weight=0,
)

CABIN_FOCUSED = DecisionProfile(
    name="CABIN_FOCUSED",
    acquisition_cost_weight=10,
    operating_cost_weight=10,
    range_weight=10,
    cabin_weight=45,
    speed_weight=5,
    resale_weight=10,
    market_liquidity_weight=5,
    ownership_risk_weight=5,
)

RESALE_FOCUSED = DecisionProfile(
    name="RESALE_FOCUSED",
    acquisition_cost_weight=10,
    operating_cost_weight=10,
    range_weight=10,
    cabin_weight=5,
    speed_weight=0,
    resale_weight=40,
    market_liquidity_weight=25,
    ownership_risk_weight=0,
)

CORPORATE_FLIGHT_DEPARTMENT = DecisionProfile(
    name="CORPORATE_FLIGHT_DEPARTMENT",
    acquisition_cost_weight=10,
    operating_cost_weight=15,
    range_weight=25,
    cabin_weight=20,
    speed_weight=10,
    resale_weight=5,
    market_liquidity_weight=5,
    ownership_risk_weight=10,
)

CHARTER_OPERATOR = DecisionProfile(
    name="CHARTER_OPERATOR",
    acquisition_cost_weight=20,
    operating_cost_weight=30,
    range_weight=15,
    cabin_weight=10,
    speed_weight=5,
    resale_weight=10,
    market_liquidity_weight=15,
    ownership_risk_weight=5,
)

BUYER_PROFILES: Dict[str, DecisionProfile] = {
    p.name: p
    for p in (
        STANDARD_BUYER,
        COST_FOCUSED,
        RANGE_FOCUSED,
        CABIN_FOCUSED,
        RESALE_FOCUSED,
        CORPORATE_FLIGHT_DEPARTMENT,
        CHARTER_OPERATOR,
    )
}


@dataclass
class AircraftOptimizationScore:
    aircraft: str
    total_score: float
    acquisition_score: float
    operating_score: float
    range_score: float
    cabin_score: float
    speed_score: float = 0.0
    resale_score: float = 0.0
    liquidity_score: float = 0.0
    risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft": self.aircraft,
            "total_score": round(float(self.total_score), 2),
            "acquisition_score": round(float(self.acquisition_score), 2),
            "operating_score": round(float(self.operating_score), 2),
            "range_score": round(float(self.range_score), 2),
            "cabin_score": round(float(self.cabin_score), 2),
            "speed_score": round(float(self.speed_score), 2),
            "resale_score": round(float(self.resale_score), 2),
            "liquidity_score": round(float(self.liquidity_score), 2),
            "risk_score": round(float(self.risk_score), 2),
        }


@dataclass
class OptimizationResult:
    winner: str
    ranked_candidates: List[AircraftOptimizationScore]
    score_breakdown: Dict[str, Dict[str, float]]
    tradeoffs: List[str]
    buyer_profile: DecisionProfile
    optimization_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "ranked_candidates": [r.to_dict() for r in self.ranked_candidates],
            "score_breakdown": {
                k: {dk: round(float(dv), 2) for dk, dv in v.items()}
                for k, v in self.score_breakdown.items()
            },
            "tradeoffs": list(self.tradeoffs),
            "buyer_profile": self.buyer_profile.to_dict(),
            "optimization_id": self.optimization_id,
        }


def decision_optimization_enabled() -> bool:
    return (os.getenv(_OPT_ENV) or "").strip().lower() in ("1", "true", "yes")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def infer_buyer_profile(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> DecisionProfile:
    """Infer buyer profile from query keywords or explicit data_used override."""
    du = data_used if isinstance(data_used, dict) else {}
    explicit = du.get("buyer_profile") or du.get("decision_profile")
    if isinstance(explicit, str) and explicit.upper() in BUYER_PROFILES:
        return BUYER_PROFILES[explicit.upper()]
    if isinstance(explicit, dict):
        name = str(explicit.get("name") or "STANDARD_BUYER").upper()
        if name in BUYER_PROFILES:
            return BUYER_PROFILES[name]

    q = (query or "").lower()
    if re.search(r"\b(?:charter|135|part\s+135)\b", q):
        return CHARTER_OPERATOR
    if re.search(r"\b(?:corporate\s+flight|flight\s+department|cfd)\b", q):
        return CORPORATE_FLIGHT_DEPARTMENT
    if re.search(r"\b(?:resale|liquidity|hold\s+value|exit\s+value)\b", q):
        return RESALE_FOCUSED
    if re.search(r"\b(?:cabin|comfort|luxury|spacious|stand[\-\s]?up)\b", q):
        return CABIN_FOCUSED
    if re.search(r"\b(?:range|nonstop|long[\-\s]?range|transcontinental)\b", q):
        return RANGE_FOCUSED
    if re.search(r"\b(?:cost|budget|economical|cheap|affordable|operating\s+cost)\b", q):
        return COST_FOCUSED
    return STANDARD_BUYER


def _normalize_across_candidates(
    values: Dict[str, float],
    *,
    lower_is_better: bool = False,
) -> Dict[str, float]:
    if not values:
        return {}
    nums = list(values.values())
    mn, mx = min(nums), max(nums)
    if mx == mn:
        return {k: 50.0 for k in values}
    out: Dict[str, float] = {}
    for name, val in values.items():
        norm = (val - mn) / (mx - mn) * 100.0
        if lower_is_better:
            norm = 100.0 - norm
        out[name] = _clamp(norm)
    return out


def _raw_metrics(aircraft: str) -> Dict[str, float]:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    if rec is None:
        return {
            "acquisition_usd": 0.0,
            "operating_index": 0.0,
            "range_nm": 0.0,
            "cabin_volume": 0.0,
            "speed_ktas": 0.0,
            "resale_raw": 0.0,
            "liquidity_raw": 0.0,
            "risk_raw": 0.0,
            "_insufficient": True,
        }

    prof = rec.to_profile_dict()
    acquisition = 20_000_000.0
    try:
        from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

        rows = find_catalog_matches([rec.canonical_name])
        if rows:
            price = typical_market_price_usd(rows[0])
            if price and price > 0:
                acquisition = float(price)
    except Exception:
        pass

    cat = rec.aircraft_category or "large-cabin"
    cabin_vol = 100.0
    if rec.cabin_height and rec.cabin_width and rec.cabin_length:
        cabin_vol = float(rec.cabin_height * rec.cabin_width * rec.cabin_length)
    else:
        cabin_vol = float(prof.get("cabin_score") or 0.8) * 200.0

    speed = float(rec.max_cruise_speed or 460.0)
    oi = float(prof.get("operating_index") or 0.65)

    resale_raw = float(prof.get("resale_score") or 0.75)
    if rec.current_in_production:
        resale_raw += 0.08
    elif rec.production_end_year:
        resale_raw -= 0.12

    liquidity_raw = _CATEGORY_LIQUIDITY.get(cat, 0.80) * resale_raw

    risk_raw = 0.5
    risk_raw += 0.25 if rec.current_in_production else -0.15
    risk_raw += float(prof.get("dispatch_score") or 0.75) * 0.35
    if rec.manufacturer in ("Gulfstream", "Bombardier", "Textron", "Dassault", "Embraer"):
        risk_raw += 0.1

    return {
        "acquisition_usd": acquisition,
        "operating_index": oi,
        "range_nm": float(rec.nbaa_range_nm),
        "cabin_volume": cabin_vol,
        "speed_ktas": speed,
        "resale_raw": _clamp(resale_raw * 100.0, 0, 100) / 100.0,
        "liquidity_raw": _clamp(liquidity_raw * 100.0, 0, 100) / 100.0,
        "risk_raw": _clamp(risk_raw * 100.0, 0, 100) / 100.0,
    }


def _resale_score(metrics: Dict[str, float]) -> float:
    """Resale score from market activity, liquidity, age, replacement risk proxies."""
    base = float(metrics.get("resale_raw") or 0.75) * 100.0
    liquidity = float(metrics.get("liquidity_raw") or 0.75) * 100.0
    return _clamp(base * 0.65 + liquidity * 0.35)


def _ownership_risk_score(metrics: Dict[str, float]) -> float:
    """Higher score = lower ownership risk (support, production, ecosystem)."""
    return _clamp(float(metrics.get("risk_raw") or 0.5) * 100.0)


def _weighted_total(score: AircraftOptimizationScore, profile: DecisionProfile) -> float:
    return _clamp(
        (
            score.acquisition_score * profile.acquisition_cost_weight
            + score.operating_score * profile.operating_cost_weight
            + score.range_score * profile.range_weight
            + score.cabin_score * profile.cabin_weight
            + score.speed_score * profile.speed_weight
            + score.resale_score * profile.resale_weight
            + score.liquidity_score * profile.market_liquidity_weight
            + score.risk_score * profile.ownership_risk_weight
        )
        / 100.0
    )


def optimize_aircraft_ranking(
    candidates: Sequence[str],
    *,
    mission: Any = None,
    profile: DecisionProfile = STANDARD_BUYER,
) -> OptimizationResult:
    """Rank candidates using normalized multi-criteria scores and buyer profile weights."""
    names = [str(c).strip() for c in candidates if str(c or "").strip()]
    if not names:
        return OptimizationResult(
            winner="",
            ranked_candidates=[],
            score_breakdown={},
            tradeoffs=[],
            buyer_profile=profile,
            optimization_id="",
        )

    raw_by_aircraft = {name: _raw_metrics(name) for name in names}
    raw_by_aircraft = {
        name: metrics
        for name, metrics in raw_by_aircraft.items()
        if not metrics.get("_insufficient")
    }
    if len(raw_by_aircraft) < 2:
        return OptimizationResult(
            winner="",
            ranked_candidates=[],
            score_breakdown={},
            tradeoffs=[],
            buyer_profile=profile,
            optimization_id="",
        )

    acq_norm = _normalize_across_candidates(
        {n: m["acquisition_usd"] for n, m in raw_by_aircraft.items()},
        lower_is_better=True,
    )
    op_norm = _normalize_across_candidates(
        {n: m["operating_index"] for n, m in raw_by_aircraft.items()},
        lower_is_better=True,
    )
    range_norm = _normalize_across_candidates(
        {n: m["range_nm"] for n, m in raw_by_aircraft.items()},
    )
    cabin_norm = _normalize_across_candidates(
        {n: m["cabin_volume"] for n, m in raw_by_aircraft.items()},
    )
    speed_norm = _normalize_across_candidates(
        {n: m["speed_ktas"] for n, m in raw_by_aircraft.items()},
    )
    resale_norm = {n: _resale_score(m) for n, m in raw_by_aircraft.items()}
    liquidity_norm = _normalize_across_candidates(
        {n: m["liquidity_raw"] for n, m in raw_by_aircraft.items()},
    )
    risk_norm = {n: _ownership_risk_score(m) for n, m in raw_by_aircraft.items()}

    scores: List[AircraftOptimizationScore] = []
    breakdown: Dict[str, Dict[str, float]] = {}

    for name in names:
        row = AircraftOptimizationScore(
            aircraft=name,
            total_score=0.0,
            acquisition_score=acq_norm.get(name, 0.0),
            operating_score=op_norm.get(name, 0.0),
            range_score=range_norm.get(name, 0.0),
            cabin_score=cabin_norm.get(name, 0.0),
            speed_score=speed_norm.get(name, 0.0),
            resale_score=resale_norm.get(name, 0.0),
            liquidity_score=liquidity_norm.get(name, 0.0),
            risk_score=risk_norm.get(name, 0.0),
        )
        row.total_score = _weighted_total(row, profile)
        scores.append(row)
        breakdown[name] = {
            "acquisition": row.acquisition_score,
            "operating": row.operating_score,
            "range": row.range_score,
            "cabin": row.cabin_score,
            "speed": row.speed_score,
            "resale": row.resale_score,
            "liquidity": row.liquidity_score,
            "risk": row.risk_score,
            "total": row.total_score,
        }

    ranked = sorted(scores, key=lambda s: (-s.total_score, s.aircraft))
    winner = ranked[0].aircraft if ranked else ""
    tradeoffs = build_tradeoff_analysis(ranked)

    opt_id = hashlib.sha256(
        "|".join(
            [
                profile.name,
                ",".join(names),
                winner,
                str(round(ranked[0].total_score, 2) if ranked else 0),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    return OptimizationResult(
        winner=winner,
        ranked_candidates=ranked,
        score_breakdown=breakdown,
        tradeoffs=tradeoffs,
        buyer_profile=profile,
        optimization_id=opt_id,
    )


def build_tradeoff_analysis(ranked: Sequence[AircraftOptimizationScore]) -> List[str]:
    """Deterministic tradeoff lines comparing winner to runners-up."""
    if not ranked:
        return []
    winner = ranked[0]
    lines: List[str] = []

    dim_map = [
        ("range_score", "range"),
        ("speed_score", "speed"),
        ("acquisition_score", "acquisition cost"),
        ("operating_score", "operating cost"),
        ("cabin_score", "cabin"),
        ("resale_score", "resale"),
        ("liquidity_score", "liquidity"),
        ("risk_score", "ownership risk profile"),
    ]

    win_dims: List[str] = []
    for attr, label in dim_map:
        w_val = getattr(winner, attr)
        if all(w_val >= getattr(r, attr) for r in ranked[1:]):
            if w_val > 0:
                win_dims.append(label)
    if win_dims:
        lines.append(f"{winner.aircraft} wins: {', '.join(win_dims)}")

    for other in ranked[1:3]:
        loses: List[str] = []
        buts: List[str] = []
        for attr, label in dim_map:
            w_val = getattr(winner, attr)
            o_val = getattr(other, attr)
            if o_val < w_val - 5:
                loses.append(f"shorter {label}" if "range" in label else f"lower {label}")
            elif o_val > w_val + 5:
                if "cost" in label or label == "acquisition cost":
                    buts.append(f"lower {label}")
                else:
                    buts.append(f"stronger {label}")
        if loses or buts:
            lose_txt = ", ".join(loses[:2]) if loses else "lower composite score"
            but_txt = f" but {', '.join(buts)}" if buts else ""
            lines.append(f"{other.aircraft} loses: {lose_txt}{but_txt}")

    return lines


def _extract_candidates(data_used: Dict[str, Any]) -> List[str]:
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
        for c in alt.get("candidates") or []:
            name = str(c or "").strip()
            if name and name not in aircraft:
                aircraft.append(name)

    comp_ds = data_used.get("authoritative_comparison_dataset")
    if isinstance(comp_ds, dict):
        for row in comp_ds.get("aircraft") or []:
            if isinstance(row, dict):
                name = str(row.get("canonical_name") or "").strip()
                if name and name not in aircraft:
                    aircraft.append(name)

    return aircraft[:8]


def build_optimization_result(
    query: str,
    response: Any,
) -> Dict[str, Any]:
    """Build optimization bundle from a consultant response payload."""
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

    profile = infer_buyer_profile(query, data_used=du)
    candidates = _extract_candidates(du)
    if len(candidates) < 2 and isinstance(du.get("alternative_execution"), dict):
        alt = du["alternative_execution"]
        target = str(alt.get("target") or "").strip()
        if target and target not in candidates:
            candidates.insert(0, target)
    if len(candidates) < 2:
        return {
            "status": "INSUFFICIENT_DATA",
            "confidence": 0,
            "winner": "",
            "ranked_candidates": [],
            "score_breakdown": {},
            "tradeoffs": [],
            "buyer_profile": profile.name,
            "optimization_panel": {},
        }

    result = optimize_aircraft_ranking(candidates, mission=mission, profile=profile)

    panel = {
        "ranking_table": [
            {"rank": i + 1, "aircraft": r.aircraft, "total_score": r.total_score}
            for i, r in enumerate(result.ranked_candidates)
        ],
        "weighted_scores": result.score_breakdown,
        "tradeoffs": result.tradeoffs,
        "buyer_profile": profile.name,
    }

    out = result.to_dict()
    out["optimization_panel"] = panel
    return out


def attach_optimization_result_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.optimization_result`` when env flag enabled."""
    if not decision_optimization_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["optimization_result"] = build_optimization_result(query, out)
    out["data_used"] = du
    return out


def evaluate_optimization_hooks(response: Any) -> List[str]:
    """
    Optional evaluation hooks — ranking stability, profile consistency, reproducibility.

    Returns failure tokens for consultant_evaluator integration.
    """
    if not isinstance(response, dict):
        return []
    du = response.get("data_used")
    if not isinstance(du, dict):
        return []
    bundle = du.get("optimization_result")
    if not isinstance(bundle, dict):
        return []

    failures: List[str] = []
    ranked = bundle.get("ranked_candidates") or []
    if isinstance(ranked, list) and len(ranked) >= 2:
        totals = [float(r.get("total_score") or 0) for r in ranked if isinstance(r, dict)]
        if totals != sorted(totals, reverse=True):
            failures.append("ranking_stability")

    profile = bundle.get("buyer_profile") or {}
    if isinstance(profile, dict):
        weight_keys = (
            "acquisition_cost_weight",
            "operating_cost_weight",
            "range_weight",
            "cabin_weight",
            "speed_weight",
            "resale_weight",
            "market_liquidity_weight",
            "ownership_risk_weight",
        )
        total = sum(float(profile.get(k) or 0) for k in weight_keys)
        if abs(total - 100.0) > 0.5:
            failures.append("profile_consistency")

    opt_id = str(bundle.get("optimization_id") or "")
    if opt_id and not ranked:
        failures.append("optimization_reproducibility")
    if ranked and not opt_id:
        failures.append("optimization_reproducibility")

    if (os.getenv("CONSULTANT_RANKING_CONSISTENCY_ASSERT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        dispatch_models = du.get("authority_dispatch_models") or []
        opt_winner = str(bundle.get("winner") or "").strip()
        if dispatch_models and opt_winner and opt_winner != dispatch_models[0]:
            failures.append("dispatch_optimization_rank_mismatch")

    return list(dict.fromkeys(failures))


__all__ = [
    "AircraftOptimizationScore",
    "BUYER_PROFILES",
    "CABIN_FOCUSED",
    "CHARTER_OPERATOR",
    "CORPORATE_FLIGHT_DEPARTMENT",
    "COST_FOCUSED",
    "DecisionProfile",
    "OptimizationResult",
    "RANGE_FOCUSED",
    "RESALE_FOCUSED",
    "STANDARD_BUYER",
    "attach_optimization_result_if_enabled",
    "build_optimization_result",
    "build_tradeoff_analysis",
    "decision_optimization_enabled",
    "evaluate_optimization_hooks",
    "infer_buyer_profile",
    "optimize_aircraft_ranking",
]

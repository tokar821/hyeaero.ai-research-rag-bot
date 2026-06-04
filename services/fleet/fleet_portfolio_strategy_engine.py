"""
Fleet Portfolio Strategy Engine (FPSE) — Phase 26.

Deterministic fleet-level strategy only. Does not alter routing or single-aircraft recommendations.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_FLEET_ENV = "ENABLE_FLEET_PORTFOLIO_STRATEGY"

_MISSION_TYPES = (
    "transcontinental",
    "intercontinental",
    "regional",
    "short_hop",
    "charter",
)

_MISSION_RANGE_NM = {
    "intercontinental": 5000,
    "transcontinental": 3500,
    "regional": 1500,
    "short_hop": 800,
    "charter": 2000,
}

_COVERAGE_THRESHOLD = 75.0


@dataclass
class FleetInput:
    aircraft_owned: List[str] = field(default_factory=list)
    mission_types: List[str] = field(default_factory=list)
    annual_utilization_hours: int = 300
    operational_constraints: Dict[str, Any] = field(default_factory=dict)
    budget_constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft_owned": list(self.aircraft_owned),
            "mission_types": list(self.mission_types),
            "annual_utilization_hours": int(self.annual_utilization_hours),
            "operational_constraints": dict(self.operational_constraints),
            "budget_constraints": dict(self.budget_constraints),
        }


@dataclass
class FleetPortfolioStrategyReport:
    fleet_profile: str
    current_aircraft: List[str]
    utilization_assumptions: Dict[str, Any]
    mission_coverage_map: Dict[str, float]
    redundancy_analysis: Dict[str, Any]
    gap_analysis: List[str]
    overcapacity_analysis: List[str]
    cost_overlap_matrix: Dict[str, Dict[str, float]]
    optimization_recommendation: List[str]
    replacement_priority_order: List[str]
    phased_upgrade_plan: List[Dict[str, Any]]
    total_fleet_efficiency_score: float
    strategy_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fleet_profile": self.fleet_profile,
            "current_aircraft": list(self.current_aircraft),
            "utilization_assumptions": dict(self.utilization_assumptions),
            "mission_coverage_map": {
                k: round(float(v), 2) for k, v in self.mission_coverage_map.items()
            },
            "redundancy_analysis": dict(self.redundancy_analysis),
            "gap_analysis": list(self.gap_analysis),
            "overcapacity_analysis": list(self.overcapacity_analysis),
            "cost_overlap_matrix": {
                a: {b: round(float(v), 2) for b, v in row.items()}
                for a, row in self.cost_overlap_matrix.items()
            },
            "optimization_recommendation": list(self.optimization_recommendation),
            "replacement_priority_order": list(self.replacement_priority_order),
            "phased_upgrade_plan": [dict(p) for p in self.phased_upgrade_plan],
            "total_fleet_efficiency_score": round(float(self.total_fleet_efficiency_score), 2),
            "strategy_id": self.strategy_id,
        }


def fleet_portfolio_strategy_enabled() -> bool:
    return (os.getenv(_FLEET_ENV) or "").strip().lower() in ("1", "true", "yes")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _aircraft_capabilities(aircraft: str) -> Dict[str, Any]:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    if rec is None:
        return {
            "canonical_name": aircraft,
            "range_nm": 2500.0,
            "category": "large-cabin",
            "pax_max": 8,
            "short_field_score": 0.5,
            "manufacturer": "",
        }

    prof = rec.to_profile_dict()
    return {
        "canonical_name": rec.canonical_name,
        "range_nm": float(rec.nbaa_range_nm),
        "category": rec.aircraft_category,
        "pax_max": rec.passenger_capacity_max,
        "short_field_score": float(prof.get("short_field_score") or 0.55),
        "manufacturer": rec.manufacturer,
        "current_in_production": rec.current_in_production,
    }


def _mission_fit_score(cap: Dict[str, Any], mission: str) -> float:
    mission_key = mission.replace("-", "_").lower()
    if mission_key not in _MISSION_RANGE_NM:
        return 0.0

    required = _MISSION_RANGE_NM[mission_key]
    range_nm = float(cap.get("range_nm") or 0)
    pax = int(cap.get("pax_max") or 0)
    short_field = float(cap.get("short_field_score") or 0.5)

    if range_nm < required:
        return _clamp((range_nm / required) * 60.0)

    margin = range_nm - required
    score = 75.0 + min(25.0, (margin / max(required, 1)) * 40.0)

    if mission_key == "charter" and pax >= 8:
        score = min(100.0, score + 10.0)
    if mission_key == "short_hop" and short_field >= 0.65:
        score = min(100.0, score + 8.0)

    return _clamp(score)


def build_mission_coverage_map(
    aircraft_owned: Sequence[str],
    mission_types: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Map fleet coverage percentage per mission type."""
    fleet = [str(a).strip() for a in aircraft_owned if str(a or "").strip()]
    missions = list(mission_types or _MISSION_TYPES)
    if not fleet:
        return {m: 0.0 for m in missions}

    caps = [_aircraft_capabilities(ac) for ac in fleet]
    coverage: Dict[str, float] = {}
    for mission in missions:
        scores = [_mission_fit_score(c, mission) for c in caps]
        coverage[mission] = max(scores) if scores else 0.0
    return coverage


def analyze_fleet_redundancy(aircraft_owned: Sequence[str]) -> Dict[str, Any]:
    """Detect overlapping range, cabin class, and mission capability."""
    fleet = [str(a).strip() for a in aircraft_owned if str(a or "").strip()]
    if len(fleet) < 2:
        return {
            "redundancy_score": 0.0,
            "overlapping_range_pairs": [],
            "overlapping_cabin_pairs": [],
            "duplicated_mission_pairs": [],
        }

    caps = [_aircraft_capabilities(ac) for ac in fleet]
    range_pairs: List[str] = []
    cabin_pairs: List[str] = []
    mission_pairs: List[str] = []

    for i in range(len(caps)):
        for j in range(i + 1, len(caps)):
            a, b = caps[i], caps[j]
            an, bn = a["canonical_name"], b["canonical_name"]
            ra, rb = float(a["range_nm"]), float(b["range_nm"])
            if ra > 0 and rb > 0 and abs(ra - rb) / max(ra, rb) <= 0.15:
                range_pairs.append(f"{an} / {bn}")
            if a["category"] == b["category"]:
                cabin_pairs.append(f"{an} / {bn}")
            overlap = 0
            for mission in _MISSION_TYPES:
                sa = _mission_fit_score(a, mission)
                sb = _mission_fit_score(b, mission)
                if sa >= 80 and sb >= 80:
                    overlap += 1
            if overlap >= 3:
                mission_pairs.append(f"{an} / {bn}")

    pair_count = len(range_pairs) + len(cabin_pairs) + len(mission_pairs)
    max_pairs = max(1, len(fleet) * (len(fleet) - 1))
    redundancy_score = _clamp((pair_count / max_pairs) * 100.0)

    return {
        "redundancy_score": round(redundancy_score, 2),
        "overlapping_range_pairs": range_pairs,
        "overlapping_cabin_pairs": cabin_pairs,
        "duplicated_mission_pairs": mission_pairs,
    }


def identify_fleet_gaps(
    aircraft_owned: Sequence[str],
    mission_types: Optional[Sequence[str]] = None,
) -> List[str]:
    """Identify missing long-range, cabin, short-field, and backup capabilities."""
    coverage = build_mission_coverage_map(aircraft_owned, mission_types)
    gaps: List[str] = []

    if coverage.get("intercontinental", 0) < _COVERAGE_THRESHOLD:
        gaps.append("missing long-range intercontinental capability")
    if coverage.get("transcontinental", 0) < _COVERAGE_THRESHOLD:
        gaps.append("missing transcontinental nonstop capability")
    if coverage.get("regional", 0) < _COVERAGE_THRESHOLD:
        gaps.append("missing efficient regional mission coverage")
    if coverage.get("short_hop", 0) < _COVERAGE_THRESHOLD:
        gaps.append("missing short-field / short-hop access")
    if coverage.get("charter", 0) < _COVERAGE_THRESHOLD:
        gaps.append("missing high-capacity charter-ready cabin")

    caps = [_aircraft_capabilities(ac) for ac in aircraft_owned]
    if len(aircraft_owned) >= 2:
        has_backup = any(c.get("current_in_production") for c in caps)
        if not has_backup:
            gaps.append("missing production-current backup redundancy")

    if not gaps:
        gaps.append("no critical mission gaps detected at current threshold")
    return gaps


def compute_fleet_cost_overlap(aircraft_owned: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Pairwise fixed-cost overlap matrix from ownership economics."""
    from services.ownership.aircraft_lifecycle_ownership_engine import build_ownership_report

    fleet = [str(a).strip() for a in aircraft_owned if str(a or "").strip()]
    matrix: Dict[str, Dict[str, float]] = {ac: {} for ac in fleet}

    reports = {ac: build_ownership_report(ac) for ac in fleet}
    caps = {ac: _aircraft_capabilities(ac) for ac in fleet}

    for i, a in enumerate(fleet):
        for j, b in enumerate(fleet):
            if a == b:
                matrix[a][b] = 100.0
                continue
            ra, rb = reports[a], reports[b]
            ca, cb = caps[a], caps[b]
            overlap = 0.0
            if ca.get("category") == cb.get("category"):
                overlap += 35.0
            if ca.get("manufacturer") == cb.get("manufacturer") and ca.get("manufacturer"):
                overlap += 20.0
            fixed_ratio = min(ra.annual_fixed_cost, rb.annual_fixed_cost) / max(
                ra.annual_fixed_cost, rb.annual_fixed_cost, 1.0
            )
            overlap += fixed_ratio * 45.0
            matrix[a][b] = round(_clamp(overlap), 2)
    return matrix


def _overcapacity_analysis(
    aircraft_owned: Sequence[str],
    redundancy: Dict[str, Any],
) -> List[str]:
    notes: List[str] = []
    score = float(redundancy.get("redundancy_score") or 0)
    if score >= 60:
        notes.append("fleet carries elevated redundant capability in overlapping range bands")
    if redundancy.get("overlapping_cabin_pairs"):
        notes.append(
            f"duplicate cabin-class pairs detected: {', '.join(redundancy['overlapping_cabin_pairs'][:3])}"
        )
    if len(aircraft_owned) >= 4 and score >= 45:
        notes.append("utilization may be diluted across too many similar mission profiles")
    if not notes:
        notes.append("no material overcapacity detected at current fleet size")
    return notes


def optimize_fleet_structure(
    fleet_input: FleetInput,
    *,
    coverage_threshold: float = _COVERAGE_THRESHOLD,
) -> List[str]:
    """Recommend add/remove actions under mission coverage and cost constraints."""
    recommendations: List[str] = []
    coverage = build_mission_coverage_map(fleet_input.aircraft_owned, fleet_input.mission_types)
    gaps = identify_fleet_gaps(fleet_input.aircraft_owned, fleet_input.mission_types)
    redundancy = analyze_fleet_redundancy(fleet_input.aircraft_owned)

    for mission, pct in coverage.items():
        if pct < coverage_threshold:
            if mission == "intercontinental":
                recommendations.append("add ultra-long-range asset to close intercontinental gap")
            elif mission == "transcontinental":
                recommendations.append("add large-cabin long-range asset for transcontinental coverage")
            elif mission == "short_hop":
                recommendations.append("add short-field capable light or turboprop asset")
            elif mission == "charter":
                recommendations.append("add higher-capacity charter-suitable cabin")
            else:
                recommendations.append(f"add mission-fit asset to improve {mission} coverage")

    if float(redundancy.get("redundancy_score") or 0) >= 55 and len(fleet_input.aircraft_owned) >= 2:
        pair = (redundancy.get("duplicated_mission_pairs") or [""])[0]
        if pair:
            recommendations.append(f"consider retiring one redundant pair: {pair}")

    budget = fleet_input.budget_constraints or {}
    max_ac = budget.get("max_aircraft")
    if isinstance(max_ac, int) and len(fleet_input.aircraft_owned) > max_ac:
        recommendations.append(f"reduce fleet count to budget cap of {max_ac} aircraft")

    if not recommendations:
        recommendations.append("maintain current fleet mix; coverage and redundancy within targets")
    return recommendations


def rank_aircraft_for_replacement(
    aircraft_owned: Sequence[str],
    *,
    mission_coverage_map: Optional[Dict[str, float]] = None,
    ownership_reports: Optional[Dict[str, Dict[str, Any]]] = None,
    market_intelligence: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Rank aircraft for replacement using ownership, market, and mission signals.
    """
    from services.ownership.aircraft_lifecycle_ownership_engine import build_ownership_report

    fleet = [str(a).strip() for a in aircraft_owned if str(a or "").strip()]
    if not fleet:
        return []

    coverage = mission_coverage_map or build_mission_coverage_map(fleet)
    own_map = ownership_reports or {}
    mi = market_intelligence if isinstance(market_intelligence, dict) else {}

    scores: List[Tuple[str, float]] = []
    for ac in fleet:
        cap = _aircraft_capabilities(ac)
        canonical = cap["canonical_name"]
        own = own_map.get(ac) or own_map.get(canonical)
        if not own:
            report = build_ownership_report(canonical, market_intelligence=mi)
            own = report.to_dict()

        dep_ratio = 0.0
        acq = float(own.get("acquisition_price") or 1)
        dep = float(own.get("depreciation_amount") or 0)
        if acq > 0:
            dep_ratio = dep / acq

        maint_proxy = float(own.get("annual_operating_cost") or 0) / max(acq, 1.0)
        risk = 100.0 - float(own.get("ownership_risk_score") or 50.0)
        liquidity_penalty = 0.0
        if mi.get("liquidity_score") is not None:
            liquidity_penalty = max(0.0, 50.0 - float(mi["liquidity_score"]))

        mission_mismatch = 0.0
        fits = sum(1 for m in _MISSION_TYPES if _mission_fit_score(cap, m) >= 75)
        mission_mismatch = max(0.0, (3 - fits) * 12.0)

        icrl_penalty = 0.0
        score = (
            dep_ratio * 35.0
            + maint_proxy * 25.0
            + risk * 0.25
            + liquidity_penalty * 0.2
            + mission_mismatch
            + icrl_penalty
        )
        scores.append((canonical, score))

    ranked = sorted(scores, key=lambda x: (-x[1], x[0]))
    return [name for name, _ in ranked]


def build_5_year_upgrade_path(
    aircraft_owned: Sequence[str],
    replacement_order: Sequence[str],
    optimization_recommendations: Sequence[str],
) -> List[Dict[str, Any]]:
    """Build year 1–5 retire/acquire/transition plan."""
    fleet = list(aircraft_owned)
    retire_queue = list(replacement_order)
    acquire_hints = [r for r in optimization_recommendations if r.startswith("add ")]

    plan: List[Dict[str, Any]] = []
    for year in range(1, 6):
        retire: List[str] = []
        acquire: List[str] = []
        if retire_queue and year <= 3:
            retire.append(retire_queue.pop(0))
        if acquire_hints and year in (2, 4):
            acquire.append(acquire_hints[min(len(acquire_hints) - 1, (year // 2) - 1)])

        transition: List[str] = []
        for r in retire:
            for a in acquire:
                transition.append(f"transition {r} missions to {a}")

        plan.append(
            {
                "year": year,
                "retire_aircraft": retire,
                "acquire_aircraft": acquire,
                "transition_mapping": transition,
            }
        )
    return plan


def _fleet_efficiency_score(
    coverage: Dict[str, float],
    redundancy_score: float,
    utilization_hours: int,
    fleet_size: int,
) -> float:
    avg_coverage = sum(coverage.values()) / max(len(coverage), 1)
    cost_efficiency = _clamp(100.0 - redundancy_score * 0.6)
    redundancy_penalty = _clamp(redundancy_score)
    util_target = 250 * max(1, fleet_size)
    util_alignment = _clamp((utilization_hours / util_target) * 100.0) if util_target else 50.0

    return _clamp(
        avg_coverage * 0.35
        + cost_efficiency * 0.25
        + (100.0 - redundancy_penalty) * 0.25
        + util_alignment * 0.15
    )


def build_fleet_portfolio_strategy_report(
    fleet_input: FleetInput,
    *,
    fleet_profile: str = "corporate_flight_department",
    ownership_bundle: Optional[Dict[str, Any]] = None,
    market_bundle: Optional[Dict[str, Any]] = None,
) -> FleetPortfolioStrategyReport:
    """Build full fleet portfolio strategy report."""
    fleet = list(fleet_input.aircraft_owned)
    missions = fleet_input.mission_types or list(_MISSION_TYPES)

    coverage = build_mission_coverage_map(fleet, missions)
    redundancy = analyze_fleet_redundancy(fleet)
    gaps = identify_fleet_gaps(fleet, missions)
    overlap = compute_fleet_cost_overlap(fleet)
    overcap = _overcapacity_analysis(fleet, redundancy)
    recommendations = optimize_fleet_structure(fleet_input)

    own_reports: Dict[str, Dict[str, Any]] = {}
    if isinstance(ownership_bundle, dict):
        for row in ownership_bundle.get("ownership_reports") or []:
            if isinstance(row, dict) and row.get("aircraft"):
                own_reports[str(row["aircraft"])] = row

    replacement_order = rank_aircraft_for_replacement(
        fleet,
        mission_coverage_map=coverage,
        ownership_reports=own_reports,
        market_intelligence=market_bundle,
    )
    upgrade_plan = build_5_year_upgrade_path(fleet, replacement_order, recommendations)
    efficiency = _fleet_efficiency_score(
        coverage,
        float(redundancy.get("redundancy_score") or 0),
        fleet_input.annual_utilization_hours,
        len(fleet),
    )

    strategy_id = hashlib.sha256(
        "|".join(
            [
                fleet_profile,
                ",".join(sorted(fleet)),
                str(round(efficiency, 2)),
                ",".join(replacement_order),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    return FleetPortfolioStrategyReport(
        fleet_profile=fleet_profile,
        current_aircraft=fleet,
        utilization_assumptions={
            "annual_utilization_hours": fleet_input.annual_utilization_hours,
            "operational_constraints": dict(fleet_input.operational_constraints),
            "budget_constraints": dict(fleet_input.budget_constraints),
        },
        mission_coverage_map=coverage,
        redundancy_analysis=redundancy,
        gap_analysis=gaps,
        overcapacity_analysis=overcap,
        cost_overlap_matrix=overlap,
        optimization_recommendation=recommendations,
        replacement_priority_order=replacement_order,
        phased_upgrade_plan=upgrade_plan,
        total_fleet_efficiency_score=efficiency,
        strategy_id=strategy_id,
    )


def _infer_fleet_from_query(query: str) -> List[str]:
    from services.consultant.recommendation_engine import detect_models_from_text

    models = detect_models_from_text(query or "")
    return models[:6]


def _extract_fleet_input(query: str, data_used: Dict[str, Any]) -> FleetInput:
    explicit = data_used.get("fleet_input") or data_used.get("fleet_profile")
    if isinstance(explicit, dict):
        return FleetInput(
            aircraft_owned=list(explicit.get("aircraft_owned") or explicit.get("current_aircraft") or []),
            mission_types=list(explicit.get("mission_types") or _MISSION_TYPES),
            annual_utilization_hours=int(explicit.get("annual_utilization_hours") or 300),
            operational_constraints=dict(explicit.get("operational_constraints") or {}),
            budget_constraints=dict(explicit.get("budget_constraints") or {}),
        )

    aircraft: List[str] = []
    for row in data_used.get("consultant_recommendations") or []:
        if isinstance(row, dict):
            name = str(row.get("model") or row.get("aircraft") or "").strip()
            if name:
                aircraft.append(name)

    own = data_used.get("ownership_intelligence")
    if isinstance(own, dict):
        for row in own.get("ownership_reports") or []:
            if isinstance(row, dict) and row.get("aircraft"):
                name = str(row["aircraft"])
                if name not in aircraft:
                    aircraft.append(name)

    if not aircraft:
        aircraft = _infer_fleet_from_query(query)

    if len(aircraft) < 2:
        return FleetInput(
            aircraft_owned=[],
            mission_types=list(_MISSION_TYPES),
            annual_utilization_hours=300,
            budget_constraints={"max_aircraft": 6},
        )

    profile_name = "corporate_flight_department"
    if re.search(r"\bcharter\b", (query or "").lower()):
        profile_name = "charter_operator"

    return FleetInput(
        aircraft_owned=aircraft,
        mission_types=list(_MISSION_TYPES),
        annual_utilization_hours=300,
        budget_constraints={"max_aircraft": 6},
    )


def build_fleet_portfolio_strategy(
    query: str,
    response: Any,
) -> Dict[str, Any]:
    """Build fleet strategy bundle from consultant response payload."""
    payload = response if isinstance(response, dict) else {}
    du = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}

    fleet_input = _extract_fleet_input(query, du)
    if len(fleet_input.aircraft_owned) < 2:
        return {
            "status": "INSUFFICIENT_DATA",
            "confidence": 0,
            "trends": {},
            "current_aircraft": [],
            "mission_coverage_map": {},
            "redundancy_analysis": {},
            "cost_overlap_matrix": {},
            "replacement_priority_order": [],
            "phased_upgrade_plan": [],
            "optimization_recommendation": [],
            "fleet_panel": {},
        }

    profile = "corporate_flight_department"
    if isinstance(du.get("fleet_input"), dict):
        profile = str(du["fleet_input"].get("profile") or profile)

    report = build_fleet_portfolio_strategy_report(
        fleet_input,
        fleet_profile=profile,
        ownership_bundle=du.get("ownership_intelligence"),
        market_bundle=du.get("market_intelligence"),
    )

    panel = {
        "fleet_coverage_map": report.mission_coverage_map,
        "redundancy_score": report.redundancy_analysis.get("redundancy_score"),
        "replacement_priority_list": report.replacement_priority_order,
        "upgrade_timeline": report.phased_upgrade_plan,
    }

    out = report.to_dict()
    out["fleet_panel"] = panel
    out["fleet_input"] = fleet_input.to_dict()
    return out


def attach_fleet_portfolio_strategy_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.fleet_portfolio_strategy`` when env flag enabled."""
    if not fleet_portfolio_strategy_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["fleet_portfolio_strategy"] = build_fleet_portfolio_strategy(query, out)
    out["data_used"] = du
    return out


def evaluate_fleet_portfolio_strategy_hooks(response: Any) -> List[str]:
    """Optional evaluation hooks for fleet strategy consistency."""
    if not isinstance(response, dict):
        return []
    du = response.get("data_used")
    if not isinstance(du, dict):
        return []
    bundle = du.get("fleet_portfolio_strategy")
    if not isinstance(bundle, dict):
        return []

    failures: List[str] = []
    coverage = bundle.get("mission_coverage_map") or {}
    redundancy = bundle.get("redundancy_analysis") or {}
    overlap = bundle.get("cost_overlap_matrix") or {}
    upgrade = bundle.get("phased_upgrade_plan") or []
    replacement = bundle.get("replacement_priority_order") or []
    fleet = bundle.get("current_aircraft") or []

    if isinstance(coverage, dict):
        for mission, pct in coverage.items():
            if pct > 100 or pct < 0:
                failures.append("mission_coverage_consistency")

    red_score = float(redundancy.get("redundancy_score") or 0)
    if red_score > 100 or red_score < 0:
        failures.append("redundancy_validity")
    if len(fleet) >= 2 and red_score == 0 and redundancy.get("duplicated_mission_pairs"):
        failures.append("redundancy_validity")

    if isinstance(overlap, dict) and fleet:
        for ac in fleet:
            row = overlap.get(ac) or {}
            if isinstance(row, dict):
                for val in row.values():
                    if float(val) > 100 or float(val) < 0:
                        failures.append("cost_overlap_accuracy")

    if isinstance(upgrade, list):
        years = [p.get("year") for p in upgrade if isinstance(p, dict)]
        if years and years != list(range(1, len(years) + 1)):
            failures.append("upgrade_feasibility")
        for step in upgrade:
            if not isinstance(step, dict):
                continue
            retire = step.get("retire_aircraft") or []
            if retire and retire[0] not in replacement:
                failures.append("upgrade_feasibility")

    return list(dict.fromkeys(failures))


__all__ = [
    "FleetInput",
    "FleetPortfolioStrategyReport",
    "analyze_fleet_redundancy",
    "attach_fleet_portfolio_strategy_if_enabled",
    "build_5_year_upgrade_path",
    "build_fleet_portfolio_strategy",
    "build_fleet_portfolio_strategy_report",
    "build_mission_coverage_map",
    "compute_fleet_cost_overlap",
    "evaluate_fleet_portfolio_strategy_hooks",
    "fleet_portfolio_strategy_enabled",
    "identify_fleet_gaps",
    "optimize_fleet_structure",
    "rank_aircraft_for_replacement",
]

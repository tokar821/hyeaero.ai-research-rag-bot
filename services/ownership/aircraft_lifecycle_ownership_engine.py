"""
Aircraft Lifecycle Ownership Intelligence Engine (ALOI) — Phase 25.

Deterministic ownership economics only. Does not alter routing or recommendations.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_OWNERSHIP_ENV = "ENABLE_OWNERSHIP_INTELLIGENCE"

_DEFAULT_ANNUAL_HOURS = 300
_CATEGORY_ACQUISITION_DEFAULT = {
    "light": 6_000_000.0,
    "super-midsize": 18_000_000.0,
    "large-cabin": 28_000_000.0,
    "ultra-long": 45_000_000.0,
}
_BASE_DEPRECIATION_RATE = {
    "light": 0.085,
    "super-midsize": 0.075,
    "large-cabin": 0.068,
    "ultra-long": 0.062,
    "turboprop": 0.09,
}


@dataclass
class OwnershipIntelligenceReport:
    aircraft: str
    acquisition_price: float
    annual_operating_cost: float
    annual_fixed_cost: float
    annual_variable_cost: float
    projected_resale_value: float
    depreciation_amount: float
    total_cost_5_year: float
    total_cost_10_year: float
    ownership_risk_score: float
    lifecycle_score: float
    confidence: float
    projected_resale_5_year: float = 0.0
    projected_resale_10_year: float = 0.0
    capital_exposure: float = 0.0
    report_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft": self.aircraft,
            "acquisition_price": round(float(self.acquisition_price), 2),
            "annual_operating_cost": round(float(self.annual_operating_cost), 2),
            "annual_fixed_cost": round(float(self.annual_fixed_cost), 2),
            "annual_variable_cost": round(float(self.annual_variable_cost), 2),
            "projected_resale_value": round(float(self.projected_resale_value), 2),
            "projected_resale_5_year": round(float(self.projected_resale_5_year), 2),
            "projected_resale_10_year": round(float(self.projected_resale_10_year), 2),
            "depreciation_amount": round(float(self.depreciation_amount), 2),
            "total_cost_5_year": round(float(self.total_cost_5_year), 2),
            "total_cost_10_year": round(float(self.total_cost_10_year), 2),
            "capital_exposure": round(float(self.capital_exposure), 2),
            "ownership_risk_score": round(float(self.ownership_risk_score), 2),
            "lifecycle_score": round(float(self.lifecycle_score), 2),
            "confidence": round(float(self.confidence), 3),
            "report_id": self.report_id,
        }


def ownership_intelligence_enabled() -> bool:
    return (os.getenv(_OWNERSHIP_ENV) or "").strip().lower() in ("1", "true", "yes")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _aircraft_economics(aircraft: str) -> Dict[str, Any]:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    if rec is None:
        return {
            "canonical_name": aircraft,
            "category": "",
            "operating_index": 0.0,
            "current_in_production": False,
            "production_end_year": None,
            "manufacturer": "",
            "replacement_models": [],
            "_insufficient": True,
        }

    prof = rec.to_profile_dict()
    acquisition = _CATEGORY_ACQUISITION_DEFAULT.get(rec.aircraft_category, 20_000_000.0)
    try:
        from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

        rows = find_catalog_matches([rec.canonical_name])
        if rows:
            price = typical_market_price_usd(rows[0])
            if price and price > 0:
                acquisition = float(price)
    except Exception:
        pass

    oi = float(prof.get("operating_index") or 0.65)
    cat = str(rec.aircraft_category or "large-cabin").lower()
    variable_hour = 2800.0 + oi * 5200.0
    if "ultra" in cat or "large" in cat:
        variable_hour += 1800.0
    elif "light" in cat or "turboprop" in cat:
        variable_hour -= 600.0

    fixed_annual = 450_000.0 + oi * 1_100_000.0
    if "ultra" in cat:
        fixed_annual += 350_000.0

    return {
        "canonical_name": rec.canonical_name,
        "category": rec.aircraft_category,
        "operating_index": oi,
        "acquisition_price": acquisition,
        "variable_hour": variable_hour,
        "fixed_annual": fixed_annual,
        "resale_score": float(prof.get("resale_score") or 0.75),
        "current_in_production": rec.current_in_production,
        "production_end_year": rec.production_end_year,
        "manufacturer": rec.manufacturer,
        "replacement_models": list(rec.replacement_models or []),
    }


def _annual_costs(
    economics: Dict[str, Any],
    *,
    annual_hours: int = _DEFAULT_ANNUAL_HOURS,
) -> Tuple[float, float, float]:
    hours = max(50, int(annual_hours or _DEFAULT_ANNUAL_HOURS))
    fixed = float(economics.get("fixed_annual") or 0)
    variable = float(economics.get("variable_hour") or 0) * hours
    return fixed + variable, fixed, variable


def estimate_depreciation_curve(
    *,
    aircraft_age_years: int = 0,
    production_status: bool = True,
    liquidity_score: float = 50.0,
    market_intelligence: Optional[Dict[str, Any]] = None,
    category: str = "large-cabin",
) -> Dict[str, Any]:
    """
    Estimate 5- and 10-year depreciation curves.

    Returns annual rates and cumulative retention factors.
    """
    base_rate = _BASE_DEPRECIATION_RATE.get(category, 0.07)
    rate = base_rate

    if not production_status:
        rate += 0.012
    if aircraft_age_years > 12:
        rate += 0.008
    elif aircraft_age_years > 8:
        rate += 0.004

    mi = market_intelligence if isinstance(market_intelligence, dict) else {}
    price_trend = str(mi.get("price_trend") or "")
    replacement_risk = str(mi.get("replacement_risk") or "")
    if price_trend == "appreciating":
        rate -= 0.005
    elif price_trend == "depreciating":
        rate += 0.005
    if replacement_risk == "HIGH":
        rate += 0.01
    elif replacement_risk == "LOW":
        rate -= 0.003

    if liquidity_score >= 70:
        rate -= 0.004
    elif liquidity_score <= 35:
        rate += 0.006

    rate = max(0.04, min(0.12, rate))

    retention_5 = (1.0 - rate) ** 5
    retention_10 = (1.0 - rate) ** 10
    return {
        "annual_rate": rate,
        "retention_5_year": retention_5,
        "retention_10_year": retention_10,
        "projected_values_5_year": retention_5,
        "projected_values_10_year": retention_10,
    }


def estimate_future_resale_value(
    *,
    current_market_value: float,
    depreciation_curve: Dict[str, Any],
    years: int = 5,
) -> float:
    """Project future resale value from current value and depreciation curve."""
    value = max(0.0, float(current_market_value or 0))
    if value <= 0:
        return 0.0
    rate = float(depreciation_curve.get("annual_rate") or 0.07)
    retention = (1.0 - rate) ** max(0, int(years))
    return value * retention


def evaluate_ownership_risk(
    aircraft: str,
    *,
    market_intelligence: Optional[Dict[str, Any]] = None,
    liquidity_score: float = 50.0,
) -> float:
    """
    Score ownership risk 0–100 where 100 = lowest risk.
    """
    econ = _aircraft_economics(aircraft)
    score = 55.0

    if econ.get("current_in_production"):
        score += 15.0
    else:
        score -= 12.0

    if econ.get("production_end_year"):
        from datetime import datetime

        years_since = datetime.now().year - int(econ["production_end_year"])
        if years_since >= 5:
            score -= 8.0

    if econ.get("replacement_models"):
        score -= 6.0

    if econ.get("manufacturer") in ("Gulfstream", "Bombardier", "Textron", "Dassault", "Embraer"):
        score += 8.0

    score += float(econ.get("resale_score") or 0.75) * 20.0
    score += (liquidity_score - 50.0) * 0.2

    mi = market_intelligence if isinstance(market_intelligence, dict) else {}
    repl = str(mi.get("replacement_risk") or "")
    if repl == "HIGH":
        score -= 12.0
    elif repl == "LOW":
        score += 8.0

    return _clamp(score)


def _lifecycle_score(
    *,
    total_cost_10_year: float,
    depreciation_amount: float,
    acquisition_price: float,
    projected_resale_10_year: float,
    ownership_risk_score: float,
    peer_costs: Sequence[float],
) -> float:
    if peer_costs:
        max_cost = max(peer_costs)
        min_cost = min(peer_costs)
        if max_cost > min_cost:
            cost_score = 100.0 - ((total_cost_10_year - min_cost) / (max_cost - min_cost)) * 100.0
        else:
            cost_score = 50.0
    else:
        cost_score = 50.0

    dep_ratio = depreciation_amount / acquisition_price if acquisition_price > 0 else 0.5
    dep_score = _clamp(100.0 - dep_ratio * 120.0)

    resale_ratio = projected_resale_10_year / acquisition_price if acquisition_price > 0 else 0.5
    resale_score = _clamp(resale_ratio * 120.0)

    return _clamp(
        cost_score * 0.30 + dep_score * 0.25 + resale_score * 0.25 + ownership_risk_score * 0.20
    )


def build_ownership_report(
    aircraft: str,
    *,
    annual_hours: int = _DEFAULT_ANNUAL_HOURS,
    aircraft_age_years: int = 5,
    market_intelligence: Optional[Dict[str, Any]] = None,
    liquidity_score: float = 50.0,
    peer_costs: Optional[Sequence[float]] = None,
) -> OwnershipIntelligenceReport:
    """Build a full ownership intelligence report for one aircraft."""
    econ = _aircraft_economics(aircraft)
    canonical = str(econ.get("canonical_name") or aircraft)
    acquisition = float(econ.get("acquisition_price") or 0)
    annual_total, annual_fixed, annual_variable = _annual_costs(econ, annual_hours=annual_hours)

    mi = market_intelligence if isinstance(market_intelligence, dict) else {}
    if mi.get("liquidity_score") is not None:
        try:
            liquidity_score = float(mi["liquidity_score"])
        except (TypeError, ValueError):
            pass

    curve = estimate_depreciation_curve(
        aircraft_age_years=aircraft_age_years,
        production_status=bool(econ.get("current_in_production")),
        liquidity_score=liquidity_score,
        market_intelligence=mi,
        category=str(econ.get("category") or "large-cabin"),
    )

    resale_5 = estimate_future_resale_value(
        current_market_value=acquisition,
        depreciation_curve=curve,
        years=5,
    )
    resale_10 = estimate_future_resale_value(
        current_market_value=acquisition,
        depreciation_curve=curve,
        years=10,
    )
    depreciation_amount = acquisition - resale_10
    total_5 = (annual_total * 5) + (acquisition - resale_5)
    total_10 = (annual_total * 10) + (acquisition - resale_10)
    risk = evaluate_ownership_risk(
        canonical,
        market_intelligence=mi,
        liquidity_score=liquidity_score,
    )
    lifecycle = _lifecycle_score(
        total_cost_10_year=total_10,
        depreciation_amount=depreciation_amount,
        acquisition_price=acquisition,
        projected_resale_10_year=resale_10,
        ownership_risk_score=risk,
        peer_costs=peer_costs or [],
    )

    confidence = 0.6
    if acquisition > 0 and acquisition != _CATEGORY_ACQUISITION_DEFAULT.get(str(econ.get("category")), 0):
        confidence += 0.15
    if mi:
        confidence += 0.1
    confidence = min(0.92, confidence)

    report_id = hashlib.sha256(
        "|".join(
            [
                canonical,
                str(round(acquisition, 0)),
                str(round(total_5, 0)),
                str(round(total_10, 0)),
                str(round(lifecycle, 2)),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    return OwnershipIntelligenceReport(
        aircraft=canonical,
        acquisition_price=acquisition,
        annual_operating_cost=annual_total,
        annual_fixed_cost=annual_fixed,
        annual_variable_cost=annual_variable,
        projected_resale_value=resale_10,
        projected_resale_5_year=resale_5,
        projected_resale_10_year=resale_10,
        depreciation_amount=depreciation_amount,
        total_cost_5_year=total_5,
        total_cost_10_year=total_10,
        capital_exposure=acquisition,
        ownership_risk_score=risk,
        lifecycle_score=lifecycle,
        confidence=confidence,
        report_id=report_id,
    )


def compare_ownership_profiles(
    candidates: Sequence[str],
    *,
    annual_hours: int = _DEFAULT_ANNUAL_HOURS,
    market_by_aircraft: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compare lifecycle economics across multiple aircraft."""
    names = [str(c).strip() for c in candidates if str(c or "").strip()]
    if not names:
        return {"cheapest_to_own": "", "strongest_resale": "", "lowest_risk": "", "reports": []}

    market_map = market_by_aircraft or {}
    preliminary = [
        build_ownership_report(
            n,
            annual_hours=annual_hours,
            market_intelligence=market_map.get(n),
        )
        for n in names
    ]
    peer_costs = [r.total_cost_10_year for r in preliminary]

    reports = [
        build_ownership_report(
            n,
            annual_hours=annual_hours,
            market_intelligence=market_map.get(n),
            peer_costs=peer_costs,
        )
        for n in names
    ]

    cheapest = min(reports, key=lambda r: r.total_cost_10_year)
    strongest = max(reports, key=lambda r: r.projected_resale_10_year)
    lowest_risk = max(reports, key=lambda r: r.ownership_risk_score)

    return {
        "cheapest_to_own": cheapest.aircraft,
        "strongest_resale": strongest.aircraft,
        "lowest_risk": lowest_risk.aircraft,
        "reports": [r.to_dict() for r in reports],
    }


def _extract_aircraft_list(data_used: Dict[str, Any]) -> List[str]:
    aircraft: List[str] = []
    rec_rows = data_used.get("consultant_recommendations")
    if isinstance(rec_rows, list):
        for row in rec_rows:
            if isinstance(row, dict):
                name = str(row.get("model") or row.get("aircraft") or "").strip()
                if name:
                    aircraft.append(name)

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
            aircraft.insert(0, name)

    opt = data_used.get("optimization_result")
    if isinstance(opt, dict):
        for row in opt.get("ranked_candidates") or []:
            if isinstance(row, dict):
                name = str(row.get("aircraft") or "").strip()
                if name and name not in aircraft:
                    aircraft.append(name)

    return aircraft[:6]


def build_ownership_intelligence(
    query: str,
    response: Any,
) -> Dict[str, Any]:
    """Build ownership intelligence bundle from a consultant response payload."""
    payload = response if isinstance(response, dict) else {}
    du = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}

    candidates = _extract_aircraft_list(du)
    if not candidates:
        return {
            "status": "INSUFFICIENT_DATA",
            "confidence": 0,
            "primary_aircraft": "",
            "ownership_reports": [],
            "comparison": {},
            "ownership_summary": {},
            "ownership_panel": {},
        }

    market_intel = du.get("market_intelligence")
    market_by_aircraft: Dict[str, Dict[str, Any]] = {}
    if isinstance(market_intel, dict) and market_intel.get("aircraft"):
        pass
    elif isinstance(market_intel, dict):
        ac = str(market_intel.get("aircraft") or candidates[0])
        market_by_aircraft[ac] = market_intel

    comparison = compare_ownership_profiles(candidates, market_by_aircraft=market_by_aircraft)
    primary_name = candidates[0]
    primary: Dict[str, Any] = comparison["reports"][0] if comparison["reports"] else {}
    for report in comparison["reports"]:
        if report.get("aircraft") == primary_name:
            primary = report
            break

    ownership_summary = {
        "five_year_ownership_cost": primary.get("total_cost_5_year"),
        "ten_year_ownership_cost": primary.get("total_cost_10_year"),
        "capital_exposure": primary.get("capital_exposure"),
        "expected_residual_value": primary.get("projected_resale_10_year"),
        "annual_operating_cost": primary.get("annual_operating_cost"),
    }

    panel = {
        "ownership_costs": {
            "annual_fixed": primary.get("annual_fixed_cost"),
            "annual_variable": primary.get("annual_variable_cost"),
            "annual_total": primary.get("annual_operating_cost"),
        },
        "depreciation": primary.get("depreciation_amount"),
        "resale_projection": {
            "five_year": primary.get("projected_resale_5_year"),
            "ten_year": primary.get("projected_resale_10_year"),
        },
        "lifecycle_score": primary.get("lifecycle_score"),
    }

    return {
        "primary_aircraft": primary.get("aircraft") or primary_name,
        "ownership_reports": comparison["reports"],
        "comparison": {
            "cheapest_to_own": comparison["cheapest_to_own"],
            "strongest_resale": comparison["strongest_resale"],
            "lowest_risk": comparison["lowest_risk"],
        },
        "ownership_summary": ownership_summary,
        "ownership_panel": panel,
    }


def attach_ownership_intelligence_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.ownership_intelligence`` when env flag enabled."""
    if not ownership_intelligence_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["ownership_intelligence"] = build_ownership_intelligence(query, out)
    out["data_used"] = du
    return out


def evaluate_ownership_intelligence_hooks(response: Any) -> List[str]:
    """
    Optional evaluation hooks — depreciation consistency, lifecycle consistency, cost completeness.

    Returns failure tokens for consultant_evaluator integration.
    """
    if not isinstance(response, dict):
        return []
    du = response.get("data_used")
    if not isinstance(du, dict):
        return []
    bundle = du.get("ownership_intelligence")
    if not isinstance(bundle, dict):
        return []

    failures: List[str] = []
    reports = bundle.get("ownership_reports") or []
    if not isinstance(reports, list):
        return failures

    for row in reports:
        if not isinstance(row, dict):
            continue
        acq = float(row.get("acquisition_price") or 0)
        resale_5 = float(row.get("projected_resale_5_year") or 0)
        resale_10 = float(row.get("projected_resale_10_year") or 0)
        dep = float(row.get("depreciation_amount") or 0)
        total_5 = float(row.get("total_cost_5_year") or 0)
        total_10 = float(row.get("total_cost_10_year") or 0)
        lifecycle = float(row.get("lifecycle_score") or 0)
        confidence = float(row.get("confidence") or 0)
        fixed = float(row.get("annual_fixed_cost") or 0)
        variable = float(row.get("annual_variable_cost") or 0)
        operating = float(row.get("annual_operating_cost") or 0)

        if acq > 0 and resale_10 > acq:
            failures.append("depreciation_consistency")
        if resale_5 > 0 and resale_10 > resale_5 and acq > 0:
            failures.append("depreciation_consistency")
        if acq > 0 and abs(dep - (acq - resale_10)) > acq * 0.02:
            failures.append("depreciation_consistency")

        if total_10 > 0 and total_5 > 0 and total_10 < total_5:
            failures.append("lifecycle_score_consistency")

        if lifecycle >= 90 and acq <= 0:
            failures.append("lifecycle_score_consistency")

        if confidence >= 0.8 and (fixed <= 0 or variable <= 0 or operating <= 0):
            failures.append("ownership_cost_completeness")

    return list(dict.fromkeys(failures))


__all__ = [
    "OwnershipIntelligenceReport",
    "attach_ownership_intelligence_if_enabled",
    "build_ownership_intelligence",
    "build_ownership_report",
    "compare_ownership_profiles",
    "estimate_depreciation_curve",
    "estimate_future_resale_value",
    "evaluate_ownership_intelligence_hooks",
    "evaluate_ownership_risk",
    "ownership_intelligence_enabled",
]

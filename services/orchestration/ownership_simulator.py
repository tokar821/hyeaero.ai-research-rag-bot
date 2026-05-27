"""
Ownership economics simulator — utilization bands, structure, and burdened hourly economics.

Reasoning-first bands derived from operating_index and mission posture — not canned quotes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import detect_models_from_text

_HOURS_RE = re.compile(
    r"(?:around\s+)?(\d{2,3})\s+hours?\s+(?:a|per|annually|/year)"
    r"|charter\s+around\s+(\d{2,3})\s+hours?",
    re.I,
)


@dataclass
class OwnershipSimulationResult:
    annual_hours: int
    structure_recommendation: str
    variable_cost_per_hour_usd: float
    fixed_annual_burden_usd: float
    all_in_hour_usd: float
    utilization_band: str
    dispatch_posture: str
    liquidity_note: str
    lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annual_hours": self.annual_hours,
            "structure_recommendation": self.structure_recommendation,
            "variable_cost_per_hour_usd": round(self.variable_cost_per_hour_usd, 0),
            "fixed_annual_burden_usd": round(self.fixed_annual_burden_usd, 0),
            "all_in_hour_usd": round(self.all_in_hour_usd, 0),
            "utilization_band": self.utilization_band,
            "dispatch_posture": self.dispatch_posture,
            "liquidity_note": self.liquidity_note,
            "lines": list(self.lines),
        }


def _parse_annual_hours(query: str) -> int:
    m = _HOURS_RE.search(query or "")
    if m:
        raw = m.group(1) or m.group(2)
        if raw:
            return max(25, min(1200, int(raw)))
    return 0


def _model_economics(model: str) -> Dict[str, float]:
    from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

    prof = AIRCRAFT_PROFILES.get(model) or AIRCRAFT_PROFILES.get(model.title()) or {}
    idx = float(prof.get("operating_index") or 0.55)
    cat = str(prof.get("category") or "").lower()
    base_var = 2800.0 + idx * 5200.0
    if "ultra" in cat or "large" in cat:
        base_var += 1800.0
    elif "light" in cat or "turboprop" in cat:
        base_var -= 600.0
    fixed = 450_000.0 + idx * 1_100_000.0
    if "ultra" in cat:
        fixed += 350_000.0
    return {"variable_hour": base_var, "fixed_annual": fixed, "operating_index": idx}


def _utilization_band(hours: int) -> str:
    if hours <= 0:
        return "unspecified (planning 150–400 hr/year)"
    if hours < 150:
        return "low (<150 hr/year)"
    if hours < 250:
        return "fractional/charter-adjacent (150–250 hr/year)"
    if hours < 400:
        return "hybrid (250–400 hr/year)"
    return "ownership-weighted (400+ hr/year)"


def _structure_for_hours(hours: int, ql: str) -> str:
    if "fractional" in ql and "full" in ql:
        return "fractional_vs_full_tradeoff"
    if "fractional" in ql or "netjets" in ql or "flexjet" in ql:
        return "fractional_program"
    if "charter" in ql or "jet card" in ql or "card" in ql:
        return "charter_or_card"
    if hours and hours < 175:
        return "fractional_or_charter"
    if hours and hours >= 400:
        return "full_ownership"
    if hours:
        return "fractional_or_light_ownership"
    return "structure_tbd"


def simulate_ownership_economics(
    query: str,
    *,
    mission: Optional[MissionState] = None,
    anchor_model: str = "",
) -> OwnershipSimulationResult:
    ql = (query or "").lower()
    hours = _parse_annual_hours(query)
    model = (anchor_model or "").strip()
    if not model:
        try:
            found = detect_models_from_text(query)
            model = found[0] if found else ""
        except Exception:
            model = ""

    econ = _model_economics(model) if model else {"variable_hour": 4200.0, "fixed_annual": 850_000.0}
    var_hr = float(econ["variable_hour"])
    fixed = float(econ["fixed_annual"])

    planning_hours = hours or 250
    all_in = var_hr + (fixed / max(planning_hours, 1))

    structure = _structure_for_hours(hours, ql)
    band = _utilization_band(hours)

    dispatch = "schedule-critical — favor ownership or higher fractional share"
    if hours and hours < 150:
        dispatch = "flexibility-first — charter/card/fractional avoids fixed crew idle"
    elif hours and hours >= 400:
        dispatch = "dispatch control — full ownership absorbs downtime if utilization holds"

    liquidity = (
        "Exit liquidity tracks program status, engine horizon, and narrow market depth — "
        "not model marketing."
    )
    if model and ("g650" in model.lower() or "global" in model.lower()):
        liquidity += " ULR inventory is deep but serial-specific."

    lines: List[str] = [
        "Ownership Economics (simulated):",
        "",
        f"* Utilization: {band}",
        f"* Structure: {structure.replace('_', ' ')}",
    ]
    if model:
        lines.append(f"* Aircraft anchor: {model}")
    lines.append(
        f"* Planning economics @ {planning_hours} hr/year: "
        f"~${int(var_hr):,}/hr variable + ~${int(fixed):,}/yr fixed "
        f"→ ~${int(all_in):,}/hr all-in (directional, not a quote)."
    )
    lines.append("")
    lines.append("Analysis:")

    lines.append(
        "Dispatch tradeoff: charter/card preserves peak-day flexibility without fixed crew idle; "
        "ownership or higher fractional share buys schedule control when ad-hoc lift fails operationally."
    )
    lines.append(
        "Capital efficiency: compare burdened all-in hourly (fixed + variable amortized) to fractional "
        "card pricing at your peak month — crossover is often utilization-driven, not average-hours driven."
    )
    if hours and 200 <= hours <= 350:
        lines.append(
            "Fractional crossover band (~200–350 hr/year): evaluate guaranteed availability days, "
            "deadhead reposition assumptions, and management overhead before full ownership."
        )
    lines.append(
        "Peak-day conflicts: charter pools thin on holidays and major events; ownership/fractional "
        "addresses availability risk that average annual hours understate."
    )
    lines.append(
        "Deadhead reality: hub-and-spoke or multi-city programs add empty reposition legs — "
        "burdened economics must include reposition, not just occupied block hours."
    )

    if hours and hours < 150:
        lines.append(
            "Utilization inflection (~150 hr/year): below this band, fixed crew, insurance, "
            "and management overhead dominate — charter/card preserves capital efficiency."
        )
        lines.append(
            "Guaranteed dispatch: ad-hoc charter does not guarantee peak-day lift; "
            "fractional share or card programs buy availability you are not paying for in hourly rate alone."
        )
        lines.append(
            "Deadhead: at low annual hours, reposition to your departure airport is a large "
            "hidden cost — burdened economics must include empty legs, not occupied block time only."
        )
        lines.append(
            "Management burden: even light ownership implies director-level oversight, "
            "maintenance tracking, and regulatory compliance — often underestimated below 200 hr/year."
        )
    if hours and 140 <= hours <= 180:
        lines.append(
            "Crossover zone (~150 hr/year): compare burdened all-in hourly to fractional/card "
            "with guaranteed peak days — ownership wins only when dispatch control is worth the fixed burden."
        )
    elif hours and hours >= 400:
        lines.append(
            "Above ~400 hr/year, fixed crew and maintenance reserves amortize — verify "
            "engine events and downtime assumptions before anchoring on acquisition price."
        )
    else:
        lines.append(
            "In the 200–400 hr band, compare burdened DOC to fractional all-in hourly "
            "before selecting airframe — structure precedes model."
        )

    if mission and (mission.operating_cost_priority or "").lower() in ("high", "medium"):
        lines.append("Operating cost is a stated priority — weight variable hour and fixed burden heavily.")

    lines.append(f"* Dispatch: {dispatch}")
    lines.append(f"* Liquidity: {liquidity}")
    lines.append("")
    lines.append(
        "Verdict: Size utilization and structure first; aircraft model follows once the "
        "ownership band is credible (verify tax posture with counsel)."
    )

    return OwnershipSimulationResult(
        annual_hours=hours,
        structure_recommendation=structure,
        variable_cost_per_hour_usd=var_hr,
        fixed_annual_burden_usd=fixed,
        all_in_hour_usd=all_in,
        utilization_band=band,
        dispatch_posture=dispatch,
        liquidity_note=liquidity,
        lines=lines,
    )

"""
Hard aircraft elimination — applied before ranking and narrative generation.

Non-negotiable exclusions; the LLM explanation layer must not restore eliminated aircraft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.feasibility_engine import (
    FeasibilityResult,
    _is_winter_westbound_transpacific,
    peak_required_route_nm,
)
from services.mission.models import MissionProfile

RULE_ULR_WESTBOUND_PACIFIC = "ulr_westbound_pacific_nonstop"
RULE_ULR_LONG_STAGE = "ulr_long_stage_nonstop"

# User-specified auto-reject (also covered by super-midsize category rule)
_EXPLICIT_HARD_REJECT: frozenset[str] = frozenset(
    {
        "Challenger 350",
        "Gulfstream G280",
        "Praetor 600",
        "Citation Longitude",
        "Challenger Longitude",
    }
)

# Canonical allowlist (catalog keys)
_ULR_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Global 7500",
        "Gulfstream G650",
        "Gulfstream G650ER",
        "Falcon 8X",
        "Global 6500",
    }
)

# Detected names → catalog key
_MODEL_ALIASES: Dict[str, str] = {
    "Gulfstream G650ER": "Gulfstream G650ER",
    "G650ER": "Gulfstream G650ER",
    "G650": "Gulfstream G650",
    "Global 6500": "Global 6500",
    "Citation Longitude": "Challenger Longitude",
}

_MIN_PRACTICAL_NM_EQUIVALENT_ULR = 5500.0
_MIN_REQUIRED_ROUTE_NM_FOR_GATE = 5500.0


@dataclass(frozen=True)
class HardEliminationContext:
    """Active hard gate for this mission profile."""

    rule_id: str
    summary: str
    required_route_nm: float


def _route_blob(profile: MissionProfile) -> str:
    return " ".join(r.label() for r in (profile.routes or [])).lower()


def _is_ny_tokyo(profile: MissionProfile) -> bool:
    blob = _route_blob(profile)
    return bool(
        re.search(r"\b(?:new\s+york|nyc|teterboro|jfk)\b", blob)
        and re.search(r"\btokyo\b", blob)
    )


def _is_westbound_pacific(profile: MissionProfile) -> bool:
    if profile.westbound_sensitive:
        return True
    blob = _route_blob(profile)
    return bool(re.search(r"\bwestbound\b", blob)) or _is_winter_westbound_transpacific(profile)


def detect_hard_elimination_context(profile: MissionProfile) -> Optional[HardEliminationContext]:
    """
    Return active hard gate when mission requires ULR westbound transpacific nonstop.

    Triggers include NY/Tokyo nonstop, westbound winter Pacific, and ~5500+ nm required
    stage length (practical operational requirement, not brochure).
    """
    routes = profile.routes or []
    if not routes:
        return None

    required_nm = peak_required_route_nm(profile)
    blob = _route_blob(profile)
    transpacific = bool(
        re.search(r"\btokyo|seoul|hong\s+kong|beijing|singapore\b", blob)
        and re.search(
            r"\b(?:san\s+francisco|los\s+angeles|seattle|new\s+york|nyc|west\s+coast)\b",
            blob,
        )
    )
    ny_tokyo = _is_ny_tokyo(profile)
    westbound_pacific = _is_westbound_pacific(profile)
    nonstop = bool(profile.nonstop_required)

    high_range_mission = required_nm >= _MIN_REQUIRED_ROUTE_NM_FOR_GATE
    if not high_range_mission and required_nm >= 4800 and nonstop and westbound_pacific:
        high_range_mission = True

    if not nonstop:
        return None

    if ny_tokyo and westbound_pacific and (high_range_mission or nonstop):
        summary = (
            "NY/Tokyo nonstop westbound Pacific — ultra-long-range only "
            f"(~{int(required_nm)} nm required with reserves/margins)."
        )
        return HardEliminationContext(
            rule_id=RULE_ULR_WESTBOUND_PACIFIC,
            summary=summary,
            required_route_nm=required_nm,
        )

    if transpacific and westbound_pacific and high_range_mission:
        summary = (
            "Westbound transpacific nonstop — ultra-long-range only "
            f"(~{int(required_nm)} nm required with reserves/margins)."
        )
        return HardEliminationContext(
            rule_id=RULE_ULR_WESTBOUND_PACIFIC,
            summary=summary,
            required_route_nm=required_nm,
        )

    if required_nm >= 6800 and nonstop:
        summary = (
            "Ultra-long stage nonstop (Dubai-class) — ultra-long-range only "
            f"(~{int(required_nm)} nm required with NBAA reserves)."
        )
        return HardEliminationContext(
            rule_id=RULE_ULR_LONG_STAGE,
            summary=summary,
            required_route_nm=required_nm,
        )

    if (
        required_nm >= _MIN_REQUIRED_ROUTE_NM_FOR_GATE
        and nonstop
        and re.search(r"\b(?:dubai|london|paris|geneva)\b", blob)
    ):
        summary = (
            "International long-stage nonstop — ultra-long-range band "
            f"(~{int(required_nm)} nm peak leg with reserves)."
        )
        return HardEliminationContext(
            rule_id=RULE_ULR_LONG_STAGE,
            summary=summary,
            required_route_nm=required_nm,
        )

    return None


def _catalog_key(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def _aircraft_spec(model: str) -> Optional[Dict]:
    key = _catalog_key(model)
    spec = AIRCRAFT_PROFILES.get(key)
    if spec:
        return spec
    return AIRCRAFT_PROFILES.get(model)


def hard_elimination_reason(
    model: str,
    ctx: HardEliminationContext,
) -> Optional[str]:
    """
  Return elimination reason if ``model`` is hard-excluded under ``ctx``; else None.
    """
    key = _catalog_key(model)
    spec = _aircraft_spec(model)
    if not spec:
        return f"hard_exclusion[{ctx.rule_id}]: unknown aircraft ({model})"

    category = str(spec.get("category") or "")
    practical_nm = float(spec.get("practical_nm") or 0)

    if key in _ULR_ALLOWLIST or model in _ULR_ALLOWLIST:
        return None

    if model in _EXPLICIT_HARD_REJECT or key in _EXPLICIT_HARD_REJECT:
        return (
            f"hard_exclusion[{ctx.rule_id}]: {model} auto-rejected — "
            f"mission requires ultra-long-range; {ctx.summary}"
        )

    if category == "super-midsize":
        return (
            f"hard_exclusion[{ctx.rule_id}]: super-midsize ({model}) auto-rejected — "
            f"{ctx.summary}"
        )

    if category != "ultra-long":
        return (
            f"hard_exclusion[{ctx.rule_id}]: {category} platform ({model}) excluded — "
            f"only ultra-long-range aircraft permitted; {ctx.summary}"
        )

    if practical_nm < _MIN_PRACTICAL_NM_EQUIVALENT_ULR:
        return (
            f"hard_exclusion[{ctx.rule_id}]: {model} practical range "
            f"~{int(practical_nm)} nm below {_MIN_PRACTICAL_NM_EQUIVALENT_ULR:.0f} nm ULR floor; "
            f"{ctx.summary}"
        )

    # Ultra-long but not on allowlist — permit equivalent ULR by practical_nm
    if practical_nm >= _MIN_PRACTICAL_NM_EQUIVALENT_ULR:
        return None

    return (
        f"hard_exclusion[{ctx.rule_id}]: {model} not on approved ULR list for this mission; "
        f"{ctx.summary}"
    )


def hard_gate_allowlist(profile: MissionProfile) -> Optional[List[str]]:
    """Models permitted to enter scoring when a hard gate is active."""
    ctx = detect_hard_elimination_context(profile)
    if not ctx:
        return None
    return [
        m
        for m in AIRCRAFT_PROFILES
        if hard_elimination_reason(m, ctx) is None
    ]


def apply_hard_mission_elimination(
    mission_profile: MissionProfile,
    feasible_models: List[str],
    feasibility_map: Optional[Dict[str, FeasibilityResult]] = None,
) -> Tuple[List[str], List[str], List[Dict[str, object]], Optional[HardEliminationContext]]:
    """
    Filter ``feasible_models`` through hard elimination rules.

    Returns ``(survivors, eliminated_models, log_entries, context)``.
    """
    ctx = detect_hard_elimination_context(mission_profile)
    if not ctx:
        return feasible_models, [], [], None

    survivors: List[str] = []
    eliminated: List[str] = []
    log: List[Dict[str, object]] = []
    fmap = feasibility_map if feasibility_map is not None else {}

    for model in feasible_models:
        reason = hard_elimination_reason(model, ctx)
        if reason:
            eliminated.append(model)
            log.append(
                {
                    "aircraft_name": model,
                    "reason": reason,
                    "mission_constraint_failed": "hard_mission_elimination",
                    "hard_rule_id": ctx.rule_id,
                }
            )
            fmap[model] = FeasibilityResult(
                feasible=False,
                elimination_reasons=[reason],
                operational_risk_level="eliminated",
                required_route_nm=ctx.required_route_nm,
            )
        else:
            survivors.append(model)

    return survivors, eliminated, log, ctx


def hard_excluded_model_set(
    mission_profile: MissionProfile,
    candidates: Optional[List[str]] = None,
) -> Set[str]:
    """All models that would be hard-excluded (for narrator firewall)."""
    ctx = detect_hard_elimination_context(mission_profile)
    if not ctx:
        return set()
    models = candidates or list(AIRCRAFT_PROFILES.keys())
    return {m for m in models if hard_elimination_reason(m, ctx)}

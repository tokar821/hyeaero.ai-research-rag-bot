"""
Operational synthesis — broker conclusions derived from structured mission evidence.

Produces reasoning from corridor, payload, wind, and dispatch context — not canned aircraft lists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile
from services.mission.mission_understanding_engine import MissionUnderstandingPacket


def _pax_label(mission: MissionState, profile: MissionProfile) -> int:
    return int(mission.passenger_count or profile.passengers or 0)


def enrich_operational_synthesis(
    packet: MissionUnderstandingPacket,
    mission: MissionState,
    profile: MissionProfile,
    *,
    query: str = "",
    operational_context: Optional[Any] = None,
) -> str:
    """
    Build broker-style operational conclusion from structured evidence.
    """
    parts: List[str] = list(
        p.strip()
        for p in (packet.operational_synthesis or "").split(". ")
        if p.strip()
    )
    seen = {p.lower() for p in parts}

    def _add(text: str) -> None:
        t = text.strip()
        if t and t.lower() not in seen:
            parts.append(t)
            seen.add(t.lower())

    pax = _pax_label(mission, profile)
    corridor = packet.corridor_type or ""
    westbound = bool(
        packet.inferred_constraints.get("westbound_winter_pressure")
        or mission.westbound
        or profile.westbound_sensitive
    )
    ulr = corridor in ("multi_leg_ultra_long", "transatlantic_ulr") or any(
        "ultra-long" in b.lower() for b in packet.fallback_operational_band
    )
    transatlantic_mid = corridor in (
        "transatlantic_super_mid",
        "transatlantic_heavy",
    ) or packet.inferred_constraints.get("transatlantic_super_mid_floor")

    if packet.inferred_constraints.get("cabin_utilization_modest") and transatlantic_mid:
        _add(
            "Passenger utilization is modest for the stated corridor — super-mid or heavy-cabin "
            "covers the mission; ULR-class capital is usually overspec unless nonstop on the longest "
            "stage is non-negotiable. Supplemental charter on peak legs is often more rational than "
            "owning a Global-class platform."
        )
    elif transatlantic_mid and not ulr:
        _add(
            "Transatlantic / Europe executive band — planning centers on super-mid or heavy-cabin "
            "with realistic winter reserves; light-jet economics are not operationally credible "
            "at full passenger load."
        )

    if ulr:
        _add(
            "Operational band: ultra-long-range — this mission is not solvable with super-mid or large-cabin "
            "aircraft on the longest stages without tech stops."
        )
        if pax >= 6:
            _add(
                f"Planning assumption: {pax} passengers with executive baggage — payload-range tradeoffs "
                "materially tighten the shortlist versus brochure range."
            )
        if westbound:
            _add(
                "Westbound winter margin is decisive — headwind and alternate fuel burn eliminate "
                "marginal ULR airframes from reliable dispatch; ER-class reserves separate conditional "
                "from strong performers."
            )
            _add(
                "Base-range G650-class may be conditional on winter westbound legs; G650ER and Global 7500-class "
                "typically carry stronger dispatch headroom at full passenger load."
            )

    if packet.inferred_constraints.get("incompatible_mission_bands"):
        _add(
            "Fleet composition: incompatible operational bands — ULR oceanic legs and short-field domestic "
            "access cannot share one platform without structural compromise."
        )
    elif packet.inferred_constraints.get("dual_use_or_multi_leg") and ulr:
        _add(
            "Multi-leg ULR portfolio from a single hub — one capable ULR platform can cover these corridors; "
            "ranking should prioritize worst-case stage (longest leg + winter westbound if stated)."
        )

    util = packet.utilization_style or ""
    if packet.inferred_constraints.get("balanced_cost_dispatch"):
        _add(
            "Cost is a priority, but dispatch reliability, pressurization, baggage, and winter "
            "margin still set the floor — runway-flexible super-mid planning beats light-jet collapse."
        )

    if packet.inferred_constraints.get("supplemental_charter_viable"):
        _add(
            "Low utilization on long stages — fractional or supplemental charter on peak months "
            "often beats full ULR ownership before airframe selection."
        )

    if util in ("executive_shuttle", "board_transport", "mixed_corporate") and pax >= 6:
        _add(
            "Executive travel profile — dispatch reliability and cabin credibility outweigh pure STOL economics; "
            "turboprop or light piston options are not operationally serious for this passenger load."
        )

    if packet.inferred_constraints.get("industrial_airport_access"):
        _add(
            "Field and industrial airport access dominate cabin spec — runway length, climb gradient, "
            "and unpaved or short-strip capability set the planning floor before interior comfort."
        )

    if packet.inferred_constraints.get("island_ops") and pax >= 6:
        _add(
            "Caribbean / tropical ops with executive load — runway access matters, but pressurized jet "
            "dispatch and cabin standards remain the floor; PC-12-class is a compromise, not the planning band."
        )
    regions = packet.explicit_constraints.get("regions_served") or []
    if regions:
        _add(
            f"Regional scope includes {', '.join(regions)} — size for hub-spoke island ops and occasional "
            "longer South America legs, not STOL utility economics."
        )

    if operational_context is not None:
        peak = float(getattr(operational_context, "peak_stage_nm", 0) or 0)
        if peak >= 4800:
            _add(f"Peak verified stage ~{int(peak)} nm — ULR band confirmed by route authority.")
        payload = getattr(operational_context, "payload", None)
        if payload is not None:
            bags = getattr(payload, "baggage_priority", "") or ""
            if bags:
                _add(f"Payload posture: {bags.replace('_', ' ')}.")

    if packet.inferred_constraints.get("ownership_economics_relevant"):
        hrs = packet.inferred_constraints.get("annual_charter_hours")
        if hrs:
            _add(
                f"Ownership crossover relevant at ~{hrs} hr/year — structure and dispatch control precede airframe."
            )

    return ". ".join(parts) + ("." if parts else "")


def build_operational_conclusion_block(
    mission: MissionState,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
    *,
    query: str = "",
    feasibility_notes: Optional[Sequence[str]] = None,
    ranked_models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Structured operational conclusion for advisory and telemetry."""
    ctx = None
    try:
        from services.operational.mission_operational_assessment import (
            build_mission_operational_context,
        )

        if mission.routes or profile.routes:
            ctx = build_mission_operational_context(mission, profile, query=query)
    except Exception:
        ctx = None

    synthesis = enrich_operational_synthesis(
        packet, mission, profile, query=query, operational_context=ctx
    )

    conclusion: Dict[str, Any] = {
        "operational_band": packet.corridor_type,
        "class_bands": list(packet.fallback_operational_band[:4]),
        "synthesis": synthesis,
        "westbound_winter": bool(packet.inferred_constraints.get("westbound_winter_pressure")),
        "incompatible_bands": bool(packet.inferred_constraints.get("incompatible_mission_bands")),
        "executive_profile": packet.utilization_style in (
            "executive_shuttle",
            "board_transport",
            "mixed_corporate",
        ),
        "planning_assumptions": [],
    }

    pax = _pax_label(mission, profile)
    if pax:
        conclusion["planning_assumptions"].append(f"{pax} passengers executive load")

    if ctx is not None:
        conclusion["peak_stage_nm"] = round(float(ctx.peak_stage_nm or 0), 1)
        conclusion["planning_mode"] = ctx.planning_mode

    if ranked_models:
        conclusion["ranked_models"] = list(ranked_models[:5])

    if feasibility_notes:
        conclusion["elimination_notes"] = list(feasibility_notes[:4])

    return conclusion

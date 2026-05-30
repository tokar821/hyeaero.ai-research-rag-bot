"""
Named aircraft capability evaluation — feasibility-only, no generic shortlist.

Output: FEASIBLE | MARGINAL | NOT REALISTIC with operational reasoning.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile


_VERDICT_FEASIBLE = "FEASIBLE"
_VERDICT_MARGINAL = "MARGINAL"
_VERDICT_NOT_REALISTIC = "NOT REALISTIC"

# Output contract: strip recommendation / class-suggestion language before return
_RECOMMENDATION_TOKEN_RE = re.compile(
    r"\b(?:alternatives?|better\s+option|if\s+not,?\s+consider|consider\s+(?:a|an|the)|"
    r"recommend(?:ed|ation)?|class(?:es)?|category|categories|ranked|shortlist|substitute|"
    r"\|\s*rank\b)\b",
    re.I,
)

_RANKED_LIST_RE = re.compile(
    r"(?:##\s*ranked|rank\s*\|\s*aircraft|\|\s*1\s*\|)",
    re.I,
)


def _sanitize_capability_line(line: str) -> Optional[str]:
    """Drop lines that violate the capability-only output contract."""
    stripped = line.strip()
    if not stripped:
        return line
    if _RECOMMENDATION_TOKEN_RE.search(stripped):
        return None
    if stripped.startswith("### ") and any(
        kw in stripped.lower()
        for kw in ("alternative", "suggestion", "consider", "instead", "recommend")
    ):
        return None
    return line


def _sanitize_capability_output(text: str) -> str:
    if _RANKED_LIST_RE.search(text):
        return (
            "## Named Aircraft Capability\n\n"
            "NOT REALISTIC — capability mode cannot emit ranked aircraft lists.\n"
        )
    kept: List[str] = []
    skip_section = False
    for line in text.splitlines():
        if line.strip().startswith("### ") and _RECOMMENDATION_TOKEN_RE.search(line):
            skip_section = True
            continue
        if skip_section:
            if line.strip().startswith("### "):
                skip_section = False
            else:
                continue
        cleaned = _sanitize_capability_line(line)
        if cleaned is not None:
            kept.append(cleaned)
    return "\n".join(kept).strip()


def _route_distance_nm(mission: MissionState) -> float:
    if not mission.routes:
        return 0.0
    try:
        from services.mission.route_distance_authority import resolve_route_distance

        res = resolve_route_distance(mission.routes[0])
        if res.distance_nm > 0:
            return float(res.distance_nm)
        from services.mission.feasibility_engine import estimate_route_distance_nm

        return float(estimate_route_distance_nm(mission.routes[0]))
    except Exception:
        return 0.0


def evaluate_named_aircraft_capability(
    model: str,
    mission: MissionState,
    *,
    mission_profile: Optional[MissionProfile] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Physics-first feasibility for a single named aircraft."""
    from services.catalog.catalog_alias_resolver import (
        resolve_canonical_display_name,
        resolve_catalog_profile_key,
    )

    display_name = resolve_canonical_display_name(model)
    profile_key = resolve_catalog_profile_key(model) or display_name
    spec: Dict[str, Any] = {}
    try:
        from services.data_authority.aircraft_spec_repository import (
            INSUFFICIENT_VERIFIED_AIRCRAFT_DATA,
            get_verified_spec,
        )

        verified = get_verified_spec(model)
        if verified is None:
            return {
                "model": display_name or model,
                "verdict": _VERDICT_NOT_REALISTIC,
                "reasons": [
                    f"{INSUFFICIENT_VERIFIED_AIRCRAFT_DATA} — no verified PostgreSQL specification for this airframe.",
                ],
            }
        spec = verified.to_profile_dict()
        profile_key = verified.canonical_name
    except Exception:
        spec = AIRCRAFT_PROFILES.get(profile_key) or {}
        if not spec:
            return {
                "model": display_name or model,
                "verdict": _VERDICT_NOT_REALISTIC,
                "reasons": [
                    "corridor infeasibility: insufficient verified performance data for this airframe in-band.",
                ],
            }

    required_nm = _route_distance_nm(mission)
    practical_nm = float(spec.get("practical_nm") or spec.get("range_nm") or 0)
    pax = int(mission.passenger_count or 0) or int(spec.get("typical_pax") or 8)
    max_pax = int(spec.get("max_pax") or pax)

    reasons: List[str] = []
    verdict = _VERDICT_FEASIBLE

    category = str(spec.get("category") or "").lower()
    if required_nm > 0 and practical_nm > 0:
        margin = practical_nm - required_nm
        if margin < -400:
            verdict = _VERDICT_NOT_REALISTIC
            reasons.append(
                "reserve-margin conflict: stage length exceeds practical range with NBAA reserve realism."
            )
        elif margin < 200:
            verdict = _VERDICT_MARGINAL
            reasons.append(
                "dispatch reliability: marginal fuel/payload margin — winter or westbound pressure may block dispatch."
            )
        if required_nm >= 4500 and category in ("super-midsize", "midsize", "light"):
            verdict = _VERDICT_NOT_REALISTIC
            reasons.insert(
                0,
                "corridor class mismatch: transpacific ULR stage — super-midsize cannot hold NBAA reserves with executive payload.",
            )

    if pax > max_pax:
        verdict = _VERDICT_NOT_REALISTIC
        reasons.append("payload assumption conflict: passenger load exceeds certified cabin capacity.")

    if mission_profile is not None:
        try:
            from services.recommendation.hack_v1_constraint_kernel import (
                run_hack_v1_constraint_kernel,
            )

            hack = run_hack_v1_constraint_kernel(
                mission_profile,
                [profile_key],
                query="",
            )
            if profile_key not in (hack.feasible_aircraft_list or []):
                rej = next((r for r in hack.rejection_log if r.model == profile_key), None)
                verdict = _VERDICT_NOT_REALISTIC
                reasons.append(
                    rej.reason
                    if rej
                    else "runway incompatibility: hard physics gate rejected this airframe for the mission."
                )
        except Exception:
            pass

    if not reasons and verdict == _VERDICT_FEASIBLE:
        reasons.append(
            "Stage length, passenger load, and reserve assumptions align with this airframe for the corridor."
        )

    result = {
        "model": display_name or model,
        "verdict": verdict,
        "reasons": reasons,
        "category": category,
    }
    try:
        from services.operations.operational_realism_bridge import (
            assess_mission_operational_realism,
            merge_operational_realism_into_capability,
        )

        realism = assess_mission_operational_realism(
            mission,
            profile_key,
            spec,
            query=str((data_used or {}).get("orchestration_query") or ""),
            mission_profile=mission_profile,
            data_used=data_used,
        )
        result = merge_operational_realism_into_capability(result, realism)
    except Exception:
        pass
    return result


def format_named_aircraft_capability_response(
    models: Sequence[str],
    mission: MissionState,
    *,
    mission_profile: Optional[MissionProfile] = None,
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> str:
    """Structured capability output — no ranked shortlist, no alternate aircraft."""
    lines: List[str] = ["## Named Aircraft Capability", ""]
    if not models:
        lines.append("NOT REALISTIC — no aircraft model identified for evaluation.")
        return "\n".join(lines)

    ql = (query or "").lower()

    for model in models[:2]:
        ev = evaluate_named_aircraft_capability(
            model,
            mission,
            mission_profile=mission_profile,
            data_used=data_used,
        )
        lines.append("### Capability Verdict")
        lines.append(f"- **Aircraft**: {model}")
        lines.append(f"- **Verdict**: {ev['verdict']}")
        # Constraint breakdown (deterministic, best-effort)
        from services.catalog.catalog_alias_resolver import resolve_catalog_profile_key

        profile_key = resolve_catalog_profile_key(model) or model
        spec = AIRCRAFT_PROFILES.get(profile_key) or {}
        practical_nm = float(spec.get("practical_nm") or 0.0)
        required_nm = _route_distance_nm(mission)
        if practical_nm and required_nm:
            lines.append(f"- **Range margin (nm)**: {int(practical_nm - required_nm)}")
        lines.append("- **Reserves**: NBAA IFR (assumed)")
        if "westbound" in ql:
            lines.append(
                "- **Wind sensitivity**: westbound winter penalty applies (dispatch margin tighter)."
            )
        lines.append("")
        lines.append("### Constraint breakdown")
        for r in ev.get("reasons") or []:
            lines.append(f"- {r}")
        lines.append("")

    lines.append(
        "*Feasibility evaluation for the named airframe(s) only — verdict and constraint breakdown.*"
    )
    return _sanitize_capability_output("\n".join(lines).strip())


__all__ = [
    "evaluate_named_aircraft_capability",
    "format_named_aircraft_capability_response",
]

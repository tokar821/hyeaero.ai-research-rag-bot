"""
Deterministic named-aircraft capability responder — route feasibility only.

Compact broker-tone output (1–3 sentences). No ranked shortlists or recommendations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.aircraft_truth.constants import UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE
from services.catalog.catalog_alias_resolver import resolve_catalog_profile_key
from services.consultant.mission_state import MissionState, build_mission_from_current_turn

_FORBIDDEN_PHRASES = re.compile(
    r"\b(?:good\s+fit|recommend|shortlist|best\s+jet|compare|versus|alternatives?)\b",
    re.I,
)

_VERDICT_FEASIBLE = "FEASIBLE"
_VERDICT_MARGINAL = "MARGINAL"

_CITY_ROUTE_RE = re.compile(
    r"\b(?:fly|from)\s+(.+?)\s+to\s+(.+?)(?:\s+nonstop|\?|$)",
    re.I,
)


def _guard_answer(text: str) -> str:
    if _FORBIDDEN_PHRASES.search(text or ""):
        return UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return " ".join(sentences[:3])


def _extract_route_label(query: str, mission: MissionState) -> Optional[str]:
    routes = [r.strip() for r in (mission.routes or []) if str(r).strip()]
    if routes:
        return routes[0]
    m = _CITY_ROUTE_RE.search(query or "")
    if m:
        return f"{m.group(1).strip()} -> {m.group(2).strip()}"
    try:
        from services.mission.route_extractor import extract_routes, routes_from_extractions

        extracted = routes_from_extractions(extract_routes(query or ""))
        if extracted:
            return extracted[0]
    except Exception:
        pass
    return None


def _mission_with_route(query: str, mission: Optional[MissionState]) -> Optional[MissionState]:
    ms = mission if mission is not None else build_mission_from_current_turn(query or "")
    route_label = _extract_route_label(query or "", ms)
    if not route_label:
        return None
    if ms.routes:
        return ms
    return MissionState(routes=[route_label])


def _format_capability_answer(
    model: str,
    evaluation: Dict[str, Any],
    *,
    route_label: str,
) -> str:
    verdict = str(evaluation.get("verdict") or "NOT REALISTIC").upper()
    reasons: List[str] = list(evaluation.get("reasons") or [])

    if verdict == _VERDICT_FEASIBLE:
        lead = (
            f"Yes, the {model} can fly {route_label} nonstop under verified performance data "
            f"and typical NBAA reserve assumptions."
        )
    elif verdict == _VERDICT_MARGINAL:
        lead = (
            f"The {model} is marginal for {route_label}; adverse winds or payload "
            f"may restrict dependable dispatch."
        )
    else:
        lead = (
            f"No, the {model} is not realistic for {route_label} with typical executive "
            f"payload and NBAA reserve assumptions."
        )

    parts = [lead]
    if reasons:
        detail = str(reasons[0]).strip()
        if detail and detail.lower() not in lead.lower():
            if not detail.endswith((".", "!", "?")):
                detail += "."
            parts.append(detail)

    return _guard_answer(" ".join(parts))


def respond_aircraft_capability(
    model: str,
    query: str,
    *,
    mission: Optional[MissionState] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return a deterministic 1–3 sentence capability answer for a named aircraft + corridor.

    Uses ``evaluate_named_aircraft_capability`` — no mission orchestration or ranking.
    """
    name = (model or "").strip()
    if not name:
        return UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE

    if not resolve_catalog_profile_key(name):
        return UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE

    ms = _mission_with_route(query or "", mission)
    if ms is None:
        return UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE

    from services.consultant.named_aircraft_capability import evaluate_named_aircraft_capability

    du = dict(data_used or {})
    du.setdefault("orchestration_query", query or "")

    evaluation = evaluate_named_aircraft_capability(
        name,
        ms,
        data_used=du,
    )
    route_label = _extract_route_label(query or "", ms) or "the stated corridor"
    return _format_capability_answer(
        evaluation.get("model") or name,
        evaluation,
        route_label=route_label,
    )


__all__ = ["respond_aircraft_capability"]

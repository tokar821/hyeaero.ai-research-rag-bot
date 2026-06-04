"""Phase 33 — Mission feasibility audit (final answer only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tests.response_quality._text_extract import (
    extract_aircraft_like_tokens,
    extract_pax,
    mentions_nonstop,
    normalize,
)


@dataclass
class MissionFeasibilityAudit:
    score: float
    failures: List[str]


def _resolve(token: str) -> str:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    return resolve_aircraft_alias(token) or token


def _estimate_route_nm(query: str) -> float:
    q = normalize(query)
    if any(k in q for k in ("london", "singapore", "tokyo", "dubai", "honolulu")):
        return 7000.0
    if any(k in q for k in ("teb", "teterboro", "lax", "los angeles", "miami", "paris")):
        return 2500.0
    return 1200.0


def _range_nm(model: str) -> Optional[float]:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=model)
    return float(rec.nbaa_range_nm) if rec and rec.nbaa_range_nm else None


def audit_mission_feasibility(*, query: str, answer: str) -> MissionFeasibilityAudit:
    failures: List[str] = []

    # Only trigger when the answer asserts feasibility language.
    t = normalize(answer)
    asserts_fit = any(k in t for k in ("can do", "nonstop", "will make", "capable", "fits this mission"))
    if not asserts_fit:
        return MissionFeasibilityAudit(score=100.0, failures=[])

    models = [_resolve(x) for x in extract_aircraft_like_tokens(answer)]
    models = [m for m in models if m]
    if not models:
        return MissionFeasibilityAudit(score=100.0, failures=[])

    pax = extract_pax(query) or extract_pax(answer)
    nonstop = mentions_nonstop(query) or mentions_nonstop(answer)
    route_nm = _estimate_route_nm(query)

    # Hard stop: nonstop+ULR route but all mentioned models are under-range.
    if nonstop and route_nm >= 4000:
        ok = False
        for m in models[:5]:
            r = _range_nm(m)
            if r and r >= route_nm * 0.85:
                ok = True
                break
        if not ok:
            failures.append("MISSION_INFEASIBLE_RECOMMENDATION")

    # Pax: if pax is very high, flag infeasible recommendation (conservative).
    if pax and pax >= 18:
        failures.append("MISSION_INFEASIBLE_RECOMMENDATION")

    score = 100.0 if not failures else 0.0
    return MissionFeasibilityAudit(score=score, failures=sorted(set(failures)))


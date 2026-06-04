"""Phase 32 — Mission fit audit."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tests.production_validation.validation_runner import ValidationResult

_PAX_RE = re.compile(r"\b(\d+)\s*(?:pax|passengers?|people)\b", re.I)
_BUDGET_RE = re.compile(r"\b(?:under|budget|\$)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:M|MM|million)\b", re.I)
_NONSTOP_RE = re.compile(r"\bnonstop\b", re.I)
_ROUTE_RE = re.compile(
    r"\b(?:from\s+)?([\w\s]+?)\s*(?:to|-)\s*([\w\s]+?)(?:\s+nonstop|\s+under|\?|$)",
    re.I,
)

_ULR_MIN_NM = 4000


def parse_mission_constraints(query: str) -> Dict[str, Any]:
    q = query or ""
    pax_m = _PAX_RE.search(q)
    budget_m = _BUDGET_RE.search(q)
    return {
        "pax": int(pax_m.group(1)) if pax_m else None,
        "budget_m": float(budget_m.group(1)) if budget_m else None,
        "nonstop": bool(_NONSTOP_RE.search(q)),
        "query": q,
    }


def _estimate_route_nm(query: str) -> Optional[float]:
    q = (query or "").lower()
    long_haul = any(k in q for k in ("singapore", "tokyo", "london", "dubai", "honolulu"))
    medium = any(k in q for k in ("los angeles", "miami", "paris", "teb", "lax"))
    if long_haul:
        return 7000.0
    if medium:
        return 2500.0
    return 1200.0


def audit_mission_single(result: "ValidationResult") -> List[str]:
    if result.category != "mission":
        return []
    flags: List[str] = []
    constraints = parse_mission_constraints(result.query)
    route_nm = _estimate_route_nm(result.query)

    if constraints.get("pax") and constraints["pax"] >= 14:
        flags.append("MISSION_MISMATCH_HIGH_PAX")

    if constraints.get("nonstop") and route_nm and route_nm >= _ULR_MIN_NM:
        if result.execution_path == "authority_dispatch" and result.authority_models:
            from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

            for m in result.authority_models:
                rec = get_aircraft_authority_record(aircraft_model=m)
                if rec and rec.nbaa_range_nm and rec.nbaa_range_nm < route_nm * 0.85:
                    flags.append("MISSION_MISMATCH_RANGE")

    if constraints.get("budget_m") and result.authority_models:
        from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

        for m in result.authority_models:
            rows = find_catalog_matches([m])
            price = typical_market_price_usd(rows[0]) if rows else 0.0
            cap = constraints["budget_m"] * 1_000_000 * 0.85
            if price > 0 and price > cap:
                flags.append("MISSION_MISMATCH_BUDGET")

    return flags


def audit_mission_fit(results: List["ValidationResult"]) -> Dict[str, Any]:
    mission = [r for r in results if r.category == "mission"]
    if not mission:
        return {"mission_fit_accuracy_pct": 100.0, "mismatches": []}
    mismatches: List[Dict[str, Any]] = []
    ok = 0
    for r in mission:
        flags = audit_mission_single(r)
        if flags:
            mismatches.append({"query_id": r.query_id, "flags": flags})
        else:
            ok += 1
    return {
        "total_mission_queries": len(mission),
        "mission_fit_accuracy_pct": round(100.0 * ok / len(mission), 2),
        "mismatches": mismatches[:50],
    }

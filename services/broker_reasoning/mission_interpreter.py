"""Infer missing mission parameters from conversational budget/mission queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MissionInterpretation:
    acquisition_budget_usd: Optional[float] = None
    acquisition_budget_musd: Optional[float] = None
    passengers: Optional[int] = None
    route: Optional[str] = None
    range_nm: Optional[int] = None
    missing_fields: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acquisition_budget_usd": self.acquisition_budget_usd,
            "acquisition_budget_musd": self.acquisition_budget_musd,
            "passengers": self.passengers,
            "route": self.route,
            "range_nm": self.range_nm,
            "missing_fields": list(self.missing_fields),
            "follow_up_questions": list(self.follow_up_questions),
        }


_DOLLAR_M_RE = re.compile(
    r"(?is)\$\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b",
)
_BUDGET_PATTERNS = (
    _DOLLAR_M_RE,
    re.compile(
        r"(?is)\b(?:under|below|within|budget|max|around|about|at)\s+\$?\s*"
        r"(\d+(?:\.\d+)?)\s*(m|mm|million|mil|k)\b",
    ),
    re.compile(
        r"(?is)\b(?:under|below|within|budget|max|around|about|at)\s+\$?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:m|mm|million|mil)\b",
    ),
    re.compile(r"(?is)\b(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\s+budget\b"),
    re.compile(
        r"(?is)\bbest\s+(?:jet|aircraft)\s+(?:for|around|about)\s+\$?\s*"
        r"(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b",
    ),
    re.compile(
        r"(?is)\bwhat\s+(?:can\s+i|should\s+i)\s+buy\s+for\s+\$?\s*"
        r"(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b",
    ),
    re.compile(
        r"(?is)\bfor\s+\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b",
    ),
)
_NM_CONTEXT_RE = re.compile(r"\b(?:nm|nautical|legs?|passengers?|pax|runway|ft)\b", re.I)
_PAX_RE = re.compile(
    r"\b(\d+)\s*(?:pax|passengers?|people|seats?)\b|\b(?:for|with)\s+(\d+)\s+(?:pax|passengers?|people)\b",
    re.I,
)
_RANGE_RE = re.compile(r"\b(\d{3,5})\s*(?:nm|nautical)\b", re.I)
_ROUTE_RE = re.compile(
    r"\b(?:from\s+)?([A-Za-z]{3,4})\s+(?:to|-)\s+([A-Za-z]{3,4})\b",
    re.I,
)
_CITY_PAIR_RE = re.compile(
    r"\b([A-Za-z][A-Za-z .'-]{2,24}?)\s+to\s+([A-Za-z][A-Za-z .'-]{2,24}?)\b",
    re.I,
)


def _to_usd(amount: str, unit: str) -> Optional[float]:
    try:
        val = float(amount)
    except ValueError:
        return None
    u = (unit or "").lower()
    if u == "k":
        return val * 1_000.0
    if u in ("m", "mm", "million", "mil", "") and val < 10_000:
        return val * 1_000_000.0
    if val >= 10_000:
        return val
    return val * 1_000_000.0


def _extract_budget_musd(query: str) -> Optional[float]:
    q = query or ""
    dm = _DOLLAR_M_RE.search(q)
    if dm:
        usd = _to_usd(dm.group(1), dm.group(2) or "m")
        if usd is not None:
            return usd / 1_000_000.0
    if _NM_CONTEXT_RE.search(q) and not re.search(
        r"(?is)\b(?:under|below|budget|\$\d|(?:\d+(?:\.\d+)?)\s*(?:m|mm|million|mil)\b.*(?:buy|budget))",
        q,
    ):
        return None
    for pat in _BUDGET_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        unit = m.group(2) if m.lastindex and m.lastindex >= 2 else "m"
        usd = _to_usd(m.group(1), unit)
        if usd is not None:
            return usd / 1_000_000.0
    return None


def interpret_mission(query: str) -> MissionInterpretation:
    """Infer mission parameters; list only missing fields needed for a recommendation."""
    q = (query or "").strip()
    budget_m = _extract_budget_musd(q)

    pax: Optional[int] = None
    pm = _PAX_RE.search(q)
    if pm:
        pax = int(pm.group(1) or pm.group(2))

    route: Optional[str] = None
    rm = _ROUTE_RE.search(q)
    if rm:
        route = f"{rm.group(1).upper()}-{rm.group(2).upper()}"
    else:
        cm = _CITY_PAIR_RE.search(q)
        if cm:
            origin = cm.group(1).strip().title()
            dest = cm.group(2).strip().title()
            route = f"{origin}-{dest}"

    range_nm: Optional[int] = None
    rng = _RANGE_RE.search(q)
    if rng:
        range_nm = int(rng.group(1))
    if re.search(r"(?is)\bcoast.?to.?coast\b", q):
        range_nm = max(range_nm or 0, 2600)
        if route is None:
            route = "COAST-TO-COAST"
    elif re.search(r"(?is)\bnonstop\b", q) and (range_nm or 0) < 2200:
        range_nm = 2200

    missing: List[str] = []
    follow_up: List[str] = []

    if budget_m is None and re.search(r"\b(?:buy|best|what|budget|afford)\b", q, re.I):
        missing.append("acquisition_budget")
        follow_up.append("What is your acquisition budget (or a realistic ceiling)?")

    if pax is None and re.search(r"\b(?:mission|route|nonstop|pax|passengers?)\b", q, re.I):
        missing.append("passengers")
        follow_up.append("How many passengers do you typically carry?")

    if route is None and re.search(r"\b(?:nonstop|route|leg|trip)\b", q, re.I):
        missing.append("route")
        follow_up.append("What is the primary city pair or typical mission length?")

    # Budget-only discovery — ask mission priority, not route/pax immediately.
    if budget_m is not None and not route and not pax:
        if re.search(r"\b(?:best|what\s+should|what\s+can)\b", q, re.I):
            missing.extend(["mission_priority"])
            follow_up.append(
                "With that budget, are you prioritizing range, cabin size, operating cost, or a specific mission profile?"
            )

    return MissionInterpretation(
        acquisition_budget_usd=budget_m * 1_000_000.0 if budget_m is not None else None,
        acquisition_budget_musd=budget_m,
        passengers=pax,
        route=route,
        range_nm=range_nm,
        missing_fields=missing,
        follow_up_questions=follow_up[:2],
    )


__all__ = ["MissionInterpretation", "interpret_mission"]

"""
Passenger load as a distribution — planning uses upper bound, not scalar collapse.
"""

from __future__ import annotations

import re
from typing import Optional

from services.mission.models import MissionProfile, PassengerDistribution, PriorityLevel
from services.mission.normalization import normalize_passenger_count

_PAX_RANGE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*(?:passengers?|pax|people|executives?|seats?|person(?:nel)?)\b",
    re.I,
)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(\d{1,2})\s+and\s+(\d{1,2})\s+(?:people|personnel|passengers?|pax|executives?)\b",
    re.I,
)
_ENGINEERS_RANGE_RE = re.compile(
    r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+engineers?\b",
    re.I,
)
_RANGING_RE = re.compile(
    r"\branging\s+from\s+(\d{1,2})\s+(?:to|and)\s+(\d{1,2})\b",
    re.I,
)
_TEAM_SPREAD_RE = re.compile(
    r"\b(\d{1,2})\s+executives?\s+to\s+(\d{1,2})\s*[-–]?\s*person\b",
    re.I,
)
_DEAL_GROUP_RE = re.compile(
    r"\b(\d{1,2})\s*[-–]?\s*person\s+deal\s+groups?\b",
    re.I,
)
_CARGO_RE = re.compile(
    r"\b(?:cargo|equipment|bulky|freight)\s+(?:capacity|aboard|on\s+board|matters?)?\b"
    r"|\b(?:matters?\s+more\s+than\s+cabin|high[- ]value\s+equipment)\b"
    r"|\bneed\s+(?:cargo|equipment)\b",
    re.I,
)
_PAX_SINGLE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:passengers?|pax|people|executives?|seats?)\b"
    r"|\b(?:for\s+)?(\d{1,2})\s+pax\b",
    re.I,
)


def extract_passenger_distribution(text: str) -> PassengerDistribution:
    """Extract min/max/planning load from user text."""
    tl = text or ""
    lo: Optional[int] = None
    hi: Optional[int] = None
    note_parts: list[str] = []

    for pat in (_BETWEEN_RE, _PAX_RANGE_RE, _RANGING_RE, _TEAM_SPREAD_RE, _ENGINEERS_RANGE_RE):
        m = pat.search(tl)
        if m:
            try:
                a, b = int(m.group(1)), int(m.group(2))
                lo, hi = (min(a, b), max(a, b))
                note_parts.append(f"range {lo}-{hi}")
                break
            except (TypeError, ValueError):
                pass

    if lo is None:
        nums: list[int] = []
        for m in _PAX_SINGLE_RE.finditer(tl):
            raw = m.group(1) or m.group(2)
            if raw:
                n = normalize_passenger_count(int(raw))
                if n:
                    nums.append(n)
        dg = _DEAL_GROUP_RE.search(tl)
        if dg:
            n = normalize_passenger_count(int(dg.group(1)))
            if n:
                nums.append(n)
        if len(nums) >= 2:
            lo, hi = min(nums), max(nums)
            note_parts.append(f"multi-mention {lo}-{hi}")
        elif len(nums) == 1:
            lo = hi = nums[0]

    cargo = bool(_CARGO_RE.search(tl))
    if cargo:
        note_parts.append("cargo/equipment")

    planning: Optional[int] = None
    typical: Optional[int] = None
    if hi is not None:
        planning = normalize_passenger_count(hi)
    elif lo is not None:
        planning = normalize_passenger_count(lo)
    if lo is not None and hi is not None:
        typical = normalize_passenger_count((lo + hi) // 2)

    return PassengerDistribution(
        min_pax=lo,
        max_pax=hi,
        planning_load=planning,
        typical_pax=typical,
        cargo_required=cargo,
        variance_note="; ".join(note_parts),
    )


def apply_passenger_distribution_to_profile(profile, dist: PassengerDistribution) -> None:
    """Sync scalar passengers field to planning_load; preserve distribution on profile."""
    if not isinstance(profile, MissionProfile):
        return
    profile.passenger_distribution = dist
    if dist.planning_load is not None:
        profile.passengers = dist.planning_load
    if dist.cargo_required and profile.baggage_priority == PriorityLevel.NONE:
        profile.baggage_priority = PriorityLevel.HIGH

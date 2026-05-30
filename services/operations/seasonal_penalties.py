"""
Seasonal penalties — winter westbound and payload-season realism beyond brochure range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

_WINTER_RE = re.compile(r"\b(?:winter|january|february|december)\b", re.I)
_WESTBOUND_RE = re.compile(r"\bwestbound\b", re.I)
_PACIFIC_RE = re.compile(r"\b(?:tokyo|hong\s+kong|seoul|sydney|pacific)\b", re.I)
_TRANSATLANTIC_RE = re.compile(r"\b(?:london|paris|europe|transatlantic)\b", re.I)


@dataclass(frozen=True)
class SeasonalPenalty:
    extra_nm: float
    payload_factor: float
    label: str
    dispatch_note: str


def infer_seasonal_penalty(
    mission: Any,
    *,
    query: str = "",
    route_label: str = "",
) -> SeasonalPenalty:
    """Compute seasonal NM and payload penalties for planning."""
    blob = f"{query} {route_label}".lower()
    extra = 0.0
    payload_factor = 1.0
    label = "standard"
    note = "No seasonal penalty applied."

    winter = bool(_WINTER_RE.search(blob))
    westbound = bool(_WESTBOUND_RE.search(blob)) or bool(getattr(mission, "westbound", False))
    pacific = bool(_PACIFIC_RE.search(blob))
    transatlantic = bool(_TRANSATLANTIC_RE.search(blob))

    if winter and westbound and pacific:
        extra = 650.0
        payload_factor = 0.88
        label = "winter_westbound_pacific"
        note = "SFO/Tokyo-style winter westbound — brochure range is not dispatch planning range."
    elif winter and westbound and transatlantic:
        extra = 420.0
        payload_factor = 0.92
        label = "winter_westbound_transatlantic"
        note = "LAX–London westbound winter — reserve and headwind margin tighten materially."
    elif winter:
        extra = 180.0
        label = "winter_general"
        note = "Winter operations reduce effective payload and fuel margin."

    return SeasonalPenalty(
        extra_nm=extra,
        payload_factor=payload_factor,
        label=label,
        dispatch_note=note,
    )


__all__ = ["SeasonalPenalty", "infer_seasonal_penalty"]

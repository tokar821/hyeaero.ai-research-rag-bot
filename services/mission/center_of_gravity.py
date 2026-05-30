"""
Operational center-of-gravity detection — what actually drives procurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.aviation_places import ALIAS_TO_PLACE

_DOMESTIC_CORE = frozenset(
    {
        "dallas",
        "houston",
        "chicago",
        "atlanta",
        "new york",
        "miami",
        "boston",
        "denver",
        "los angeles",
        "san francisco",
    }
)
_EPISODIC_HUBS = frozenset(
    {"dubai", "singapore", "riyadh", "tokyo", "hong kong", "sydney", "london", "paris"}
)
_FREQ_DOMINANT_RE = re.compile(
    r"\b(?:most\s+annual\s+utilization|majority\s+of\s+(?:hours|flying)|"
    r"(\d{1,3})\s*%\s*(?:domestic|regional|north\s+america))\b",
    re.I,
)
_OCCASIONAL_RE = re.compile(r"\b(?:occasional(?:ly)?|quarterly|episodic|few\s+times)\b", re.I)


@dataclass
class CenterOfGravityResult:
    primary_hubs: List[str] = field(default_factory=list)
    episodic_nodes: List[str] = field(default_factory=list)
    dominant_band: str = "unresolved"
    domestic_dominant: bool = False
    episodic_distortion_risk: bool = False
    procurement_driver: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_hubs": list(self.primary_hubs),
            "episodic_nodes": list(self.episodic_nodes),
            "dominant_band": self.dominant_band,
            "domestic_dominant": self.domestic_dominant,
            "episodic_distortion_risk": self.episodic_distortion_risk,
            "procurement_driver": self.procurement_driver,
            "notes": list(self.notes),
        }


def _places_in_text(text: str) -> List[str]:
    tl = (text or "").lower()
    found: List[str] = []
    for alias, place in sorted(ALIAS_TO_PLACE.items(), key=lambda x: -len(x[0])):
        if len(alias) < 3:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", tl):
            canon = place.canonical
            if canon not in found:
                found.append(canon)
    return found


def detect_center_of_gravity(
    query: str,
    mission: Any = None,
) -> CenterOfGravityResult:
    """Detect operational center of gravity vs episodic continuation nodes."""
    ql = (query or "").lower()
    places = _places_in_text(ql)
    routes = list(getattr(mission, "routes", None) or []) if mission else []
    for r in routes:
        places.extend(_places_in_text(r))
    places = list(dict.fromkeys(places))

    primary = [p for p in places if any(c in p.lower() for c in _DOMESTIC_CORE)]
    episodic = [p for p in places if any(c in p.lower() for c in _EPISODIC_HUBS)]

    result = CenterOfGravityResult(
        primary_hubs=primary[:6],
        episodic_nodes=episodic[:6],
    )

    if _FREQ_DOMINANT_RE.search(ql) or len(primary) >= 2:
        result.domestic_dominant = True
        result.dominant_band = "domestic_executive"
        result.procurement_driver = "domestic_network"
        result.notes.append("Domestic network dominates annual utilization.")

    if episodic and (_OCCASIONAL_RE.search(ql) or len(episodic) <= 2):
        result.episodic_distortion_risk = bool(primary)
        result.notes.append(
            "Episodic international nodes present — must not override domestic procurement center."
        )

    if not result.primary_hubs and episodic:
        result.dominant_band = "international_continuation"
        result.procurement_driver = "episodic_ulr"
    elif result.domestic_dominant:
        result.procurement_driver = "domestic_executive_core"

    return result


def attach_center_of_gravity_metadata(
    data_used: Optional[Dict[str, Any]],
    query: str,
    mission: Any = None,
) -> CenterOfGravityResult:
    """Persist CoG on data_used for ranking and strategic renderers."""
    result = detect_center_of_gravity(query, mission)
    if isinstance(data_used, dict):
        data_used["mission_center_of_gravity"] = result.to_dict()
    return result


__all__ = [
    "CenterOfGravityResult",
    "detect_center_of_gravity",
    "attach_center_of_gravity_metadata",
]

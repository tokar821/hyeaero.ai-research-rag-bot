"""
Hierarchy weighting — dominant utilization vs continuation vs executive overlays.

Continuation hubs (Dubai, Singapore, Doha, Honolulu) must never override dominant
origin structure or utilization hierarchy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.mission_understanding_engine import MissionUnderstandingPacket

_CONTINUATION_HUBS = frozenset(
    {
        "dubai",
        "dxb",
        "singapore",
        "sin",
        "doha",
        "doh",
        "honolulu",
        "hnl",
        "anchorage",
        "anc",
    }
)

_UTILIZATION_PCT_RE = re.compile(
    r"(\d{1,3})\s*%\s*(?:of\s+)?(?:annual\s+)?(?:hours?|utilization|flying)\s*(?:are|is)\s*(\w[\w\s-]{2,40})",
    re.I,
)
_DOMESTIC_DOMINANCE_RE = re.compile(
    r"\b(?:most\s+flying|majority\s+of\s+(?:hours?|flying)|dominant\s+utilization|"
    r"primary\s+utilization|(\d{1,3})\s*%\s*(?:domestic|corridor|regional))\b",
    re.I,
)
_EXECUTIVE_EXCEPTION_RE = re.compile(
    r"\b(?:executives?\s+occasionally|founder\s+ulr|ceo\s+(?:nonstop|mandate)|"
    r"occasional(?:ly)?\s+(?:fly|trips?)\s+to)\b",
    re.I,
)

HIERARCHY_WEIGHTING_KEY = "hierarchy_weighting"


@dataclass
class HierarchyWeightingResult:
    dominant_utilization: str = ""
    secondary_traffic: List[str] = field(default_factory=list)
    continuation_constraints: List[str] = field(default_factory=list)
    executive_exceptions: List[str] = field(default_factory=list)
    seasonal_overlays: List[str] = field(default_factory=list)
    continuation_hub_discipline: str = ""
    weighting_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_utilization": self.dominant_utilization,
            "secondary_traffic": list(self.secondary_traffic),
            "continuation_constraints": list(self.continuation_constraints),
            "executive_exceptions": list(self.executive_exceptions),
            "seasonal_overlays": list(self.seasonal_overlays),
            "continuation_hub_discipline": self.continuation_hub_discipline,
            "weighting_notes": list(self.weighting_notes),
        }


def _hub_in_text(text: str) -> List[str]:
    tl = (text or "").lower()
    found: List[str] = []
    for hub in _CONTINUATION_HUBS:
        if re.search(rf"\b{re.escape(hub)}\b", tl):
            found.append(hub.upper() if len(hub) <= 4 else hub.title())
    return found


def detect_dominant_mission(
    packet: Optional[MissionUnderstandingPacket],
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> HierarchyWeightingResult:
    """Infer utilization hierarchy from packet constraints, planning data, and query."""
    result = HierarchyWeightingResult()
    inf: Dict[str, Any] = {}
    if packet is not None:
        inf = dict(packet.inferred_constraints or {})

    order = list(inf.get("operational_priority_order") or [])
    if order:
        result.dominant_utilization = str(order[0])
        result.secondary_traffic = [str(x) for x in order[1:4]]

    if inf.get("domestic_utilization_dominates_except_founder_ulr"):
        result.dominant_utilization = result.dominant_utilization or "domestic corridor utilization"
        result.executive_exceptions.append("ULR continuation is secondary unless founder/CEO mandated")
        result.weighting_notes.append(
            "Domestic utilization dominates; ULR legs are overlay traffic, not the primary optimization target."
        )

    if inf.get("domestic_utilization_dominant"):
        result.dominant_utilization = result.dominant_utilization or "domestic executive network"
        result.weighting_notes.append("Domestic executive network dominates annual hours.")

    if inf.get("continuation_hubs_semantic_only_not_primary_origin"):
        result.continuation_hub_discipline = (
            "Continuation hubs are connectors only — they do not become primary origins."
        )

    # Planning hierarchy from phase-2 synthesis
    if isinstance(data_used, dict):
        ph = data_used.get("planning_hierarchy") or {}
        if isinstance(ph, dict):
            util_legs = list(ph.get("utilization_legs") or [])
            cont_legs = list(ph.get("continuation_legs") or [])
            peak = str(ph.get("peak_leg") or "")
            if util_legs and not result.dominant_utilization:
                result.dominant_utilization = f"domestic/corridor legs ({len(util_legs)} route(s))"
            for leg in cont_legs[:4]:
                hubs = _hub_in_text(leg)
                label = leg
                if hubs:
                    label = f"{leg} ({', '.join(hubs)} continuation)"
                if label not in result.continuation_constraints:
                    result.continuation_constraints.append(label)
            if peak and cont_legs and peak in cont_legs:
                result.weighting_notes.append(
                    "Peak leg is a continuation segment — it must not override dominant origin structure."
                )

    # Query-level utilization percentages
    q = query or ""
    m = _UTILIZATION_PCT_RE.search(q)
    if m:
        pct, domain = m.group(1), m.group(2).strip()
        result.dominant_utilization = f"{domain.strip()} ({pct}% of annual hours)"
        result.weighting_notes.append(
            f"Stated utilization split: {pct}% {domain} — this dominates the structural interpretation."
        )

    if _DOMESTIC_DOMINANCE_RE.search(q) and not result.dominant_utilization:
        result.dominant_utilization = "domestic corridor / regional executive network"

    if _EXECUTIVE_EXCEPTION_RE.search(q):
        for hub in _hub_in_text(q):
            exc = f"Executive exception traffic via {hub}"
            if exc not in result.executive_exceptions:
                result.executive_exceptions.append(exc)
        if not result.executive_exceptions:
            result.executive_exceptions.append("Executive ULR exceptions — secondary to dominant utilization")

    if packet and packet.travel_pattern and packet.travel_pattern != "unknown":
        if not result.dominant_utilization:
            result.dominant_utilization = packet.travel_pattern

    if inf.get("westbound_winter_pressure") or inf.get("seasonal_overlay"):
        result.seasonal_overlays.append("Seasonal/westbound winter overlay — does not redefine primary domain")

    if not result.continuation_hub_discipline and result.continuation_constraints:
        result.continuation_hub_discipline = (
            "Continuation hubs remain secondary to dominant origin structure and utilization hierarchy."
        )

    return result


def format_hierarchy_weighting_section(result: HierarchyWeightingResult) -> str:
    """Render hierarchy weighting as a broker-grade analytical block."""
    lines: List[str] = ["Utilization Hierarchy"]
    if result.dominant_utilization:
        lines.append(f"- dominant: {result.dominant_utilization}")
    else:
        lines.append("- dominant: unresolved — state annual hour split before model selection")

    lines.append("")
    lines.append("Secondary / Overlay Traffic")
    secondary = (
        result.secondary_traffic
        + result.executive_exceptions
        + result.seasonal_overlays
    )
    if secondary:
        for s in secondary[:6]:
            lines.append(f"- {s}")
    else:
        lines.append("- none identified")

    lines.append("")
    lines.append("Continuation Constraints")
    if result.continuation_constraints:
        for c in result.continuation_constraints[:5]:
            lines.append(f"- {c}")
    else:
        lines.append("- none")

    if result.continuation_hub_discipline:
        lines.append("")
        lines.append("Continuation Hub Discipline")
        lines.append(f"- {result.continuation_hub_discipline}")

    if result.weighting_notes:
        lines.append("")
        lines.append("Weighting Notes")
        for n in result.weighting_notes[:4]:
            lines.append(f"- {n}")

    return "\n".join(lines).strip()


def attach_hierarchy_weighting_metadata(
    data_used: Optional[Dict[str, Any]],
    result: HierarchyWeightingResult,
) -> None:
    if isinstance(data_used, dict):
        data_used[HIERARCHY_WEIGHTING_KEY] = result.to_dict()


__all__ = [
    "HIERARCHY_WEIGHTING_KEY",
    "HierarchyWeightingResult",
    "attach_hierarchy_weighting_metadata",
    "detect_dominant_mission",
    "format_hierarchy_weighting_section",
]

"""
Context continuity — preserve mission evolution, procurement anchors, and operator priorities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_REFERENCE_AIRCRAFT_RE = re.compile(
    r"\b(?:still|same|lower\s+(?:operating\s+)?costs?\s+than|cheaper\s+than|below|without\s+moving\s+up\s+from)\s+"
    r"(?:a|an|the|our)?\s*"
    r"(global\s+7500|global\s+6500|g650er?|gulfstream\s+g650|falcon\s+8x|g600|g500)\b",
    re.I,
)
_G650_CONTINUITY_RE = re.compile(
    r"\b(?:lower\s+operating\s+costs?\s+than|below|cheaper\s+than)\s+(?:a\s+)?g650(?:er)?\b",
    re.I,
)
_SAME_NETWORK_RE = re.compile(
    r"\b(?:same\s+network|same\s+routes?|as\s+before|like\s+before|prior\s+mission)\b",
    re.I,
)
_PAYLOAD_CONTINUITY_RE = re.compile(
    r"\b(?:without\s+payload\s+penalties?|same\s+passenger|still\s+\d+\s+passengers?)\b",
    re.I,
)
_COST_CEILING_RE = re.compile(
    r"\b(?:below|lower\s+(?:operating\s+)?costs?\s+than|under)\s+(?:a\s+)?(?:global\s+7500|g650|g650er)\b",
    re.I,
)


@dataclass
class ContextContinuityState:
    reference_aircraft: str = ""
    cost_ceiling_reference: str = ""
    network_phrase: str = ""
    passenger_anchor: Optional[int] = None
    carry_forward_routes: List[str] = field(default_factory=list)
    continuity_phrases: List[str] = field(default_factory=list)
    apply_to_ranking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_aircraft": self.reference_aircraft,
            "cost_ceiling_reference": self.cost_ceiling_reference,
            "network_phrase": self.network_phrase,
            "passenger_anchor": self.passenger_anchor,
            "carry_forward_routes": list(self.carry_forward_routes),
            "continuity_phrases": list(self.continuity_phrases),
            "apply_to_ranking": self.apply_to_ranking,
        }


def _extract_reference_aircraft(query: str) -> str:
    m = _REFERENCE_AIRCRAFT_RE.search(query or "")
    if not m:
        return ""
    raw = m.group(1).strip()
    try:
        from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

        return resolve_canonical_display_name(raw) or raw
    except Exception:
        return raw


def resolve_context_continuity(
    query: str,
    *,
    broker_memory: Optional[Dict[str, Any]] = None,
    continuity_assessment: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> ContextContinuityState:
    """Resolve sticky context from query + session memory."""
    state = ContextContinuityState()
    ql = (query or "").lower()

    ref = _extract_reference_aircraft(query)
    if not ref and _G650_CONTINUITY_RE.search(ql):
        ref = "Gulfstream G650"
    if ref:
        state.reference_aircraft = ref
        state.continuity_phrases.append(f"reference_aircraft:{ref}")

    if _COST_CEILING_RE.search(ql):
        state.cost_ceiling_reference = state.reference_aircraft or "Global 7500"
        state.continuity_phrases.append("cost_ceiling_active")

    if _SAME_NETWORK_RE.search(ql):
        state.network_phrase = "prior_network"
        state.continuity_phrases.append("same_network")
        if isinstance(broker_memory, dict):
            state.carry_forward_routes = list(broker_memory.get("recurring_routes") or [])[:6]

    pax_m = re.search(r"\bstill\s+(\d{1,2})\s+passengers?\b", ql)
    if pax_m:
        state.passenger_anchor = int(pax_m.group(1))
    elif _PAYLOAD_CONTINUITY_RE.search(ql) and isinstance(broker_memory, dict):
        pass  # posture only

    conf = 0.0
    if isinstance(continuity_assessment, dict):
        conf = float(continuity_assessment.get("continuity_confidence") or 0)
        if continuity_assessment.get("mission_pivot"):
            state.carry_forward_routes = []
            state.network_phrase = ""

    if isinstance(broker_memory, dict) and conf >= 0.5:
        if not state.reference_aircraft:
            prefs = broker_memory.get("preferred_categories") or []
            if prefs:
                state.continuity_phrases.append(f"prior_category:{prefs[0]}")
        if not state.carry_forward_routes:
            state.carry_forward_routes = list(broker_memory.get("recurring_routes") or [])[:6]

    state.apply_to_ranking = bool(
        state.reference_aircraft
        or state.cost_ceiling_reference
        or state.carry_forward_routes
        or state.network_phrase
        or conf >= 0.72
    )
    return state


def attach_context_continuity(
    data_used: Optional[Dict[str, Any]],
    query: str,
    *,
    broker_memory: Optional[Dict[str, Any]] = None,
    continuity_assessment: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> ContextContinuityState:
    """Persist continuity state for ranking, rendering, and memory."""
    state = resolve_context_continuity(
        query,
        broker_memory=broker_memory,
        continuity_assessment=continuity_assessment,
        history=history,
    )
    if isinstance(data_used, dict):
        data_used["context_continuity"] = state.to_dict()
        if state.reference_aircraft:
            data_used["continuity_reference_aircraft"] = state.reference_aircraft
        if state.cost_ceiling_reference:
            data_used["continuity_cost_ceiling_reference"] = state.cost_ceiling_reference
    return state


def apply_continuity_to_mission_state(mission: Any, state: ContextContinuityState) -> Any:
    """Merge carry-forward routes and passenger anchor into mission state."""
    if state.passenger_anchor and not getattr(mission, "passenger_count", None):
        mission.passenger_count = state.passenger_anchor
    if state.carry_forward_routes and not getattr(mission, "routes", None):
        mission.routes = list(state.carry_forward_routes)[:4]
    return mission


__all__ = [
    "ContextContinuityState",
    "resolve_context_continuity",
    "attach_context_continuity",
    "apply_continuity_to_mission_state",
]

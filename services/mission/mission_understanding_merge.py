"""
Merge LLM mission inference into deterministic MissionUnderstandingPacket.

Explicit extracted facts and rule-derived signals win over LLM guesses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.mission.llm_mission_understanding import LLMMissionUnderstandingResult
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.mission_profile_inference import UtilizationStyle
from services.mission.mission_operational_graph import PROTECTED_INFERRED_KEYS

_PRIORITY_RANK = {"standard": 0, "secondary": 1, "high": 2}


def _upgrade_priority(current: str, proposed: str) -> str:
    cur = (current or "standard").strip().lower()
    prop = (proposed or "").strip().lower()
    if _PRIORITY_RANK.get(prop, 0) > _PRIORITY_RANK.get(cur, 0):
        return prop if prop in ("high", "secondary", "standard") else cur
    return cur


def _merge_unique_str(existing: List[str], incoming: List[str], *, limit: int = 8) -> List[str]:
    out = list(existing or [])
    for item in incoming or []:
        s = str(item or "").strip()
        if not s:
            continue
        if s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _merge_constraints(
    base: Dict[str, Any],
    llm: Dict[str, Any],
    *,
    protected_keys: Optional[set[str]] = None,
) -> Dict[str, Any]:
    merged = dict(base or {})
    protected = protected_keys or set()
    for key, value in (llm or {}).items():
        if key in protected and key in merged:
            continue
        if key not in merged:
            merged[key] = value
    return merged


def merge_llm_understanding_into_packet(
    packet: MissionUnderstandingPacket,
    llm: LLMMissionUnderstandingResult,
    *,
    rule_confidence: float,
) -> MissionUnderstandingPacket:
    """
    Blend LLM inference into an existing rules-based packet.

    - Explicit constraints are never modified here.
    - LLM fills gaps and enriches inferred posture.
    - Recommendation gating remains rule-controlled (``recommend_aircraft`` untouched).
    """
    if not llm.ok:
        packet.confidence_scores["llm_inference"] = 0.0
        if llm.error:
            packet.understanding_notes.append(f"LLM understanding skipped: {llm.error}")
        return packet

    protected = set(packet.inferred_constraints.keys()) | PROTECTED_INFERRED_KEYS
    packet.inferred_constraints = _merge_constraints(
        packet.inferred_constraints,
        llm.inferred_constraints,
        protected_keys=protected,
    )
    packet.operational_environment = _merge_unique_str(
        packet.operational_environment,
        llm.operational_environment,
    )
    packet.understanding_notes = _merge_unique_str(
        packet.understanding_notes,
        llm.understanding_notes,
        limit=12,
    )

    if packet.ownership_profile in ("unknown", "") and llm.ownership_profile:
        packet.ownership_profile = llm.ownership_profile
    if packet.travel_pattern in ("unknown", "") and llm.travel_pattern:
        packet.travel_pattern = llm.travel_pattern
    if packet.corridor_type in ("unknown", "") and llm.corridor_type:
        packet.corridor_type = llm.corridor_type
    if packet.utilization_style in (UtilizationStyle.UNKNOWN, "", None) and llm.utilization_style:
        packet.utilization_style = llm.utilization_style

    packet.runway_complexity = _upgrade_priority(packet.runway_complexity, llm.runway_complexity)
    packet.dispatch_priority = _upgrade_priority(packet.dispatch_priority, llm.dispatch_priority)
    packet.comfort_priority = _upgrade_priority(packet.comfort_priority, llm.comfort_priority)
    packet.operating_cost_priority = _upgrade_priority(
        packet.operating_cost_priority, llm.operating_cost_priority
    )
    packet.nonstop_priority = _upgrade_priority(packet.nonstop_priority, llm.nonstop_priority)

    rule_syn = (packet.operational_synthesis or "").strip()
    llm_syn = (llm.operational_synthesis or "").strip()
    if llm_syn and not rule_syn:
        packet.operational_synthesis = llm_syn
    # When rules already built synthesis (enriched), do not append LLM generic prose.

    if llm.clarifying_question and "clarifying_question" not in packet.inferred_constraints:
        packet.inferred_constraints["llm_clarifying_question"] = llm.clarifying_question

    packet.confidence_scores["llm_inference"] = llm.confidence
    blended = min(
        1.0,
        max(
            float(rule_confidence),
            float(rule_confidence) * 0.72 + float(llm.confidence) * 0.28,
        ),
    )
    packet.overall_confidence = blended
    packet.understanding_notes.append("Hybrid mission understanding: rules + LLM inference.")
    return packet


__all__ = ["merge_llm_understanding_into_packet"]

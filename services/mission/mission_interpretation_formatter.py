"""Mission interpretation formatter — semantic/structural output only.

This is the "mission architect" voice layer. It must:
- classify operational domains explicitly
- preserve hierarchy (primary utilization vs continuation vs overlays)
- state structural conflicts + decomposition status
- end with an authoritative verdict

Hard constraints:
- no aircraft model names
- no route dumps / graph prints
- no generic consultant filler
- avoid airline network jargon unless necessary
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

from services.consultant.mission_state import MissionState
from services.mission.mission_authority_kernel import MissionAuthorityKernel
from services.mission.mission_understanding_engine import MissionUnderstandingPacket

if TYPE_CHECKING:
    from services.orchestration.hierarchy_weighting import HierarchyWeightingResult

_BANNED_GENERIC_RE = re.compile(
    r"\b(?:to optimize your operations|here's a breakdown|you should consider|"
    r"operational flexibility is important|this reflects a diversified strategy)\b",
    re.I,
)
_BANNED_JARGON_RE = re.compile(
    r"\b(?:hub-and-spoke|point-to-point|network theory|airline)\b", re.I
)
_AIRCRAFT_LEAK_RE = re.compile(
    r"\b(?:gulfstream|global\s+\d+|falcon\s+\d+|citation|learjet|embraer|phenom|hawker)\b",
    re.I,
)


def _domain_labels(packet: MissionUnderstandingPacket) -> List[str]:
    inf = packet.inferred_constraints or {}
    domains = list(inf.get("mission_semantic_domains") or [])
    weights = inf.get("mission_domain_weights") or {}
    out: List[str] = []
    for d in domains:
        w = weights.get(d)
        if isinstance(w, (int, float)):
            out.append(f"{d} (weight {round(float(w), 2)})")
        else:
            out.append(str(d))
    if not out and packet.operational_environment:
        out = [str(x) for x in packet.operational_environment[:8]]
    return out


def _primary_utilization(packet: MissionUnderstandingPacket) -> str:
    inf = packet.inferred_constraints or {}
    order = list(inf.get("operational_priority_order") or [])
    if order:
        return str(order[0])
    if packet.travel_pattern and packet.travel_pattern != "unknown":
        return packet.travel_pattern
    return "unknown"


def _secondary_traffic(packet: MissionUnderstandingPacket) -> List[str]:
    inf = packet.inferred_constraints or {}
    order = list(inf.get("operational_priority_order") or [])
    if len(order) >= 2:
        return [str(x) for x in order[1:4]]
    cont = [k for k, v in inf.items() if v is True and "continuation" in k]
    return cont[:3]


def _conflicts(kernel: MissionAuthorityKernel, packet: MissionUnderstandingPacket) -> List[str]:
    inf = packet.inferred_constraints or {}
    invalid = list(inf.get("semantic_invalid_interpretations") or [])
    conflicts: List[str] = [str(x) for x in invalid[:6]]
    if inf.get("incompatible_mission_bands") and "incompatible_mission_bands" not in conflicts:
        conflicts.append("incompatible_mission_bands")
    if kernel.structural_decomposition and kernel.structural_reason:
        conflicts.append(kernel.structural_reason)
    return conflicts[:8]


def _continuation_semantics(packet: MissionUnderstandingPacket) -> List[str]:
    inf = packet.inferred_constraints or {}
    out: List[str] = []
    if inf.get("continuation_hubs_semantic_only_not_primary_origin"):
        out.append("Continuation hubs are connectors only — they do not become primary origins.")
    if inf.get("ulr_continuation_requires_mandate_hub_origin"):
        out.append(
            "Continuation legs require mandate-hub authority (CEO/founder), not generic routing."
        )
    if inf.get("domestic_utilization_dominates_except_founder_ulr"):
        out.append(
            "Domestic utilization remains primary; ULR continuation is secondary traffic unless mandated."
        )
    return out


def _verdict(kernel: MissionAuthorityKernel, packet: MissionUnderstandingPacket) -> str:
    inf = packet.inferred_constraints or {}
    if kernel.structural_decomposition or inf.get("defer_global_shortlist") or inf.get(
        "multi_hard_domain_mission"
    ):
        return "This is structurally a decomposed fleet problem."
    if inf.get("arctic_hard_domain") or inf.get("industrial_hard_domain") or inf.get(
        "mining_hard_domain"
    ):
        return "Industrial reliability requirements dominate the acquisition problem."
    return "This is a coherent single-mission utilization profile (structure-first; no aircraft yet)."


def _structural_conflict_statement(
    kernel: MissionAuthorityKernel,
    packet: MissionUnderstandingPacket,
) -> Optional[str]:
    inf = packet.inferred_constraints or {}
    if kernel.structural_decomposition or inf.get("multi_hard_domain_mission"):
        return "This is structurally multiple missions — single-platform optimization is operationally unstable."
    if inf.get("incompatible_mission_bands"):
        return "Incompatible operational domains coexist — this is not a single optimization problem."
    return None


def format_mission_interpretation(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    kernel: MissionAuthorityKernel,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    hierarchy: Optional["HierarchyWeightingResult"] = None,
) -> str:
    """Render structured mission interpretation only (no aircraft, no route dumps)."""
    del mission, query, data_used, hierarchy

    lines: List[str] = []
    lines.append("Operational Interpretation")
    conflict_headline = _structural_conflict_statement(kernel, packet)
    if conflict_headline:
        lines.append(f"- {conflict_headline}")

    lines.append("")
    lines.append("Operational Structure")
    lines.append(f"- decomposition: {'yes' if kernel.structural_decomposition else 'no'}")
    if kernel.structural_reason:
        lines.append(f"- structural_reason: {kernel.structural_reason}")

    lines.append("")
    lines.append("Primary Utilization")
    lines.append(f"- {_primary_utilization(packet)}")

    lines.append("")
    lines.append("Secondary / Continuation Traffic")
    for s in _secondary_traffic(packet):
        lines.append(f"- {s}")
    for s in _continuation_semantics(packet):
        lines.append(f"- {s}")

    lines.append("")
    lines.append("Operational Domains")
    for d in _domain_labels(packet)[:10]:
        lines.append(f"- {d}")

    lines.append("")
    lines.append("Structural Conflicts")
    conflicts = _conflicts(kernel, packet)
    if conflicts:
        for c in conflicts:
            lines.append(f"- {c}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Interpretation Verdict")
    lines.append(f"- {_verdict(kernel, packet)}")

    text = "\n".join(lines).strip()
    text = _BANNED_GENERIC_RE.sub("", text)
    text = _BANNED_JARGON_RE.sub("", text)
    if _AIRCRAFT_LEAK_RE.search(text):
        text = _AIRCRAFT_LEAK_RE.sub("[REDACTED]", text)

    # hard gate against route dumps
    cleaned: List[str] = []
    for line in text.splitlines():
        if "->" in line or "Routes:" in line:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def is_interpretation_only_query(query: str) -> bool:
    """Heuristic: user asked for structure/interpretation, not aircraft options."""
    try:
        from services.orchestration.response_mode_classifier import (
            classify_orchestration_response_mode,
            explicit_aircraft_request,
        )

        if explicit_aircraft_request(query):
            return False
        result = classify_orchestration_response_mode(query)
        return result.suppresses_aircraft_recommendations
    except Exception:
        pass

    q = (query or "").lower()
    if re.search(r"\b(?:which\s+aircraft|which\s+jet|recommend|shortlist|models?)\b", q):
        return False
    return bool(
        re.search(
            r"\b(?:"
            r"mission\s+structure|how\s+should\s+this\s+be\s+(?:understood|interpreted)|"
            r"how\s+should\s+(?:this|the)\s+network\s+be\s+(?:interpreted|represented)|"
            r"what\s+structure\s+fits|is\s+this\s+(?:structurally\s+)?coherent|"
            r"what\s+operational\s+domains\s+exist|what\s+(?:actually\s+)?dominates?\s+utilization|"
            r"continuation\s+hubs?\s+be\s+represented|dominant\s+mission\s+domains|"
            r"interpret|classification|decomposed|structural\s+conflict|utilization"
            r")\b",
            q,
        )
    )


__all__ = ["format_mission_interpretation", "is_interpretation_only_query"]


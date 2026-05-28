"""
Strategic comparison renderer — for fleet archetype tradeoffs (no aircraft spec tables).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState


def format_strategic_comparison_response(
    mission: MissionState,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    ql = (query or "").lower()

    lines: List[str] = [
        "## STRATEGIC COMPARISON",
        "",
        "This is a fleet-structure question — comparing *operational coherence* and *dispatch reliability*, not brochure range.",
        "",
    ]

    if "los angeles" in ql and "tokyo" in ql:
        lines.append("- **Peak capability driver:** Pacific long-stage (LAX ↔ Tokyo) is a true capability constraint.")
    if "caribbean" in ql or "miami" in ql:
        lines.append("- **Dominant utilization clue:** high-cycle regional legs (Miami/Caribbean) punish an oversized ULR platform.")
    if "houston" in ql or "energy" in ql or "permian" in ql:
        lines.append("- **Operational domain:** energy ops implies short-notice, high-frequency dispatch where availability matters more than max range.")

    lines.extend(
        [
            "",
            "### Option A: One ultra-long-range flagship",
            "- **Strength:** minimizes tech stops and protects the longest oceanic legs.",
            "- **Weakness:** becomes a mismatch for short corridors (cycle cost, scheduling inefficiency, crew pairing), which *creates* dispatch pressure when it’s used as a regional shuttle.",
            "- **Failure mode:** aircraft is “available” but operationally wrong for the majority of legs → utilization incoherence and maintenance-driven downtime on the wrong mission set.",
            "",
            "### Option B: Segmented fleet (super-midsize + regional)",
            "- **Strength:** high dispatch reliability on the dominant utilization band (regional/domestic) while still covering long legs with the appropriate class.",
            "- **Weakness:** adds training/crew/maintenance complexity; requires disciplined scheduling so the long-range aircraft isn’t underutilized.",
            "- **Failure mode:** poor governance (wrong aircraft assigned) or insufficient annual hours on the long-range tail.",
            "",
            "### Dispatch-reliability verdict",
            "- If **most hours are domestic/regional**, the segmented fleet usually produces **fewer dispatch failures** (better fit, fewer forced compromises).",
            "- If **the long-haul leg is truly frequent and mission-critical**, a single ULR flagship can be coherent — but only if it *isn’t* also forced to cover high-cycle regional flying.",
        ]
    )

    if isinstance(data_used, dict):
        data_used["broker_narrative_authoritative"] = True
        data_used["strategic_comparison_renderer"] = True

    return "\n".join(lines).strip()


__all__ = ["format_strategic_comparison_response"]


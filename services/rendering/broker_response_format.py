"""
Broker response formatting — readable structure without UI framework dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def section_header(title: str, level: int = 2) -> str:
    prefix = "#" * min(max(level, 2), 4)
    return f"{prefix} {title}\n"


def bullet_list(items: List[str], *, max_items: int = 8) -> str:
    return "\n".join(f"- {item}" for item in items[:max_items] if item)


def format_recommendation_card(
    rank: Any,
    aircraft: str,
    fit: str,
    *,
    procurement_note: str = "",
    operational_note: str = "",
) -> str:
    lines = [f"### {rank}. {aircraft}", f"- **Fit:** {fit}"]
    if procurement_note:
        lines.append(f"- **Procurement:** {procurement_note}")
    if operational_note:
        lines.append(f"- **Operations:** {operational_note}")
    return "\n".join(lines)


def format_comparison_intelligence_block(rows: List[Dict[str, Any]]) -> str:
    """Operational tradeoff block below comparison table."""
    if not rows:
        return ""
    lines = ["", "### Operational tradeoffs", ""]
    for row in rows:
        label = str(row.get("label") or row.get("aircraft_id") or "Aircraft")
        maint = row.get("maintenance_ecosystem") or "—"
        dispatch = row.get("dispatch_maturity") or "—"
        cabin = row.get("cabin_usability") or "—"
        airport = row.get("airport_flexibility") or "—"
        lines.append(f"**{label}**")
        lines.append(f"- Maintenance / support: {maint}")
        lines.append(f"- Dispatch maturity: {dispatch}")
        lines.append(f"- Cabin usability: {cabin}")
        lines.append(f"- Airport flexibility: {airport}")
        lines.append("")
    return "\n".join(lines).strip()


def format_continuity_acknowledgment(continuity: Dict[str, Any]) -> str:
    """Brief acknowledgment when session context carries forward."""
    if not continuity:
        return ""
    ref = continuity.get("reference_aircraft")
    network = continuity.get("network_phrase")
    parts = []
    if ref:
        parts.append(f"still evaluating against **{ref}**")
    if network:
        parts.append(f"same network context: {network}")
    if not parts:
        return ""
    return "_Continuing from prior turn: " + "; ".join(parts) + "._\n\n"


__all__ = [
    "section_header",
    "bullet_list",
    "format_recommendation_card",
    "format_comparison_intelligence_block",
    "format_continuity_acknowledgment",
]

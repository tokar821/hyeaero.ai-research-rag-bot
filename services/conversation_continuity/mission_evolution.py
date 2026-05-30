"""
Mission evolution — follow-up turns that change constraints (e.g. Aspen winter frequency).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_EVOLUTION_RE = re.compile(
    r"\b(?:"
    r"would\s+(?:your\s+)?(?:answer|recommendation)\s+change|"
    r"recommendation\s+change\s+if|"
    r"if\s+.+\s+become|"
    r"became\s+a\s+quarterly|"
    r"how\s+would\s+that\s+change"
    r")\b",
    re.I,
)
_MISSION_SHIFT_RE = re.compile(
    r"\b(?:aspen|vail|telluride|tokyo|singapore|quarterly\s+mission|winter\s+westbound\s+pacific)\b",
    re.I,
)


def is_mission_evolution_query(query: str) -> bool:
    q = query or ""
    return bool(_EVOLUTION_RE.search(q) and _MISSION_SHIFT_RE.search(q))


def _evolution_theme(query: str) -> str:
    ql = (query or "").lower()
    if re.search(r"\b(?:tokyo|singapore|hong\s+kong|asia|pacific)\b", ql):
        return "ulr_continuation"
    if re.search(r"\b(?:aspen|vail|telluride|jackson|ski)\b", ql):
        return "mountain_winter"
    return "general"


def format_mission_evolution_response(
    query: str,
    mission: Any,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Broker narrative when episodic constraints evolve mid-conversation."""
    from services.rendering.broker_response_format import format_continuity_acknowledgment

    du = data_used if isinstance(data_used, dict) else {}
    ack = format_continuity_acknowledgment(du.get("context_continuity") or {})
    theme = _evolution_theme(query)

    prior_routes: List[str] = []
    bm = du.get("hye_broker_memory") or {}
    if isinstance(bm, dict):
        prior_routes = list(bm.get("recurring_routes") or [])[:4]
    if not prior_routes:
        prior_routes = list((du.get("context_continuity") or {}).get("carry_forward_routes") or [])[:4]

    lines: List[str] = ["## Mission evolution — recommendation shift", ""]
    if ack:
        lines.append(ack)
        lines.append("")

    if theme == "ulr_continuation":
        lines.append(
            "Yes — **quarterly Tokyo (or similar ULR continuation)** would change the recommendation "
            "versus a Texas–Florida + episodic London profile alone."
        )
        lines.append("")
        lines.append("### What changes operationally")
        lines.append(
            "- **Peak stage becomes binding:** westbound Pacific winter reserves and payload "
            "move from episodic to procurement-critical — brochure range is not planning range."
        )
        lines.append(
            "- **Category floor rises:** domestic-optimized super-midsize and large-cabin types "
            "lose credibility for dependable nonstop Asia continuation; ULR band or segmented lift required."
        )
        lines.append(
            "- **Network shape:** do not size the whole fleet around <5% of hours — treat Tokyo as "
            "**continuation segment** (ULR + charter) unless frequency justifies a dedicated ULR asset."
        )
        lines.append("")
        lines.append("### Broker posture")
        lines.append(
            "- **If Tokyo is truly quarterly:** keep domestic-optimized lift for the 80%+ network; "
            "add **ULR charter, card, or fractional** for continuation — or a second ULR only if "
            "schedule control justifies the fixed cost."
        )
        lines.append(
            "- **If Tokyo becomes monthly:** re-center procurement on **Falcon 8X / Global 6500 / G650ER** "
            "class with documented westbound margin — not a Caribbean shuttle airframe."
        )
    elif theme == "mountain_winter":
        lines.append(
            "Yes — **frequent Aspen / high-altitude winter operations** would change the recommendation "
            "versus a warm-climate domestic + episodic London profile alone."
        )
        lines.append("")
        lines.append("### What changes operationally")
        lines.append(
            "- **Field performance becomes binding:** ASE-style runways, winter density altitude, "
            "and de-icing / FBO reliability move from edge-case to procurement drivers."
        )
        lines.append(
            "- **Category floor rises:** light and entry super-mid types lose credibility for "
            "reliable winter mountain dispatch; strong hot/high scores matter more."
        )
        lines.append(
            "- **Network shape:** you need **dual competence** (shuttle + mountain winter) or **segmented lift**."
        )
        lines.append("")
        lines.append("### Broker posture")
        lines.append(
            "- **If Aspen winter is frequent:** bias toward **Challenger 650**, **Falcon 2000**, "
            "**Gulfstream G280** with documented hot/high margin."
        )
        lines.append(
            "- **If Aspen stays occasional:** keep the warm-climate platform and use **charter** "
            "for ski peaks — do not distort the fleet around episodic mountain legs."
        )
    else:
        lines.append(
            "Yes — the stated mission shift would change procurement posture versus the prior network alone."
        )
        lines.append("")
        lines.append("### What changes operationally")
        lines.append("- Re-weight **peak leg** and **dispatch-critical** segments before cabin or brand preferences.")
        lines.append("- Treat new long-range nodes as **continuation** unless frequency crosses ~25% of annual hours.")

    if prior_routes:
        lines.append("")
        lines.append(f"**Prior network context:** {', '.join(prior_routes[:3])}")

    lines.append("")
    lines.append(
        "Specify expected **frequency** and **typical passenger load** on the new segment if you want a revised ranked shortlist."
    )

    if isinstance(data_used, dict):
        data_used["mission_evolution_response"] = True
        data_used["broker_narrative_authoritative"] = True

    return "\n".join(lines)

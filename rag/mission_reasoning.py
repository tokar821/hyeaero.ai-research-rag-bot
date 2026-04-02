"""
Aviation mission / recommendation hints for the LLM (great-circle distance, fuel-stop cue).

Output is **operator context** — must not be echoed as internal system messaging to the client.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rag.aviation_engines.geo import ICAO_COORDS, extract_icaos, nm_between


def build_mission_reasoning_hint(
    query: str,
    fine_intent: str,
    entities: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Returns a short block for the **draft model context** (not verbatim to user as a system quote).

    For ``aviation_mission`` / ``aircraft_recommendation``, inserts distance + stop cue when ≥2 known ICAOs.
    """
    ent = entities or {}
    icaos: List[str] = []
    if isinstance(ent.get("icaos"), list):
        icaos.extend(str(x).upper() for x in ent["icaos"] if x)
    icaos.extend(extract_icaos(query or ""))
    icaos = list(dict.fromkeys(icaos))[:6]

    fi = (fine_intent or "").strip().lower()

    lines: List[str] = []
    if len(icaos) >= 2:
        coords = [ICAO_COORDS[c] for c in icaos[:2] if c in ICAO_COORDS]
        if len(coords) == 2:
            nm = nm_between(coords[0], coords[1])
            lines.append(
                "[MISSION PLANNING NOTE — for model reasoning only; paraphrase professionally for the client]\n"
                f"- Great-circle distance estimate between {icaos[0]} and {icaos[1]}: ~{nm:.0f} nm (no wind; not a flight plan).\n"
                "- Compare to **realistic** aircraft range with reserves; flag likely fuel stops for missions near or over practical range.\n"
                "- Transatlantic eastbounds often stage via Newfoundland, Iceland/Greenland, or Azores depending on aircraft and winds — qualitative only unless context cites a route."
            )
    elif fi in ("aviation_mission", "aircraft_recommendation"):
        lines.append(
            "[MISSION PLANNING NOTE — for model reasoning only]\n"
            "- Assess mission against **practical** range (not brochure max): reserves, alternates, winds, payload.\n"
            "- State when tech stops are **likely** vs **nonstop**; avoid guaranteeing routing without OFP context."
        )

    if fi == "aircraft_recommendation":
        lines.append(
            "[RECOMMENDATION NOTE — for model reasoning only]\n"
            "- If the user gave pax / budget / rough mission, suggest **3–5** aircraft families from context; explain tradeoffs (range, cabin, operating cost bands)."
        )

    if not lines:
        return ""
    return "\n\n".join(lines)

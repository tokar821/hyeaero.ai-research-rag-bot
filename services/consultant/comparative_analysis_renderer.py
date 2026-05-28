"""
Comparative analysis renderer — structured tradeoff tables for class comparisons.

Used for converted airliner vs large business jet and similar explicit comparisons.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from services.consultant.mission_state import MissionState


_COMPARATIVE_RE = re.compile(
    r"\b(?:"
    r"converted\s+airliner|at\s+what\s+point|rational\s+than\s+a\s+large\s+business\s+jet|"
    r"large\s+business\s+jet\?|bbj|acj319|acj\s*319|airbus\s+acj"
    r")\b",
    re.I,
)

_NAMED_COMPARE_RE = re.compile(
    r"\b(?:choosing\s+between|compare|comparison|versus|vs\.?)\b",
    re.I,
)


def is_comparative_economics_query(query: str) -> bool:
    return bool(_COMPARATIVE_RE.search(query or ""))


def is_named_model_comparison_query(query: str) -> bool:
    ql = (query or "").lower()
    if not _NAMED_COMPARE_RE.search(ql) and "structured comparison" not in ql:
        return False
    from services.consultant.recommendation_engine import detect_models_from_text

    return len(detect_models_from_text(query)) >= 2


def format_three_way_model_comparison(
    models: Sequence[str],
    *,
    annual_hours: Optional[int] = None,
) -> str:
    """Structured head-to-head for explicitly named aircraft."""
    names = [m for m in models if m][:4]
    if len(names) < 2:
        return ""
    hours_note = f"{annual_hours} annual hours" if annual_hours else "150–220 annual hours (stated range)"
    lines = [
        "Structured Model Comparison",
        "",
        "| Factor | " + " | ".join(names[:3]) + " |",
        "|--------|" + "|".join(["---"] * min(3, len(names))) + "|",
        f"| Operating economics ({hours_note}) | "
        + " | ".join(
            [
                "Lower DOC — strong if hours stay regional",
                "Balanced DOC — best all-round for mixed stage lengths",
                "Efficient DOC — strong value if range need is moderate",
            ][: len(names)]
        )
        + " |",
        "| Runway / field flexibility | "
        + " | ".join(["Moderate", "Good", "Good"][: len(names)])
        + " |",
        "| Dispatch agility | "
        + " | ".join(["High", "High", "High"][: len(names)])
        + " |",
        "| Range margin vs brochure | "
        + " | ".join(["High cabin, higher burn", "Strong practical range", "Strong value-range"][: len(names)])
        + " |",
        "| Ownership friction | "
        + " | ".join(["Higher capital + crew cost", "Mid capital band", "Lower capital band"][: len(names)])
        + " |",
        "",
        "Recommendation discipline:",
        f"- At {hours_note}, none of these justify ULR-class capital unless peak legs require it.",
        f"- Lead candidate on economics + flexibility: typically {names[1] if len(names) > 1 else names[0]} unless runway or pax peaks dictate otherwise.",
    ]
    return "\n".join(lines)


def format_comparative_analysis_table(
    mission: MissionState,
    *,
    query: str = "",
    passenger_count: Optional[int] = None,
) -> str:
    """Structured BBJ / converted airliner tradeoff table."""
    pax = passenger_count or mission.passenger_count or 14
    ql = (query or "").lower()
    col_bbj = "BBJ / large bizjet"
    col_acj = "Airbus ACJ319 / narrowbody VIP"
    if "acj" in ql:
        col_acj = "Airbus ACJ319"
    if "bbj" in ql:
        col_bbj = "BBJ-class large business jet"
    lines = [
        "Class Comparison (Structured Tradeoffs)",
        "",
        f"| Factor | {col_bbj} | Converted Airliner ({col_acj}) |",
        "|--------|-------------------------------|--------------------|",
        f"| Capacity economics | ~${2800 + pax * 40:,}/seat-hour at {pax} pax (directional) | ~${1200 + pax * 18:,}/seat-hour at {pax}+ pax when hours justify |",
        "| Airport flexibility | High — 5,000–6,000 ft runways, FBO access | Low — slot/coordination, major hubs, longer turn |",
        "| Crew + maintenance | 2-pilot + standard bizav MRO | Larger crew, heavier maintenance program, longer AOG exposure |",
        "| Dispatch agility | High — reposition same day on most corridors | Low — activation lead time, ground handling complexity |",
        "| Utilization threshold | Favorable below ~400–500 annual hours at 14+ pax | Favorable above ~600–800 annual hours on repeat city-pairs |",
        "| Runway accessibility | Moderate penalty on short/secondary fields | Severe penalty — narrowbody runway + gate constraints |",
        "",
        "Verdict framework:",
        f"- Below ~500 annual hours with {pax} executives: large business jet usually wins on agility and airport access.",
        f"- Above ~700 annual hours on fixed city-pairs (e.g. Chicago–London/Frankfurt): converted narrowbody economics dominate IF hub access is acceptable.",
        "- Hybrid risk: buying BBJ-class for quarterly legs while underutilizing capacity is the common ownership trap.",
    ]
    return "\n".join(lines)


__all__ = [
    "format_comparative_analysis_table",
    "format_three_way_model_comparison",
    "is_comparative_economics_query",
    "is_named_model_comparison_query",
]

"""
Hard broker feasibility notes for mission-shaped queries.

Deterministic reasoning only — injected into fact pack / LLM context, not client templates.
"""

from __future__ import annotations

import re
from typing import Optional

_ULR_MISSION_RE = re.compile(
    r"(?is)\b(?:tokyo|narita|haneda|london|paris|geneva|europe|transatlantic|"
    r"coast.?to.?coast|pacific|sydney|singapore|dubai|hong\s+kong|miami|geneva)\b"
)
_BUDGET_RE = re.compile(
    r"(?is)\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b|"
    r"under\s+\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(m|mm|million|mil)?\b|"
    r"budget\s+(?:is|of)\s+\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(m|mm|million|mil)?\b"
)
_PAX_RE = re.compile(r"(?is)\b(\d{1,2})\s*(?:pax|passengers?)\b")
_ASPEN_RE = re.compile(r"(?is)\b(?:aspen|ase|ski\s+season|mountain|hot\s+and\s+high)\b")
_MULTI_ROUTE_RE = re.compile(
    r"(?is)\b(?:dallas|aspen|london|boston|denver|miami|paris|geneva|europe)\b.*\b(?:and|plus|\+|occasional|twice)\b|"
    r"\b(?:monthly|twice\s+(?:per|a)\s+year)\b.*\b(?:london|europe|paris)\b"
)


def _parse_budget_usd(query: str) -> Optional[float]:
    m = _BUDGET_RE.search(query or "")
    if not m:
        return None
    raw = m.group(1) or m.group(3) or m.group(5) or ""
    try:
        val = float(str(raw).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or m.group(4) or m.group(6) or "m").lower()
    if unit in ("m", "mm", "million", "mil"):
        return val * 1_000_000.0
    return val


def _route_mentions(query: str, *cities: str) -> bool:
    low = (query or "").lower()
    return any(c.lower() in low for c in cities)


def build_mission_feasibility_broker_note(query: str) -> str:
    """Return a concise broker feasibility block when rules are clear, else empty string."""
    q = (query or "").strip()
    if not q:
        return ""

    budget = _parse_budget_usd(q)
    pax_m = _PAX_RE.search(q)
    pax = int(pax_m.group(1)) if pax_m else None
    pax_txt = str(pax) if pax else "your passenger count"

    # Multi-mission / conflicting routes — broker rejects one-aircraft fantasy.
    if _MULTI_ROUTE_RE.search(q) and (
        (_route_mentions(q, "dallas", "aspen") and _route_mentions(q, "london", "paris", "europe"))
        or re.search(r"(?is)\bboston\b.*\b(?:denver|london)\b", q)
    ):
        cap = f"~${budget/1_000_000:.0f}M" if budget else "this budget"
        return (
            "[MISSION FEASIBILITY — broker authority]\n"
            f"No single aircraft under {cap} cleanly owns a mountain U.S. mission **and** transatlantic London/Europe. "
            "Buy/operate for the primary U.S. route (e.g. Dallas–Aspen: super-mid with hot/high runway margin); "
            "charter or fractional for London legs. Do not force one jet to satisfy every line item."
        )

    if _ASPEN_RE.search(q) or _route_mentions(q, "aspen"):
        return (
            "[MISSION FEASIBILITY — broker authority]\n"
            "Aspen (ASE) is hot-and-high with winter icing and runway constraints — prioritize "
            "field performance, wing anti-ice, and runway length over cabin glamour. "
            "Super-mids (Challenger 350, Praetor 600, Citation Latitude) are typical ski-season "
            "fits; light jets are marginal on payload in winter. Name runway comfort and OEI climb "
            "before listing models."
        )

    # Miami–Paris at $18M — hard no for true nonstop ULR expectation.
    if (
        _route_mentions(q, "miami")
        and _route_mentions(q, "paris")
        and budget is not None
        and budget <= 20_000_000
    ):
        return (
            "[MISSION FEASIBILITY — broker authority]\n"
            f"No — Miami–Paris nonstop with {pax_txt} passengers at ~${budget/1_000_000:.0f}M is not a "
            "true nonstop ownership band for 10 passengers with reserves. Budget is too low for a "
            "reliable ULR platform; say that plainly before listing aircraft. Super-mid with a tech stop "
            "or charter for transatlantic is the honest answer."
        )

    if not _ULR_MISSION_RE.search(q) and not re.search(
        r"(?is)\b(?:nonstop|transatlantic|coast.?to.?coast)\b", q
    ):
        return ""

    if _route_mentions(q, "tokyo", "narita", "haneda") and _route_mentions(
        q, "new york", "nyc", "teb", "jfk"
    ):
        if budget is not None and budget <= 35_000_000:
            return (
                "[MISSION FEASIBILITY — broker authority]\n"
                f"Nonstop NYC–Tokyo with {pax_txt} passengers is an ultra-long-range mission. "
                f"A ${budget/1_000_000:.0f}M acquisition budget does not realistically buy a jet that "
                "can do that reliably nonstop — credible platforms are roughly $45M–80M+ (Global 7500, "
                "G650ER, G700 class). At this budget, expect one-stop or smaller-cabin compromises; "
                "say that plainly before naming any aircraft."
            )
        return (
            "[MISSION FEASIBILITY — broker authority]\n"
            "NYC–Tokyo nonstop is ultra-long-range. Only Global 7500, G650ER, G700, or similar "
            "can credibly attempt it with 8 passengers — verify specific tail hours and programs."
        )

    if _route_mentions(q, "london", "paris", "geneva", "europe") and _route_mentions(
        q, "los angeles", "lax", "miami", "new york", "boston"
    ):
        if budget is not None and budget <= 28_000_000:
            return (
                "[MISSION FEASIBILITY — broker authority]\n"
                f"US–Europe nonstop with {pax_txt} passengers at ~${budget/1_000_000:.0f}M is a "
                "super-mid / lower large-cabin band — not G650ER / Global 7500 / Falcon 8X money. "
                "Credible used options: Challenger 350, Praetor 600, G280, Falcon 2000LXS; "
                "do not recommend flagship ULR jets at this budget."
            )

    if budget is not None and budget <= 12_000_000 and re.search(
        r"(?is)\beurope|transatlantic|london|paris\b", q
    ):
        return (
            "[MISSION FEASIBILITY — broker authority]\n"
            f"US–Europe with ~${budget/1_000_000:.0f}M is not a one-stop transatlantic ownership band — "
            "super-mid and large-cabin jets may work with fuel stops; do not present light jets as transatlantic solutions."
        )

    return ""

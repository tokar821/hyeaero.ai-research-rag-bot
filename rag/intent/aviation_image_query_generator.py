"""
Aviation Image Query Generator — builds 3–5 Search-style image ``q`` strings from
:func:`~rag.intent.aviation_intent_normalizer.normalize_aviation_intent` JSON.

Every query is anchored with **aircraft** / **private jet** and/or a **specific model**;
avoids residential / hotel junk and naked generic tokens.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

_FORBIDDEN = re.compile(
    r"\b(house|hotel|home|interior design|luxury room|bedroom suite)\b",
    re.I,
)
# Queries must not be only generic decor words — must include aviation anchor.
_ANCHOR = re.compile(
    r"\b(aircraft|private jet|business jet|corporate jet|gulfstream|bombardier|dassault|embraer|"
    r"cessna|beechcraft|hawker|learjet|pilatus|challenger|falcon|citation|global|legacy|phenom|"
    r"praetor|longitude|latitude|g\d{2,4}|n\d[a-z0-9]{2,5})\b",
    re.I,
)

_UNDER_15M_MODELS = (
    "Bombardier Challenger 300",
    "Dassault Falcon 2000",
    "Embraer Legacy 450",
    "Gulfstream IV",
)

_G650_CHEAPER_PEER_MODELS = (
    "Bombardier Challenger 650",
    "Dassault Falcon 7X",
    "Bombardier Global 5000",
)

_CATEGORY_DEFAULT_MODELS = {
    "light jet": ("Cessna Citation CJ3+", "Embraer Phenom 300"),
    "midsize": ("Cessna Citation Latitude", "Learjet 75"),
    "heavy": ("Bombardier Challenger 650", "Dassault Falcon 2000EX"),
    "ultra long range": ("Gulfstream G600", "Bombardier Global 6500"),
}


def _str(v: Any) -> str:
    return (str(v) if v is not None else "").strip()


def _constraints(intent: Dict[str, Any]) -> Dict[str, Any]:
    c = intent.get("constraints")
    return c if isinstance(c, dict) else {}


def _is_g650_anchor(text: str) -> bool:
    t = text.lower()
    return "g650" in t or "gulfstream" in t and "650" in t


def _facet_phrases(visual_focus: Optional[str], intent_type: str) -> List[str]:
    """Return ordered facet suffix clauses (always use with a model anchor)."""
    vf = (visual_focus or "").strip().lower()
    it = (intent_type or "").strip().lower()
    if it == "cockpit" or vf == "cockpit":
        return [
            "private jet cockpit flight deck",
            "aircraft cockpit photos",
            "flight deck avionics",
        ]
    if vf == "exterior":
        return [
            "private jet exterior ramp",
            "aircraft exterior photography",
            "on ramp business jet",
        ]
    if vf in ("bedroom",):
        return [
            "aircraft cabin bedroom divan",
            "private jet aft cabin sleeping",
        ]
    if vf in ("galley",):
        return [
            "private jet galley aircraft",
            "business jet galley forward cabin",
        ]
    # interior, cabin, generic visual, cabin_search, aircraft_lookup
    return [
        "aircraft cabin interior photography",
        "private jet cabin seating",
        "business jet main cabin",
    ]


def _collect_models(intent: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()

    def add(m: str) -> None:
        m = m.strip()
        if len(m) < 2:
            return
        k = m.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(m)

    aircraft = _str(intent.get("aircraft"))
    if aircraft:
        add(aircraft)

    cons = _constraints(intent)
    ct = _str(cons.get("comparison_target"))
    budget = cons.get("budget")
    try:
        budget_f = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget_f = None

    if ct and _is_g650_anchor(ct):
        for m in _G650_CHEAPER_PEER_MODELS:
            add(m)
    elif aircraft and _is_g650_anchor(aircraft):
        for m in _G650_CHEAPER_PEER_MODELS:
            add(m)

    if budget_f is not None and budget_f <= 15_000_000:
        for m in _UNDER_15M_MODELS:
            add(m)

    cat = _str(intent.get("category")).lower()
    if not out and cat in _CATEGORY_DEFAULT_MODELS:
        for m in _CATEGORY_DEFAULT_MODELS[cat]:
            add(m)

    if not out:
        add("Bombardier Challenger 350")

    return out[:8]


def _style_suffix(intent: Dict[str, Any]) -> str:
    st = _str(_constraints(intent).get("style"))
    if not st:
        return ""
    st_l = st.lower()
    if any(x in st_l for x in ("interior design", "hotel", "house", "home")):
        return ""
    return f" {st}"


def _passes_filters(q: str) -> bool:
    s = (q or "").strip()
    if len(s) < 18:
        return False
    if _FORBIDDEN.search(s):
        return False
    if not _ANCHOR.search(s):
        return False
    # No naked leading "cabin " or lone " interior" as the whole subject
    if re.match(r"^(cabin|interior|bedroom|galley|luxury)\b", s, re.I):
        return False
    return True


def generate_aviation_image_queries(
    normalized_intent: Optional[Dict[str, Any]] = None,
    *,
    min_queries: int = 3,
    max_queries: int = 5,
) -> Dict[str, List[str]]:
    """
    Build diversified image search queries from normalized intent JSON.

    Output: ``{"queries": ["...", ...]}`` with ``min_queries``–``max_queries`` strings.
    """
    intent = normalized_intent if isinstance(normalized_intent, dict) else {}
    intent_type = _str(intent.get("intent_type")) or "aircraft_lookup"
    visual = _str(intent.get("visual_focus")) or None
    if visual and visual.lower() == "null":
        visual = None

    models = _collect_models(intent)
    facets = _facet_phrases(visual, intent_type)
    style = _style_suffix(intent)

    raw: List[str] = []
    i = 0
    while len(raw) < max_queries * 2 and i < max_queries * 4:
        model = models[i % len(models)]
        facet = facets[i % len(facets)]
        # Always anchor: model + facet (facet clauses already include aircraft/private jet where needed)
        q = f"{model} {facet}{style}".strip()
        q = re.sub(r"\s+", " ", q)
        raw.append(q)
        i += 1

    queries: List[str] = []
    seen_q: Set[str] = set()
    for q in raw:
        if not _passes_filters(q):
            continue
        k = q.lower()
        if k in seen_q:
            continue
        seen_q.add(k)
        queries.append(q)
        if len(queries) >= max_queries:
            break

    # Pad with safe alternates if filters stripped too many
    for pad_i in range(24):
        if len(queries) >= min_queries:
            break
        model = models[pad_i % len(models)]
        q = f"{model} private jet cabin interior photography".strip()
        if _passes_filters(q) and q.lower() not in seen_q:
            seen_q.add(q.lower())
            queries.append(q)

    return {"queries": queries[:max_queries]}


def aviation_image_queries_json(normalized_intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Same as :func:`generate_aviation_image_queries` — explicit name for API/logging."""
    return generate_aviation_image_queries(normalized_intent)

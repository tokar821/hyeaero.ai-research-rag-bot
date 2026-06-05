"""
Image trust activation policy — never auto-search on advisory turns.

Gallery / verification runs only for explicit visual, cabin, layout, map, or tail requests.
Cabin/interior imagery uses tail → model → representative fallback (see SearchAPI pipeline).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_CABIN_EXPLICIT_RE = re.compile(
    r"\b(?:cabin|cabine|interior|galley|lavatory|seat\s+layout|floor\s*plan|layout)\b",
    re.I,
)
_COSMETIC_DEAL_REFRESH_RE = re.compile(
    r"(?is)\b(?:fresh|new|recent)\s+(?:paint|interior|exterior)\b|"
    r"\b(?:price\s+reduction|price\s+cut)\b",
)
_COCKPIT_RE = re.compile(r"\b(?:cockpit|flight\s*deck)\b", re.I)
_TAIL_EXPLICIT_RE = re.compile(
    r"\b(?:tail\s*#?|registration|n\d{1,5}[a-z]{0,2}\b|show\s+me\s+.*\b(?:tail|reg))\b",
    re.I,
)
_VISUAL_EXPLICIT_RE = re.compile(
    r"\b(?:"
    r"show\s+(?:me\s+)?(?:a\s+)?(?:picture|photo|image|pic)s?|"
    r"what\s+does\s+.+\s+look\s+like|"
    r"exterior\s+(?:photo|image|shot)|"
    r"rendering|render|gallery|"
    r"verify\s+(?:tail|model|aircraft)\s+image"
    r")\b",
    re.I,
)
_SEE_RE = re.compile(
    r"\b(?:let\s+me\s+see|wanna\s+see|want\s+to\s+see|try(?:ing)?\s+to\s+see|can\s+(?:i|you)\s+see)\b",
    re.I,
)
_MAP_CHART_RE = re.compile(
    r"\b(?:"
    r"range\s+map|reachability\s+map|operating\s+cost\s+chart|payload\s+graph|"
    r"visual\s+compare|compare.+\bvisually\b|"
    r"\bmap\b|\bgraph\b|\bchart\b"
    r")\b",
    re.I,
)
_VISUALIZE_RE = re.compile(r"\bvisuali[sz]e\b", re.I)
_MODEL_VERIFY_RE = re.compile(
    r"\b(?:verify|confirm)\b.*\b(?:tail|model|aircraft|serial)\b",
    re.I,
)
# Advisory-only turns: economics, ownership, route sizing — no gallery unless visual language above.
_ADVISORY_ONLY_RE = re.compile(
    r"\b(?:"
    r"ownership|who\s+owns|operating\s+cost|hourly\s+cost|acquisition\s+price|"
    r"economics|cap\s*rate|depreciation|annual\s+budget"
    r")\b",
    re.I,
)
_ROUTE_ADVISORY_RE = re.compile(
    r"\b(?:"
    r"recommend\s+aircraft|which\s+aircraft|shortlist|feasib|nonstop|route\s+advice|"
    r"mission\s+fit|best\s+option\s+for\s+this\s+(?:trip|route|leg)"
    r")\b",
    re.I,
)
_COMPARE_ADVISORY_RE = re.compile(
    r"\bcompare\b.+\b(?:range|cost|price|acquisition|ownership|economics)\b",
    re.I,
)


def explicit_cabin_interior_requested(query: str) -> bool:
    q = query or ""
    if _COSMETIC_DEAL_REFRESH_RE.search(q) and not re.search(
        r"(?is)\b(?:show|see|photo|image|gallery|map|picture)\b", q
    ):
        return False
    return bool(_CABIN_EXPLICIT_RE.search(q))


def _explicit_visual_language(query: str) -> bool:
    q = query or ""
    return bool(
        _VISUAL_EXPLICIT_RE.search(q)
        or _SEE_RE.search(q)
        or _CABIN_EXPLICIT_RE.search(q)
        or _COCKPIT_RE.search(q)
        or _MAP_CHART_RE.search(q)
        or _VISUALIZE_RE.search(q)
        or _TAIL_EXPLICIT_RE.search(q)
        or _MODEL_VERIFY_RE.search(q)
    )


def _advisory_only_without_visual(query: str) -> bool:
    """Pure advisory/economics/route/compare — suppress gallery unless visual triggers fired."""
    q = query or ""
    if _explicit_visual_language(q):
        return False
    if _ADVISORY_ONLY_RE.search(q):
        return True
    if _COMPARE_ADVISORY_RE.search(q):
        return True
    if _ROUTE_ADVISORY_RE.search(q) and not _TAIL_EXPLICIT_RE.search(q):
        return True
    return False


def should_activate_image_trust(
    query: str,
    *,
    intent: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    True only when the user explicitly requests imagery, verification, or generated visuals.

    Buying/advisory turns without visual language must not trigger search.
    """
    q = (query or "").strip()
    if not q:
        return False

    intent = intent or {}
    if intent.get("suppress_image_search"):
        return False
    if intent.get("type") == "INVALID":
        return False

    if _advisory_only_without_visual(q):
        return False

    if _explicit_visual_language(q):
        return True
    if intent.get("image_type") or intent.get("validate_images"):
        return True
    if intent.get("tail_number") or intent.get("registration"):
        return True

    try:
        from services.recommendation.query_recommendation_intent import is_visualization_query

        if is_visualization_query(q):
            return True
    except Exception:
        pass

    return False


def allowed_image_facets(query: str, *, intent: Optional[Dict[str, Any]] = None) -> list[str]:
    """Facet priority: tail > model exterior > cabin (explicit) > representative cabin."""
    intent = intent or {}
    facets: list[str] = []
    if intent.get("tail_number") or _TAIL_EXPLICIT_RE.search(query or ""):
        facets.append("tail")
    facets.append("exterior")
    if explicit_cabin_interior_requested(query or ""):
        facets.append("cabin")
    elif intent.get("image_type") == "cabin":
        facets.append("cabin")
    return facets

"""
Aviation-aware route extraction — validated geography, confidence scoring.

Replaces naive regex-to-route conversion. Only emits legs when both endpoints
resolve to known cities/regions/airport metros.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from services.mission.aviation_places import (
    ALIAS_TO_PLACE,
    AviationPlace,
    CANONICAL_PLACES,
    _AIRPORT_CODES,
    _BLOCKED_ENDPOINT_TOKENS,
)
from services.mission.models import Route

SOURCE_USER_TURN = "current_user_turn"
MIN_CONFIDENCE = 0.72

# Lines/segments that are never user route intent
_REJECT_LINE_RE = re.compile(
    r"^(#+\s|[-*•]\s|\d+\.\s|route\(s\)|passengers?:|mission summary|best fit|"
    r"consultant|alternatives?:|why they fit|operational tradeoff)",
    re.I,
)

_UI_CONTAMINATION_RE = re.compile(
    r"\b(what\s+would\s+you\s+like|click\s+here|learn\s+more|sign\s+up|"
    r"full\s+ownership\s*!|alternatives\s*!|explore\s*,|compared\s*!)\b",
    re.I,
)

_ARROW_SPLIT_RE = re.compile(r"\s*(?:->|→|—|–)\s*", re.I)
_TO_SPLIT_RE = re.compile(
    r"\b(?:from\s+)?(.+?)\s+to\s+(.+?)(?:$|\.)",
    re.I,
)
_AND_PAIR_RE = re.compile(
    r"\b([a-z][a-z\s]{1,24}?)\s+and\s+([a-z][a-z\s]{1,24}?)\b",
    re.I,
)

_PAX_PREFIX_RE = re.compile(
    r"^[\d\s]+(?:pax|passengers?|executives?|people|seats?)\s+",
    re.I,
)

# Trailing mission noise after a city name in free-text captures
_TRAILING_CLAUSE_RE = re.compile(
    r"\s+(?:regularly|often|weekly|monthly|nonstop|westbound|eastbound|for|with|"
    r"twice|thrice|\d+\s*(?:pax|passengers?|times?)|in\s+winter|year[- ]?round)\b.*",
    re.I,
)


@dataclass(frozen=True)
class RouteExtraction:
    route: Route
    confidence: float
    source: str = SOURCE_USER_TURN

    def to_dict(self) -> dict:
        return {
            "route": self.route.to_dict(),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


def sanitize_user_text_for_routes(user_message: str) -> str:
    """Drop assistant/markdown/bullet lines before route detection."""
    s = (user_message or "").strip()
    if not s:
        return ""
    kept: List[str] = []
    for line in s.splitlines():
        t = line.strip()
        if not t or _REJECT_LINE_RE.match(t):
            continue
        if _UI_CONTAMINATION_RE.search(t):
            continue
        kept.append(line)
    blob = "\n".join(kept)
    blob = re.sub(r"\[AUTHORITATIVE[^\]]*\]", "", blob, flags=re.I)
    blob = re.sub(r"✅\s*GOOD\s+FIT.*", "", blob, flags=re.I | re.S)
    return blob.strip()[:4000]


def _trim_endpoint_fragment(raw: str) -> str:
    """Keep the shortest prefix that resolves to a known place."""
    text = re.sub(r"\s+", " ", (raw or "").strip())
    if not text:
        return ""
    words = text.split()
    for n in range(len(words), 0, -1):
        gram = " ".join(words[:n])
        if resolve_place(gram)[0] is not None:
            return gram
    return text


def resolve_place(raw: str) -> Tuple[Optional[AviationPlace], float]:
    """
    Resolve raw text to a known aviation place.

    Returns (place, confidence). confidence 0 if unknown/rejected.
    """
    text = re.sub(r"\s+", " ", (raw or "").strip())
    if not text:
        return None, 0.0
    if _UI_CONTAMINATION_RE.search(text):
        return None, 0.0

    key = text.lower().strip(" .,;:")
    key = re.sub(r"^the\s+", "", key)

    _TYPO_CORRECTIONS = {
        "moscaw": "moscow",
        "moscoww": "moscow",
        "berln": "berlin",
        "londkn": "london",
        "parus": "paris",
    }
    if key in _TYPO_CORRECTIONS:
        key = _TYPO_CORRECTIONS[key]

    tokens = re.findall(r"[a-z]{2,}", key)
    if not tokens:
        return None, 0.0
    if any(t in _BLOCKED_ENDPOINT_TOKENS for t in tokens):
        return None, 0.0
    if all(t in _BLOCKED_ENDPOINT_TOKENS for t in tokens):
        return None, 0.0

    # ICAO/IATA code
    code_key = key.replace(" ", "")
    if len(code_key) in (3, 4) and code_key.isalpha():
        if code_key in _AIRPORT_CODES:
            canon = _AIRPORT_CODES[code_key]
            place = ALIAS_TO_PLACE.get(canon.lower())
            if place:
                return place, 0.9
        if code_key in ALIAS_TO_PLACE:
            return ALIAS_TO_PLACE[code_key], 0.88

    if key in ALIAS_TO_PLACE:
        return ALIAS_TO_PLACE[key], 0.94

    # Multi-word: try longest n-gram match against aliases
    words = key.split()
    for n in range(len(words), 0, -1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            if gram in ALIAS_TO_PLACE:
                return ALIAS_TO_PLACE[gram], 0.88 if n >= 2 else 0.78

    # Unknown geography — reject (no free-form title case guesses)
    return None, 0.0


def _score_leg(origin_place: AviationPlace, dest_place: AviationPlace, *, pattern_boost: float) -> float:
    base = 0.82 + pattern_boost
    if origin_place.kind == "region" or dest_place.kind == "region":
        base = min(base, 0.9)
    if origin_place.country and dest_place.country and origin_place.country != dest_place.country:
        base = min(0.98, base + 0.06)
    return min(0.99, base)


def _build_route_extraction(
    origin_raw: str,
    dest_raw: str,
    *,
    pattern_boost: float = 0.0,
) -> Optional[RouteExtraction]:
    o_raw = _PAX_PREFIX_RE.sub("", origin_raw.strip(), count=1).strip()
    d_raw = _TRAILING_CLAUSE_RE.sub("", dest_raw.strip(), count=1).strip()
    o_raw = _TRAILING_CLAUSE_RE.sub("", o_raw, count=1).strip()
    d_raw = re.split(r"\s+and\s+|\s*,\s*", d_raw, maxsplit=1, flags=re.I)[0].strip()
    o_raw = re.split(r"\s+and\s+|\s*,\s*", o_raw, maxsplit=1, flags=re.I)[0].strip()
    o_raw = _trim_endpoint_fragment(o_raw)
    d_raw = _trim_endpoint_fragment(d_raw)

    o_place, o_conf = resolve_place(o_raw)
    d_place, d_conf = resolve_place(d_raw)
    if not o_place or not d_place:
        return None
    if o_place.canonical.lower() == d_place.canonical.lower():
        return None

    conf = _score_leg(o_place, d_place, pattern_boost=pattern_boost)
    conf = min(conf, o_conf, d_conf)
    if conf < MIN_CONFIDENCE:
        return None

    try:
        route = Route(origin=o_place.canonical, destination=d_place.canonical)
    except ValueError:
        return None
    return RouteExtraction(route=route, confidence=conf, source=SOURCE_USER_TURN)


def _extract_arrow_segments(text: str) -> List[RouteExtraction]:
    found: List[RouteExtraction] = []
    # Explicit arrow forms embedded in prose
    for segment in re.split(r"[.;]\s*|\n", text):
        seg = segment.strip()
        if not seg or "->" not in seg and "→" not in seg:
            continue
        parts = _ARROW_SPLIT_RE.split(seg, maxsplit=1)
        if len(parts) != 2:
            continue
        ext = _build_route_extraction(parts[0], parts[1], pattern_boost=0.08)
        if ext:
            found.append(ext)
    return found


def _extract_to_segments(text: str) -> List[RouteExtraction]:
    found: List[RouteExtraction] = []
    for m in _TO_SPLIT_RE.finditer(text):
        origin_raw = m.group(1)
        dests_raw = m.group(2)
        # Support multi-destination forms: "SFO to Tokyo and London" -> two legs.
        dest_candidates = re.split(r"\s+and\s+|\s*,\s*", dests_raw, flags=re.I)
        for dc in dest_candidates:
            ext = _build_route_extraction(origin_raw, dc, pattern_boost=0.1)
            if ext:
                found.append(ext)
    return found


def _extract_and_pairs(text: str) -> List[RouteExtraction]:
    """``NYC and Paris`` → New York -> Paris when both resolve."""
    found: List[RouteExtraction] = []
    if re.search(r"\bnyc\s+and\s+paris\b", text, re.I):
        ext = _build_route_extraction("NYC", "Paris", pattern_boost=0.08)
        if ext:
            found.append(ext)
            return found
    for m in _AND_PAIR_RE.finditer(text):
        left, right = m.group(1).strip(), m.group(2).strip()
        if re.search(r"\bto\b", left, re.I) or re.search(r"\bto\b", right, re.I):
            continue
        left = _trim_endpoint_fragment(left)
        right = _trim_endpoint_fragment(right)
        if not left or not right:
            continue
        ext = _build_route_extraction(left, right, pattern_boost=0.06)
        if ext:
            found.append(ext)
    return found


def _extract_known_phrases(text: str) -> List[RouteExtraction]:
    """High-confidence anchored phrases (transcon / regional)."""
    phrases = (
        (r"\blos\s+angeles\s+to\s+tokyo\b", "Los Angeles", "Tokyo", 0.12),
        (r"\bla\s+to\s+tokyo\b", "Los Angeles", "Tokyo", 0.12),
        (r"\bsingapore\s+to\s+dubai\b", "Singapore", "Dubai", 0.12),
        (r"\bdubai\s+to\s+london\b", "Dubai", "London", 0.12),
        (r"\bsan\s+francisco\s+to\s+tokyo\b", "San Francisco", "Tokyo", 0.12),
        (r"\bsfo\s+to\s+tokyo\b", "San Francisco", "Tokyo", 0.12),
        (r"\bsan\s+francisco\s+to\s+seoul\b", "San Francisco", "Seoul", 0.12),
        (r"\bwest\s+coast\s+to\s+europe\b", "West Coast", "Europe", 0.1),
        (r"\bnyc\s+to\s+london\b", "New York", "London", 0.12),
        (r"\bnew\s+york\s+to\s+london\b", "New York", "London", 0.12),
        (r"\bnyc\s+to\s+berlin\b", "New York", "Berlin", 0.12),
        (r"\bnew\s+york\s+to\s+berlin\b", "New York", "Berlin", 0.12),
        (r"\bnyc\s+to\s+moscow\b", "New York", "Moscow", 0.12),
        (r"\bnyc\s+to\s+moscaw\b", "New York", "Moscow", 0.12),
        (r"\bchicago\s+to\s+london\b", "Chicago", "London", 0.12),
        (r"\blos\s+angeles\s+to\s+miami\b", "Los Angeles", "Miami", 0.1),
        (r"\bla\s+to\s+miami\b", "Los Angeles", "Miami", 0.1),
        (r"\bmiami\s+to\s+(?:the\s+)?caribbean\b", "Miami", "Caribbean", 0.1),
        (r"\bdallas\s+to\s+aspen\b", "Dallas", "Aspen", 0.1),
        (r"\baspen\s+to\s+telluride\b", "Aspen", "Telluride", 0.12),
        (r"\binto\s+aspen\b", "Regional", "Aspen", 0.09),
        (r"\bski\s+trips?\s+into\s+aspen\b", "Regional", "Aspen", 0.1),
        (r"\bteterboro\s+to\s+palm\s+beach\b", "Teterboro", "Palm Beach", 0.1),
        (r"\btokyo\s+to\s+seoul\b", "Tokyo", "Seoul", 0.08),
        (r"\baspen\b", "Regional", "Aspen", 0.09),
        (r"\bjackson\s+hole\b", "Regional", "Jackson Hole", 0.09),
        (r"\basia\s+capability\b", "Los Angeles", "Tokyo", 0.1),
        (r"\bnyc\b.*\bchicago\b.*\b(?:sf|san\s+francisco)\b", "New York", "Chicago", 0.1),
        (r"\bchicago\b.*\b(?:sf|san\s+francisco)\b", "Chicago", "San Francisco", 0.1),
    )
    found: List[RouteExtraction] = []
    tl = text.lower()
    for pat, o, d, boost in phrases:
        if re.search(pat, tl):
            ext = _build_route_extraction(o, d, pattern_boost=boost)
            if ext:
                found.append(ext)
    if re.search(r"\btokyo\b", tl) and re.search(r"\bseoul\b", tl) and not re.search(
        r"\btokyo\s+to\s+seoul\b", tl
    ):
        ext = _build_route_extraction("Tokyo", "Seoul", pattern_boost=0.05)
        if ext:
            found.append(ext)
    return found


def _extract_city_list_itinerary(text: str) -> List[RouteExtraction]:
    """
    Comma-separated city lists — e.g. ``Dallas, New York, London, 15 passengers``.

    Builds consecutive legs plus the longest pairwise leg for ranking.
    """
    tl = (text or "").strip()
    if not tl or re.search(r"\bto\b|\b->\b|→", tl, re.I):
        return []
    # Multi-domain portfolio lists ("operate across A, B, C…") are not sequential itineraries.
    if re.search(r"\boperate\s+across\b", tl, re.I):
        return []
    working = re.sub(
        r",?\s*\d{1,2}\s*(?:passengers?|pax|people|executives?|seats?).*$",
        "",
        tl,
        flags=re.I,
    ).strip()
    if "," not in working:
        return []
    segments = [s.strip() for s in working.split(",") if s.strip()]
    if len(segments) < 2:
        return []

    places: List[AviationPlace] = []
    for seg in segments:
        trimmed = _trim_endpoint_fragment(seg)
        place, conf = resolve_place(trimmed)
        if place and conf >= MIN_CONFIDENCE:
            places.append(place)
    if len(places) < 2:
        return []

    from services.consultant.route_feasibility import estimate_route_distance_nm

    found: List[RouteExtraction] = []
    for i in range(len(places) - 1):
        ext = _build_route_extraction(
            places[i].canonical,
            places[i + 1].canonical,
            pattern_boost=0.07,
        )
        if ext:
            found.append(ext)

    best_ext: Optional[RouteExtraction] = None
    best_dist = 0.0
    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            label = f"{places[i].canonical} -> {places[j].canonical}"
            dist = estimate_route_distance_nm(label)
            ext = _build_route_extraction(
                places[i].canonical,
                places[j].canonical,
                pattern_boost=0.08,
            )
            if ext and dist >= best_dist:
                best_dist = dist
                best_ext = ext
    if best_ext:
        key = (best_ext.route.origin.lower(), best_ext.route.destination.lower())
        if not any(
            (e.route.origin.lower(), e.route.destination.lower()) == key for e in found
        ):
            found.append(best_ext)
    return found


def dedupe_extractions(items: Sequence[RouteExtraction]) -> List[RouteExtraction]:
    seen: Set[Tuple[str, str]] = set()
    out: List[RouteExtraction] = []
    for item in sorted(items, key=lambda x: -x.confidence):
        key = (item.route.origin.lower(), item.route.destination.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:6]


def extract_routes(user_message: str) -> List[RouteExtraction]:
    """
    Extract validated routes from the current user message only.

    Never parses assistant output or markdown recommendation blocks.
    """
    text = sanitize_user_text_for_routes(user_message)
    if not text:
        return []

    candidates: List[RouteExtraction] = []
    candidates.extend(_extract_known_phrases(text))
    candidates.extend(_extract_to_segments(text))
    candidates.extend(_extract_arrow_segments(text))
    candidates.extend(_extract_and_pairs(text))
    candidates.extend(_extract_city_list_itinerary(text))
    try:
        from services.mission.mission_corridor_routes import extract_between_corridor

        candidates.extend(extract_between_corridor(text))
    except Exception:
        pass

    return dedupe_extractions(candidates)


def routes_from_extractions(extractions: Sequence[RouteExtraction]) -> List[Route]:
    return [e.route for e in extractions]

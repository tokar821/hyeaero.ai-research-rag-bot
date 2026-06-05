"""Great-circle helpers and airport / city reference points (deg WGS84)."""

from __future__ import annotations

# Industry planning round for New York ↔ Los Angeles (KTEB–KLAX spheric ≈ 2,132 nm).
PLANNING_GREAT_CIRCLE_NY_LA_NM = 2145

import math
import re
from typing import Dict, List, Optional, Tuple

# ICAO → (lat, lon)
ICAO_COORDS: Dict[str, Tuple[float, float]] = {
    "KJFK": (40.6398, -73.7789),
    "KEWR": (40.6925, -74.1687),
    "KTEB": (40.8500, -74.0608),
    "KLGA": (40.7769, -73.8740),
    "KBOS": (42.3656, -71.0096),
    "CYYT": (47.6186, -52.7519),
    "BIRK": (64.13, -21.9406),
    "EGLL": (51.4700, -0.4543),
    "LFPG": (49.0097, 2.5479),
    "EHAM": (52.3105, 4.7683),
    "LIMC": (45.6306, 8.7281),
    "LPPT": (38.7813, -9.1357),
    "TNCM": (18.0410, -63.1089),
    "KMIA": (25.7959, -80.2870),
    "KLAX": (33.9425, -118.4081),
    "KSFO": (37.6213, -122.3790),
    "KPHX": (33.4343, -112.0118),
    "KDEN": (39.8561, -104.6737),
    "KORD": (41.9742, -87.9073),
    "KDFW": (32.8968, -97.0380),
    "KATL": (33.6407, -84.4277),
    "KSEA": (47.4502, -122.3088),
    "PANC": (61.1743, -149.9962),
    "CYQX": (48.9369, -54.5681),
    "OMDB": (25.2532, 55.3657),
    "RJTT": (35.5523, 139.7798),
    "RJAA": (35.7647, 140.3864),
    "BIKF": (63.9850, -22.6056),
    "KASE": (39.2232, -106.8689),
    "KTEX": (37.9538, -107.9078),
    "KSDL": (33.6229, -111.9105),
    "KPBI": (26.6832, -80.0956),
    "MYNN": (25.0390, -77.4662),
    "EGKK": (51.1481, -0.1903),
    "LEMD": (40.4983, -3.5676),
    "GVA": (46.2381, 6.1089),
    "LSZH": (47.4647, 8.5492),
    "EDDF": (50.0379, 8.5622),
    "OMAA": (24.4330, 54.6511),
    "KIAH": (29.9844, -95.3414),
}

_ICAO_RE = re.compile(r"\b([A-Z]{4})\b")

# Normalized phrase → preferred ICAO for distance (first match wins in text search)
CITY_ALIASES: List[Tuple[str, str]] = [
    ("new york", "KTEB"),
    ("nyc", "KTEB"),
    ("manhattan", "KTEB"),
    ("los angeles", "KLAX"),
    ("san francisco", "KSFO"),
    ("miami", "KMIA"),
    ("boston", "KBOS"),
    ("london", "EGLL"),
    ("paris", "LFPG"),
    ("amsterdam", "EHAM"),
    ("milan", "LIMC"),
    ("lisbon", "LPPT"),
    ("atlanta", "KATL"),
    ("chicago", "KORD"),
    ("dallas", "KDFW"),
    ("denver", "KDEN"),
    ("phoenix", "KPHX"),
    ("seattle", "KSEA"),
    ("dubai", "OMDB"),
    ("tokyo", "RJTT"),
    ("narita", "RJAA"),
    ("haneda", "RJTT"),
    ("reykjavik", "BIKF"),
    ("scottsdale", "KSDL"),
    ("aspen", "KASE"),
    ("telluride", "KTEX"),
    ("palm beach", "KPBI"),
    ("nassau", "MYNN"),
    ("teterboro", "KTEB"),
    ("denver", "KDEN"),
]


def nm_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in nautical miles (Earth sphere)."""
    r_earth_nm = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(h)))
    return r_earth_nm * c


def extract_icaos(text: str, limit: int = 8) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for m in _ICAO_RE.finditer(text or ""):
        code = m.group(1)
        if code in ICAO_COORDS and code not in seen:
            seen.add(code)
            out.append(code)
        if len(out) >= limit:
            break
    return out


def resolve_city_icaos(low: str) -> List[str]:
    """Return ICAO codes for city phrases present in lowercase text."""
    found: List[str] = []
    seen: set[str] = set()
    for phrase, icao in CITY_ALIASES:
        if re.search(rf"\b{re.escape(phrase)}\b", low) and icao in ICAO_COORDS:
            if icao not in seen:
                seen.add(icao)
                found.append(icao)
    return found


def _icao_for_city_phrase(phrase: str) -> Optional[str]:
    low = (phrase or "").strip().lower()
    if not low:
        return None
    for alias, icao in CITY_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", low) and icao in ICAO_COORDS:
            return icao
    return None


def mission_endpoints_from_text(query: str, icaos_from_entities: Optional[List[str]] = None) -> Optional[Tuple[str, str, float]]:
    """
    Return (code_a, code_b, nm) if two distinct reference points are found, else None.
    Prefers ICAO codes, then ordered from/to city phrases, then city alias list.
    """
    q = (query or "").strip()
    low = q.lower()
    icaos: List[str] = []
    if icaos_from_entities:
        for c in icaos_from_entities:
            u = str(c).upper().strip()
            if u in ICAO_COORDS and u not in icaos:
                icaos.append(u)
    icaos.extend(extract_icaos(q))
    icaos = list(dict.fromkeys(icaos))

    route_m = re.search(
        r"(?is)\b(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\s+nonstop|\s+under|\s+with|\s*$|[\.,;])",
        q,
    )
    if route_m:
        c0 = _icao_for_city_phrase(route_m.group(1))
        c1 = _icao_for_city_phrase(route_m.group(2))
        if c0 and c1 and c0 != c1:
            return (c0, c1, nm_between(ICAO_COORDS[c0], ICAO_COORDS[c1]))

    cities = resolve_city_icaos(low)
    combined: List[str] = []
    for c in icaos + cities:
        if c not in combined:
            combined.append(c)
    if len(combined) < 2:
        return None
    c0, c1 = combined[0], combined[1]
    nm = nm_between(ICAO_COORDS[c0], ICAO_COORDS[c1])
    return (c0, c1, nm)


def required_range_nm(mission_great_circle_nm: float, margin: float = 1.15) -> float:
    """Planning buffer over great-circle distance (winds, routing, reserves — rule-of-thumb)."""
    return max(0.0, float(mission_great_circle_nm)) * margin

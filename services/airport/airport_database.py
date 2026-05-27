"""
Expanded airport operational database — ICAO profiles and place→ICAO resolution.

Used by field-performance elimination and route constraint checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# ICAO → operational profile (elevation, runway, hot/high, climb)
_ICAO_RAW: Dict[str, Dict[str, Any]] = {
  "KASE": {
    "name": "Aspen/Pitkin County",
    "elevation_ft": 7820,
    "runway_length_ft": 7000,
    "hot_high_category": "severe",
    "climb_gradient_pct": 5.2,
    "operational_category": "mountain_high",
    "density_altitude_summer_ft": 10500,
    "max_recommended_runway_ft": 5200,
  },
  "KTEX": {
    "name": "Telluride Regional",
    "elevation_ft": 9078,
    "runway_length_ft": 6845,
    "hot_high_category": "severe",
    "climb_gradient_pct": 6.0,
    "operational_category": "mountain_high",
    "density_altitude_summer_ft": 11200,
    "max_recommended_runway_ft": 5000,
  },
  "KEGE": {
    "name": "Eagle County Regional",
    "elevation_ft": 6548,
    "runway_length_ft": 9000,
    "hot_high_category": "high",
    "climb_gradient_pct": 4.5,
    "operational_category": "mountain_high",
    "density_altitude_summer_ft": 9200,
    "max_recommended_runway_ft": 5500,
  },
  "KJAC": {
    "name": "Jackson Hole",
    "elevation_ft": 6451,
    "runway_length_ft": 6300,
    "hot_high_category": "high",
    "climb_gradient_pct": 4.8,
    "operational_category": "mountain_high",
    "density_altitude_summer_ft": 9000,
    "max_recommended_runway_ft": 5400,
  },
  "KTEB": {
    "name": "Teterboro",
    "elevation_ft": 9,
    "runway_length_ft": 7000,
    "hot_high_category": "low",
    "climb_gradient_pct": 3.0,
    "operational_category": "metro",
    "density_altitude_summer_ft": 1200,
    "max_recommended_runway_ft": 6500,
  },
  "KOPF": {
    "name": "Miami-Opa Locka",
    "elevation_ft": 8,
    "runway_length_ft": 8002,
    "hot_high_category": "tropical",
    "climb_gradient_pct": 2.8,
    "operational_category": "metro",
    "density_altitude_summer_ft": 1500,
    "max_recommended_runway_ft": 6000,
  },
  "KPBI": {
    "name": "Palm Beach Intl",
    "elevation_ft": 19,
    "runway_length_ft": 10000,
    "hot_high_category": "tropical",
    "climb_gradient_pct": 2.8,
    "operational_category": "metro",
    "density_altitude_summer_ft": 1400,
    "max_recommended_runway_ft": 6200,
  },
  "KLAX": {
    "name": "Los Angeles Intl",
    "elevation_ft": 125,
    "runway_length_ft": 12091,
    "hot_high_category": "moderate",
    "climb_gradient_pct": 3.0,
    "operational_category": "metro",
    "density_altitude_summer_ft": 3500,
    "max_recommended_runway_ft": 7000,
  },
  "KSFO": {
    "name": "San Francisco Intl",
    "elevation_ft": 13,
    "runway_length_ft": 11500,
    "hot_high_category": "moderate",
    "climb_gradient_pct": 3.2,
    "operational_category": "metro",
    "density_altitude_summer_ft": 2200,
    "max_recommended_runway_ft": 6800,
  },
  "KJFK": {
    "name": "New York JFK",
    "elevation_ft": 13,
    "runway_length_ft": 14511,
    "hot_high_category": "low",
    "climb_gradient_pct": 2.8,
    "operational_category": "metro",
    "density_altitude_summer_ft": 1500,
    "max_recommended_runway_ft": 7000,
  },
  "KBOS": {
    "name": "Boston Logan",
    "elevation_ft": 20,
    "runway_length_ft": 10083,
    "hot_high_category": "low",
    "climb_gradient_pct": 3.0,
    "operational_category": "metro",
    "density_altitude_summer_ft": 1800,
    "max_recommended_runway_ft": 6500,
  },
  "KORD": {
    "name": "Chicago O'Hare",
    "elevation_ft": 672,
    "runway_length_ft": 13000,
    "hot_high_category": "moderate",
    "climb_gradient_pct": 3.0,
    "operational_category": "metro",
    "density_altitude_summer_ft": 2800,
    "max_recommended_runway_ft": 6800,
  },
  "KDFW": {
    "name": "Dallas/Fort Worth",
    "elevation_ft": 607,
    "runway_length_ft": 13700,
    "hot_high_category": "moderate",
    "climb_gradient_pct": 3.0,
    "operational_category": "metro",
    "density_altitude_summer_ft": 3200,
    "max_recommended_runway_ft": 6800,
  },
  "KDEN": {
    "name": "Denver Intl",
    "elevation_ft": 5431,
    "runway_length_ft": 16000,
    "hot_high_category": "high",
    "climb_gradient_pct": 4.0,
    "operational_category": "mountain_high",
    "density_altitude_summer_ft": 8500,
    "max_recommended_runway_ft": 5800,
  },
  "EGLL": {
    "name": "London Heathrow",
    "elevation_ft": 83,
    "runway_length_ft": 12799,
    "hot_high_category": "low",
    "climb_gradient_pct": 2.8,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 1500,
    "max_recommended_runway_ft": 7000,
  },
  "EGLF": {
    "name": "Farnborough",
    "elevation_ft": 238,
    "runway_length_ft": 8005,
    "hot_high_category": "low",
    "climb_gradient_pct": 3.0,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 1600,
    "max_recommended_runway_ft": 6200,
  },
  "EGGW": {
    "name": "London Luton",
    "elevation_ft": 526,
    "runway_length_ft": 7087,
    "hot_high_category": "low",
    "climb_gradient_pct": 3.2,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 2000,
    "max_recommended_runway_ft": 6000,
  },
  "LFPB": {
    "name": "Paris Le Bourget",
    "elevation_ft": 218,
    "runway_length_ft": 9843,
    "hot_high_category": "low",
    "climb_gradient_pct": 3.0,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 1800,
    "max_recommended_runway_ft": 6200,
  },
  "LFPG": {
    "name": "Paris Charles de Gaulle",
    "elevation_ft": 392,
    "runway_length_ft": 13829,
    "hot_high_category": "low",
    "climb_gradient_pct": 2.8,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 2000,
    "max_recommended_runway_ft": 7000,
  },
  "LSGG": {
    "name": "Geneva",
    "elevation_ft": 1411,
    "runway_length_ft": 12795,
    "hot_high_category": "moderate",
    "climb_gradient_pct": 3.5,
    "operational_category": "mountain_high",
    "density_altitude_summer_ft": 4500,
    "max_recommended_runway_ft": 5800,
  },
  "TNCM": {
    "name": "St Maarten",
    "elevation_ft": 14,
    "runway_length_ft": 7546,
    "hot_high_category": "tropical",
    "climb_gradient_pct": 3.0,
    "operational_category": "caribbean_short",
    "density_altitude_summer_ft": 2000,
    "max_recommended_runway_ft": 5500,
  },
  "MYNN": {
    "name": "Nassau",
    "elevation_ft": 16,
    "runway_length_ft": 11200,
    "hot_high_category": "tropical",
    "climb_gradient_pct": 2.8,
    "operational_category": "caribbean",
    "density_altitude_summer_ft": 1500,
    "max_recommended_runway_ft": 6000,
  },
  "TIST": {
    "name": "St Thomas",
    "elevation_ft": 23,
    "runway_length_ft": 7000,
    "hot_high_category": "tropical",
    "climb_gradient_pct": 3.2,
    "operational_category": "caribbean_short",
    "density_altitude_summer_ft": 1800,
    "max_recommended_runway_ft": 5200,
  },
  "OMDB": {
    "name": "Dubai Intl",
    "elevation_ft": 62,
    "runway_length_ft": 13124,
    "hot_high_category": "severe",
    "climb_gradient_pct": 3.5,
    "operational_category": "hot_high_hub",
    "density_altitude_summer_ft": 5500,
    "max_recommended_runway_ft": 6500,
  },
  "RJTT": {
    "name": "Tokyo Haneda",
    "elevation_ft": 35,
    "runway_length_ft": 10000,
    "hot_high_category": "low",
    "climb_gradient_pct": 3.0,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 2000,
    "max_recommended_runway_ft": 6500,
  },
  "VHHH": {
    "name": "Hong Kong Intl",
    "elevation_ft": 28,
    "runway_length_ft": 12467,
    "hot_high_category": "tropical",
    "climb_gradient_pct": 3.2,
    "operational_category": "international_hub",
    "density_altitude_summer_ft": 2500,
    "max_recommended_runway_ft": 6800,
  },
}

_PLACE_TO_ICAO: Dict[str, str] = {
    "aspen": "KASE",
    "telluride": "KTEX",
    "eagle": "KEGE",
    "vail": "KEGE",
    "jackson": "KJAC",
    "jackson hole": "KJAC",
    "teterboro": "KTEB",
    "teb": "KTEB",
    "new york": "KTEB",
    "nyc": "KTEB",
    "manhattan": "KTEB",
    "miami": "KOPF",
    "opa locka": "KOPF",
    "palm beach": "KPBI",
    "los angeles": "KLAX",
    "la": "KLAX",
    "san francisco": "KSFO",
    "sfo": "KSFO",
    "boston": "KBOS",
    "chicago": "KORD",
    "dallas": "KDFW",
    "denver": "KDEN",
    "london": "EGLF",
    "heathrow": "EGLL",
    "farnborough": "EGLF",
    "luton": "EGGW",
    "paris": "LFPB",
    "le bourget": "LFPB",
    "geneva": "LSGG",
    "nassau": "MYNN",
    "st maarten": "TNCM",
    "st thomas": "TIST",
    "caribbean": "TNCM",
    "dubai": "OMDB",
    "tokyo": "RJTT",
    "hong kong": "VHHH",
}

_ICAO_IN_TEXT_RE = re.compile(r"\b([A-Z]{4})\b")


@dataclass(frozen=True)
class AirportOperationalProfile:
    icao: str
    name: str
    elevation_ft: int
    runway_length_ft: int
    hot_high_category: str
    climb_gradient_pct: float
    operational_category: str
    density_altitude_summer_ft: int
    max_recommended_runway_ft: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "icao": self.icao,
            "name": self.name,
            "elevation_ft": self.elevation_ft,
            "runway_length_ft": self.runway_length_ft,
            "hot_high_category": self.hot_high_category,
            "climb_gradient_pct": self.climb_gradient_pct,
            "operational_category": self.operational_category,
            "density_altitude_summer_ft": self.density_altitude_summer_ft,
            "max_recommended_runway_ft": self.max_recommended_runway_ft,
        }


def profile_from_icao(icao: str) -> Optional[AirportOperationalProfile]:
    raw = _ICAO_RAW.get((icao or "").upper())
    if not raw:
        return None
    return AirportOperationalProfile(
        icao=icao.upper(),
        name=str(raw["name"]),
        elevation_ft=int(raw["elevation_ft"]),
        runway_length_ft=int(raw["runway_length_ft"]),
        hot_high_category=str(raw["hot_high_category"]),
        climb_gradient_pct=float(raw["climb_gradient_pct"]),
        operational_category=str(raw["operational_category"]),
        density_altitude_summer_ft=int(raw["density_altitude_summer_ft"]),
        max_recommended_runway_ft=int(raw["max_recommended_runway_ft"]),
    )


def all_icao_codes() -> List[str]:
    return sorted(_ICAO_RAW.keys())


def resolve_icao_from_place_token(token: str) -> Optional[str]:
    key = (token or "").strip().lower()
    if not key:
        return None
    if len(key) == 4 and key.isalpha():
        code = key.upper()
        if code in _ICAO_RAW:
            return code
    return _PLACE_TO_ICAO.get(key)


def resolve_airports_in_text(route_label: str) -> List[AirportOperationalProfile]:
    """Resolve constraint airports from route label (places, ICAO codes)."""
    blob = (route_label or "").lower()
    found: List[AirportOperationalProfile] = []
    seen: Set[str] = set()

    for m in _ICAO_IN_TEXT_RE.finditer(route_label or ""):
        code = m.group(1).upper()
        if code in _ICAO_RAW and code not in seen:
            p = profile_from_icao(code)
            if p:
                found.append(p)
                seen.add(code)

    for place, icao in sorted(_PLACE_TO_ICAO.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(place)}\b", blob):
            if icao not in seen:
                p = profile_from_icao(icao)
                if p:
                    found.append(p)
                    seen.add(icao)
    # Prefer mountain/short-field operational categories when multiple airports match.
    def _mountain_first(p: AirportOperationalProfile) -> int:
        cat = (getattr(p, "operational_category", "") or "").lower()
        return 0 if "mountain" in cat else 1

    return sorted(found, key=_mountain_first)


def mission_airport_profiles(route_labels: List[str]) -> List[AirportOperationalProfile]:
    airports: List[AirportOperationalProfile] = []
    seen: Set[str] = set()
    for label in route_labels:
        for ap in resolve_airports_in_text(label):
            if ap.icao not in seen:
                airports.append(ap)
                seen.add(ap.icao)
    return airports

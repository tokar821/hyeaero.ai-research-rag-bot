"""
Verified route distance catalog (nm) — no imports, no heuristics.

Single source of truth for catalog distances; consumed by route_distance_authority.
"""

from __future__ import annotations

from typing import Dict

VERIFIED_ROUTE_DISTANCE_NM: Dict[str, float] = {
    "los angeles -> miami": 1950.0,
    "la -> miami": 1950.0,
    "new york -> london": 3000.0,
    "nyc -> london": 3000.0,
    "teterboro -> london": 3100.0,
    "teb -> london": 3100.0,
    "san francisco -> paris": 4900.0,
    "sfo -> paris": 4900.0,
    "west coast -> europe": 4800.0,
    "boston -> paris": 3200.0,
    "dallas -> aspen": 650.0,
    "aspen -> telluride": 120.0,
    "teterboro -> palm beach": 900.0,
    "miami -> caribbean": 950.0,
    "miami -> nassau": 950.0,
    "miami -> south america": 2800.0,
    "miami -> sao paulo": 2800.0,
    "san francisco -> tokyo": 5100.0,
    "sfo -> tokyo": 5100.0,
    "san francisco -> london": 4700.0,
    "sfo -> london": 4700.0,
    "dallas -> london": 4700.0,
    "new york -> dallas": 1200.0,
    "nyc -> dallas": 1200.0,
    "london -> dubai": 3000.0,
    "dallas -> dubai": 7300.0,
    "new york -> tokyo": 5900.0,
    "nyc -> tokyo": 5900.0,
    "tokyo -> seoul": 720.0,
    "los angeles -> london": 5450.0,
    "la -> london": 5450.0,
    "honolulu -> sydney": 4400.0,
    "dallas -> new york": 1200.0,
    "chicago -> london": 3200.0,
    "aspen -> europe": 4200.0,
    "aspen -> geneva": 4200.0,
    "new york -> dubai": 6200.0,
    "nyc -> dubai": 6200.0,
    "jfk -> dubai": 6200.0,
    "los angeles -> dubai": 7200.0,
    "la -> dubai": 7200.0,
    "new york -> los angeles": 2450.0,
    "nyc -> los angeles": 2450.0,
    "boston -> los angeles": 2600.0,
    "boston -> miami": 1100.0,
    "boston -> abu dhabi": 5900.0,
    "new york -> abu dhabi": 5900.0,
    "nyc -> abu dhabi": 5900.0,
    "houston -> london": 4200.0,
    "houston -> calgary": 1750.0,
}

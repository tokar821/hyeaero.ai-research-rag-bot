"""
Aviation geography database — cities, regions, and airport metro aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional


@dataclass(frozen=True)
class AviationPlace:
    canonical: str
    kind: str  # city | region | airport_metro
    region: Optional[str] = None
    country: Optional[str] = None


def _p(
    canonical: str,
    kind: str = "city",
    *,
    region: Optional[str] = None,
    country: Optional[str] = None,
    aliases: tuple[str, ...] = (),
) -> tuple[str, AviationPlace, tuple[str, ...]]:
    return (canonical.lower(), AviationPlace(canonical, kind, region, country), aliases)


# (canonical_key, place, extra_aliases)
_PLACE_ROWS = [
    _p("New York", "city", region="US Northeast", country="US", aliases=("nyc", "new york city", "manhattan", "teterboro")),
    _p("Los Angeles", "city", region="US West Coast", country="US", aliases=("la", "l.a.", "van nuys", "burbank")),
    _p("San Francisco", "city", region="US West Coast", country="US", aliases=("sf", "san fran", "oakland", "sjc", "sfo")),
    _p("Miami", "city", region="South Florida", country="US", aliases=("mia", "opf", "fll", "south florida")),
    _p("Chicago", "city", region="US Midwest", country="US", aliases=("ord", "mdw", "chi")),
    _p("Boston", "city", region="US Northeast", country="US", aliases=("bos", "bedford")),
    _p("Dallas", "city", region="US South Central", country="US", aliases=("dfw", "dal", "addison")),
    _p("Houston", "city", region="US South Central", country="US", aliases=("iah", "hou", "houston")),
    _p("Seattle", "city", region="US West Coast", country="US", aliases=("sea", "boeing field")),
    _p("Denver", "city", region="US Mountain", country="US", aliases=("den", "centennial")),
    _p("Las Vegas", "city", region="US West", country="US", aliases=("las", "henderson")),
    _p("Atlanta", "city", region="US Southeast", country="US", aliases=("atl", "pdk")),
    _p("Palm Beach", "city", region="South Florida", country="US", aliases=("pbi", "west palm")),
    _p("Teterboro", "airport_metro", region="US Northeast", country="US", aliases=("teb",)),
    _p("Aspen", "city", region="US Mountain", country="US", aliases=("ase", "snowmass")),
    _p("Telluride", "city", region="US Mountain", country="US", aliases=("tex", "mountain village")),
    _p("Jackson Hole", "city", region="US Mountain", country="US", aliases=("jac", "jackson")),
    _p("Vail", "city", region="US Mountain", country="US", aliases=("ege", "vail")),
    _p("Eagle", "city", region="US Mountain", country="US", aliases=("ege", "eagle county")),
    _p("Sun Valley", "city", region="US Mountain", country="US", aliases=("sun", "hailey")),
    _p("Banff", "city", region="Canada Mountain", country="Canada", aliases=("banff", "yyc ski")),
    _p("Salt Lake City", "city", region="US Mountain", country="US", aliases=("slc", "salt lake")),
    _p("Provo", "city", region="US Mountain", country="US", aliases=("pvu",)),
    _p("Philadelphia", "city", region="US Northeast", country="US", aliases=("phl", "philly")),
    _p("Reno", "city", region="US West", country="US", aliases=("rno",)),
    _p("San Diego", "city", region="US West Coast", country="US", aliases=("san", "lindbergh")),
    _p("Norway", "city", region="Europe", country="Norway", aliases=("osl", "norwegian")),
    _p("London", "city", region="Europe", country="UK", aliases=("lon", "lhr", "fab", "stansted")),
    _p("Reykjavik", "city", region="Europe", country="Iceland", aliases=("rey", "bikf", "keflavik")),
    _p("Scottsdale", "city", region="US West", country="US", aliases=("sdl", "phoenix scottsdale")),
    _p("Lisbon", "city", region="Europe", country="Portugal", aliases=("lis", "lppt")),
    _p("Paris", "city", region="Europe", country="France", aliases=("par", "cdg", "lbg", "le bourget")),
    _p("Geneva", "city", region="Europe", country="Switzerland", aliases=("gva", "lszh")),
    _p("Zurich", "city", region="Europe", country="Switzerland", aliases=("zrh",)),
    _p("Frankfurt", "city", region="Europe", country="Germany", aliases=("fra",)),
    _p("Berlin", "city", region="Europe", country="Germany", aliases=("ber", "txl", "berlin brandenburg")),
    _p("Moscow", "city", region="Europe", country="Russia", aliases=("moscaw", "moscoww", "svo", "dme", "vko")),
    _p("Munich", "city", region="Europe", country="Germany", aliases=("muc",)),
    _p("Tokyo", "city", region="Asia-Pacific", country="Japan", aliases=("tok", "hnd", "nrt")),
    _p("Seoul", "city", region="Asia-Pacific", country="Korea", aliases=("icn", "gimpo")),
    _p("Hong Kong", "city", region="Asia-Pacific", country="China", aliases=("hkg",)),
    _p("Singapore", "city", region="Asia-Pacific", country="Singapore", aliases=("sin",)),
    _p("Dubai", "city", region="Middle East", country="UAE", aliases=("dxb", "dwc")),
    _p("Abu Dhabi", "city", region="Middle East", country="UAE", aliases=("auh", "abu dhabi")),
    _p("Doha", "city", region="Middle East", country="Qatar", aliases=("doh",)),
    _p("Washington", "city", region="US Northeast", country="US", aliases=("dca", "iad", "bwi")),
    _p("Sydney", "city", region="Oceania", country="Australia", aliases=("syd",)),
    _p("Melbourne", "city", region="Oceania", country="Australia", aliases=("mel",)),
    _p("Perth", "city", region="Oceania", country="Australia", aliases=("per",)),
    _p(
        "Australian Extraction Strips",
        "region",
        region="Oceania",
        country=None,
        aliases=("australian extraction", "australian mining", "remote australian"),
    ),
    _p("Toronto", "city", region="North America", country="Canada", aliases=("yyz",)),
    _p("Vancouver", "city", region="North America", country="Canada", aliases=("yvr",)),
    _p("Mexico City", "city", region="Latin America", country="Mexico", aliases=("mex",)),
    _p("Sao Paulo", "city", region="Latin America", country="Brazil", aliases=("sao paulo", "são paulo", "gru")),
    _p("Madrid", "city", region="Europe", country="Spain", aliases=("mad", "lemd")),
    _p("Calgary", "city", region="North America", country="Canada", aliases=("yyc",)),
    _p("Yellowknife", "city", region="Arctic Canada", country="Canada", aliases=("yzf", "northwest territories")),
    _p(
        "Nunavut Field Ops",
        "region",
        region="Arctic Canada",
        country=None,
        aliases=("nunavut field", "nunavut logistics", "nunavut operations"),
    ),
    _p(
        "Northern Alberta Oil Fields",
        "region",
        region="Arctic Canada",
        country=None,
        aliases=("northern alberta oil", "northern alberta fields", "calgary oil fields"),
    ),
    _p(
        "Remote Gravel Strips",
        "region",
        region="Arctic Canada",
        country=None,
        aliases=("remote gravel", "arctic gravel strips", "ice strip operations", "gravel strips northern canada"),
    ),
    _p("Anchorage", "city", region="North America", country="US", aliases=("anc", "alaska")),
    _p("Lagos", "city", region="Africa", country="Nigeria", aliases=("los nigeria", "dnmm", "nigeria")),
    _p("West Africa", "region", region="Africa", country=None, aliases=("west african", "west africa mining")),
    _p(
        "Permian Basin",
        "region",
        region="US South Central",
        country="US",
        aliases=("permian basin", "permian", "west texas oil"),
    ),
    _p(
        "Nigerian Energy Corridor",
        "region",
        region="Africa",
        country=None,
        aliases=("nigerian energy", "nigeria drilling", "nigeria energy corridor"),
    ),
    _p(
        "Northern Africa",
        "region",
        region="Africa",
        country=None,
        aliases=("northern africa", "north africa"),
    ),
    _p(
        "Pilbara",
        "region",
        region="Oceania",
        country=None,
        aliases=("pilbara", "pilbara sites", "remote pilbara"),
    ),
    _p(
        "Offshore Rigs",
        "region",
        region="Remote Field",
        country=None,
        aliases=("offshore rigs", "offshore oil", "offshore platforms"),
    ),
    _p("Remote Drilling Sites", "region", region="Remote Field", country=None, aliases=("remote drilling", "drilling sites")),
    _p("Arctic Oil Platforms", "region", region="Arctic", country=None, aliases=("arctic oil", "oil platforms")),
    _p("Arctic Industrial Access", "region", region="Arctic", country=None, aliases=("arctic mining", "arctic industrial")),
    _p("Desert Energy Corridor", "region", region="Middle East", country=None, aliases=("desert strips", "desert energy", "remote desert")),
    _p("Riyadh", "city", region="Middle East", country="Saudi Arabia", aliases=("ruh",)),
    _p("West Coast", "region", region="US West Coast", country="US", aliases=("us west coast",)),
    _p("East Coast", "region", region="US East Coast", country="US", aliases=("us east coast",)),
    _p("Europe", "region", region="Europe", country=None, aliases=("eu", "continental europe")),
    _p("Caribbean", "region", region="Caribbean", country=None, aliases=("the caribbean", "bahamas region")),
    _p("Transatlantic", "region", region="Transatlantic", country=None, aliases=("transatlantic",)),
]

ALIAS_TO_PLACE: Dict[str, AviationPlace] = {}
CANONICAL_PLACES: Dict[str, AviationPlace] = {}

for _key, place, extra_aliases in _PLACE_ROWS:
    CANONICAL_PLACES[_key] = place
    ALIAS_TO_PLACE[_key] = place
    ALIAS_TO_PLACE[place.canonical.lower()] = place
    for a in extra_aliases:
        ALIAS_TO_PLACE[a.lower().strip()] = place

# ICAO/IATA codes (3-4 letter) mapped separately for code-in-text resolution
_AIRPORT_CODES: Dict[str, str] = {
    "teb": "Teterboro",
    "pbi": "Palm Beach",
    "sfo": "San Francisco",
    "lax": "Los Angeles",
    "mia": "Miami",
    "bos": "Boston",
    "ord": "Chicago",
    "dfw": "Dallas",
    "lhr": "London",
    "cdg": "Paris",
    "hnd": "Tokyo",
    "nrt": "Tokyo",
    "icn": "Seoul",
    "dxb": "Dubai",
    "gva": "Geneva",
    "ber": "Berlin",
    "svo": "Moscow",
    "yyc": "Calgary",
    "yzf": "Yellowknife",
}

# Tokens that invalidate an endpoint (aircraft, UI, advisory)
_BLOCKED_ENDPOINT_TOKENS: FrozenSet[str] = frozenset(
    """
    what would you like work full higher alternatives explore compared chartering
    efficiency ownership recommend best option help please show tell about into onto
    consultant insight bottom line assuming passengers typical business aircraft jet
    planes plane flights flying operate operating efficiency pax passengers seats
    executives people travel with gulfstream bombardier dassault falcon challenger
    citation phenom praetor global learjet hawker embraer netjets flexjet honda
    citation latitude longitude phenom legacy vision global7500 g650 g700 falcon8x
    alternative alternatives recommendation recommendations mission summary best fit
    bottom line consultant insight heading bullet section
    """.split()
)

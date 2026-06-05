"""
Operational route distance overrides (nm) — **not** the primary distance source.

Hybrid resolution order (see ``route_distance_authority.resolve_route_distance``):

  1. **Geodesic** — great-circle from airport reference coordinates (deterministic, scalable).
  2. **Operational overrides** (this file) — only where broker *planning* distance materially
     differs from great-circle (NAT corridor conventions, macro regions, reserve-inclusive legs).
  3. **Persistent cache** (future DB table) — audited pairs from prior verified lookups.
  4. **Unresolved** — 0 nm; never invent distance via LLM or Tavily in ranking/feasibility.

LLM and Tavily are **not** used to compute stage length for mission ranking or route-map BINDS.
They may narrate or suggest sources for human verification only.
"""

from __future__ import annotations

from typing import Dict

# Small curated set — add only when geodesic ≠ operational planning distance.
OPERATIONAL_ROUTE_OVERRIDES_NM: Dict[str, float] = {
    # Macro corridors (not point airports)
    "west coast -> europe": 4800.0,
    "miami -> caribbean": 950.0,
    "aspen -> europe": 4200.0,
    # TEB–London: planning convention ~100 nm above great-circle for NE corridor
    "teterboro -> london": 3100.0,
    "teb -> london": 3100.0,
    "new york -> london": 3000.0,
    "nyc -> london": 3000.0,
}

# Back-compat alias — authority imports this name; now means overrides only.
VERIFIED_ROUTE_DISTANCE_NM = OPERATIONAL_ROUTE_OVERRIDES_NM

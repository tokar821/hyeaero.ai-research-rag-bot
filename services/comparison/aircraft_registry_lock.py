"""
Canonical aircraft registry lock for Comparison v2.

Only catalog-verified aircraft may appear in structured comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.ontology.aircraft_normalization import normalize_aircraft_model

_BANNED_NAME_RE = re.compile(
    r"\b(?:unverified|unknown|tbd|n/?a|placeholder|partial|generic)\b",
    re.I,
)

# Display names allowed for comparison (catalog keys only — no new data sources).
CANONICAL_COMPARISON_REGISTRY: frozenset[str] = frozenset(AIRCRAFT_PROFILES.keys())

_SPOKEN_ALIASES: Dict[str, str] = {
    "global seven five zero zero": "Global 7500",
    "bombardier global seven five zero zero": "Global 7500",
    "eight x extended range": "Falcon 8X",
    "falcon eight x extended range": "Falcon 8X",
    "falcon eight x": "Falcon 8X",
    "citation longitude": "Citation Longitude",
    "cessna citation longitude": "Citation Longitude",
    "longitude": "Citation Longitude",
    "g700": "Gulfstream G700",
    "gulfstream g700": "Gulfstream G700",
    "g650": "Gulfstream G650",
    "gulfstream g650": "Gulfstream G650",
    "legacy 600": "Legacy 600",
    "embraer legacy 600": "Legacy 600",
}


@dataclass(frozen=True)
class RegistryLockResult:
    canonical: Tuple[str, ...]
    rejected: Tuple[str, ...]
    reasons: Tuple[str, ...]


def _is_valid_canonical_name(name: str) -> bool:
    n = (name or "").strip()
    if not n or len(n) < 4:
        return False
    if _BANNED_NAME_RE.search(n):
        return False
    if n not in CANONICAL_COMPARISON_REGISTRY:
        return False
    return True


def resolve_to_registry_name(raw: str) -> Optional[str]:
    """Map raw token to a single canonical catalog name, or None."""
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias
    from services.catalog.catalog_alias_resolver import (
        resolve_canonical_display_name,
        resolve_catalog_profile_key,
    )

    raw = (raw or "").strip()
    if not raw:
        return None
    if _BANNED_NAME_RE.search(raw):
        return None

    alias_canonical = resolve_aircraft_alias(raw)
    if alias_canonical and alias_canonical in CANONICAL_COMPARISON_REGISTRY:
        return alias_canonical

    canonical = resolve_canonical_display_name(raw)
    profile_key = resolve_catalog_profile_key(raw)
    if profile_key and profile_key in CANONICAL_COMPARISON_REGISTRY:
        return profile_key
    if canonical in CANONICAL_COMPARISON_REGISTRY:
        return canonical

    spoken = re.sub(r"[^\w\s]", " ", raw.lower())
    spoken = re.sub(r"\s+", " ", spoken).strip()
    if spoken in _SPOKEN_ALIASES:
        return _SPOKEN_ALIASES[spoken]
    if spoken in ("falcon eight x", "eight x"):
        return "Falcon 8X"
    if "eight x" in spoken and ("extended" in spoken or spoken.startswith("falcon")):
        return "Falcon 8X"
    if spoken == "global seven five zero zero" or (
        "global" in spoken and "seven five zero zero" in spoken
    ):
        return "Global 7500"

    if raw in CANONICAL_COMPARISON_REGISTRY:
        return raw

    norm = normalize_aircraft_model(raw)
    if norm is None:
        return None
    display = norm.display_name.strip()
    if display in CANONICAL_COMPARISON_REGISTRY:
        return display

    # Case-insensitive catalog match
    low = display.lower()
    for catalog_name in CANONICAL_COMPARISON_REGISTRY:
        if catalog_name.lower() == low:
            return catalog_name

    return None


def lock_comparison_aircraft(models: Sequence[str]) -> RegistryLockResult:
    """
    Resolve and dedupe models to canonical registry names only.
    Unknown / partial / unmapped tokens are rejected.
    """
    rejected: List[str] = []
    reasons: List[str] = []
    seen: set[str] = set()
    canonical: List[str] = []

    for raw in models or []:
        token = str(raw or "").strip()
        if not token:
            continue
        resolved = resolve_to_registry_name(token)
        if resolved is None:
            rejected.append(token)
            reasons.append(f"not_in_registry:{token}")
            continue
        if not _is_valid_canonical_name(resolved):
            rejected.append(token)
            reasons.append(f"invalid_canonical:{resolved}")
            continue
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        canonical.append(resolved)

    return RegistryLockResult(
        canonical=tuple(canonical),
        rejected=tuple(rejected),
        reasons=tuple(reasons),
    )


__all__ = [
    "CANONICAL_COMPARISON_REGISTRY",
    "RegistryLockResult",
    "lock_comparison_aircraft",
    "resolve_to_registry_name",
]

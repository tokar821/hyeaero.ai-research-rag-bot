"""
Resolve manufacturer/category phrases into ranked catalog-verified candidates.

Uses only models present in the comparison registry / AKAL authority records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from services.comparison.aircraft_registry_lock import (
    CANONICAL_COMPARISON_REGISTRY,
    lock_comparison_aircraft,
)
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

# Versioned catalog — see acquisition_tier_catalog.py (checksum validated in CI).
from services.broker_reasoning.acquisition_tier_catalog import ACQUISITION_TIER_MUSD as _ACQUISITION_TIER_MUSD

_MANUFACTURER_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "Gulfstream": (
        "Gulfstream G280",
        "Gulfstream G650",
        "Gulfstream G650ER",
        "Gulfstream G700",
    ),
    "Dassault": ("Falcon 2000", "Falcon 7X", "Falcon 8X"),
    "Bombardier": (
        "Challenger 350",
        "Challenger 650",
        "Challenger Longitude",
        "Global 6500",
        "Global 7500",
        "Learjet 75",
    ),
    "Cessna": (
        "Citation CJ2",
        "Citation CJ4",
        "Citation Latitude",
        "Citation Longitude",
    ),
    "Embraer": ("Praetor 600",),
    "Citation": (
        "Citation CJ2",
        "Citation CJ4",
        "Citation Latitude",
        "Citation Longitude",
    ),
    "Challenger": ("Challenger 350", "Challenger 650", "Challenger Longitude"),
    "Falcon": ("Falcon 2000", "Falcon 7X", "Falcon 8X"),
}


@dataclass(frozen=True)
class CategoryResolution:
    phrase: str
    manufacturer: Optional[str]
    candidates: Tuple[str, ...]
    ranking_basis: str
    notes: Tuple[str, ...] = ()


def _verified_models(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for n in names:
        lock = lock_comparison_aircraft([n])
        out.extend(lock.canonical)
    seen: set[str] = set()
    deduped: List[str] = []
    for m in out:
        k = m.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(m)
    return deduped


def _rank_by_price(models: Sequence[str], *, ascending: bool = True) -> List[str]:
    def _tier(name: str) -> float:
        if name in _ACQUISITION_TIER_MUSD:
            return _ACQUISITION_TIER_MUSD[name]
        profile = AIRCRAFT_PROFILES.get(name) or {}
        oi = float(profile.get("operating_index") or 0.5)
        return oi * 25.0

    return sorted(models, key=_tier, reverse=not ascending)


def _rank_by_budget(models: Sequence[str], budget_musd: float) -> List[str]:
    ranked = _rank_by_price(models, ascending=True)
    feasible = [m for m in ranked if _ACQUISITION_TIER_MUSD.get(m, 99) <= budget_musd * 1.15]
    return feasible or ranked[:3]


def resolve_category(
    query: str,
    *,
    manufacturer: Optional[str] = None,
    budget_musd: Optional[float] = None,
    price_sensitive: bool = False,
) -> CategoryResolution:
    """Resolve category/manufacturer phrases to ranked verified candidates."""
    q = (query or "").strip()
    low = q.lower()
    notes: List[str] = []

    mfr = manufacturer
    if not mfr:
        if re.search(r"\b(?:gulfstream|g\s*\d{3})\b", low):
            mfr = "Gulfstream"
        elif re.search(r"\b(?:dassault|falcon)\b", low):
            mfr = "Dassault"
        elif re.search(r"\b(?:citation|cessna|latitude|longitude)\b", low):
            mfr = "Citation"
        elif re.search(r"\b(?:challenger|bombardier|global|learjet)\b", low):
            mfr = "Challenger"
        elif re.search(r"\b(?:embraer|phenom|praetor)\b", low):
            mfr = "Embraer"

    phrase = mfr or "general"
    family_key = mfr or ""
    if family_key == "Citation" and re.search(r"\bcitation\s+alternative\b", low):
        family_key = "Citation"

    if re.search(r"\bsuper-?\s*midsize\b", low):
        candidates = _verified_models(
            ("Citation Longitude", "Challenger 350", "Praetor 600", "Gulfstream G280")
        )
        if budget_musd is not None:
            ranked = _rank_by_price(candidates, ascending=True)
            stretch = 1.5
            feasible = [m for m in ranked if _ACQUISITION_TIER_MUSD.get(m, 99) <= budget_musd * stretch]
            candidates = feasible or ranked[:3]
        return CategoryResolution(
            phrase="super-midsize",
            manufacturer=mfr,
            candidates=tuple(candidates[:5]),
            ranking_basis="super_midsize_class",
            notes=tuple(notes),
        )

    raw_family = _MANUFACTURER_FAMILIES.get(family_key, ())
    candidates = _verified_models(raw_family)

    if not candidates and mfr:
        # Fall back to registry scan by manufacturer token in name.
        token = mfr.lower()
        candidates = _verified_models(
            [n for n in CANONICAL_COMPARISON_REGISTRY if token in n.lower()]
        )

    ranking_basis = "catalog_verified"
    if price_sensitive or (
        re.search(r"\b(?:cheap|cheapest|affordable)\b", low)
        or (re.search(r"\bbudget\b", low) and not re.search(r"\bsuper-?\s*midsize\b", low))
    ):
        candidates = _rank_by_price(candidates, ascending=True)
        ranking_basis = "entry_acquisition_tier"
        if mfr == "Gulfstream":
            # Cheap/budget Gulfstream → entry catalog tier only (G280), not G650/G700 class.
            entry = [c for c in candidates if _ACQUISITION_TIER_MUSD.get(c, 99) <= 18.0]
            candidates = entry or candidates[:1]
            notes.append(
                "Entry Gulfstream in verified catalog starts with G280; older G450/G550-class "
                "may trade lower but are not in the active comparison registry."
            )
        elif budget_musd is not None:
            candidates = _rank_by_budget(candidates, budget_musd)
    elif budget_musd is not None:
        candidates = _rank_by_budget(candidates, budget_musd)
        ranking_basis = f"budget_fit_{budget_musd}M"
    else:
        candidates = _rank_by_price(candidates, ascending=True)

    return CategoryResolution(
        phrase=phrase,
        manufacturer=mfr,
        candidates=tuple(candidates[:5]),
        ranking_basis=ranking_basis,
        notes=tuple(notes),
    )


def resolve_reference_alternatives(
    reference_model: str,
    *,
    budget_musd: Optional[float] = None,
    lower_cost: bool = False,
) -> CategoryResolution:
    """Find catalog-verified alternatives to a reference model."""
    ref = (reference_model or "").strip()
    lock = lock_comparison_aircraft([ref])
    canonical_ref = lock.canonical[0] if lock.canonical else ref

    verified: List[str] = []
    # Same-family lower-tier models from verified catalog.
    for _mfr, family in _MANUFACTURER_FAMILIES.items():
        if canonical_ref in family:
            idx = family.index(canonical_ref)
            verified = _verified_models(family[:idx])
            break

    if not verified:
        # Cross-class peers by operating index proximity.
        ref_profile = AIRCRAFT_PROFILES.get(canonical_ref) or {}
        ref_oi = float(ref_profile.get("operating_index") or 0.5)
        scored: List[tuple[float, str]] = []
        for name in CANONICAL_COMPARISON_REGISTRY:
            if name == canonical_ref:
                continue
            oi = float((AIRCRAFT_PROFILES.get(name) or {}).get("operating_index") or 0.5)
            scored.append((abs(oi - ref_oi), name))
        scored.sort(key=lambda x: x[0])
        verified = _verified_models([n for _, n in scored[:6]])

    if lower_cost or budget_musd is not None:
        verified = _rank_by_price(verified, ascending=True)
        if budget_musd is not None:
            verified = _rank_by_budget(verified, budget_musd)
    elif verified:
        verified = _rank_by_price(verified, ascending=True)

    notes: Tuple[str, ...] = ()
    if lower_cost:
        notes = (f"Alternatives below {canonical_ref} on acquisition tier.",)

    return CategoryResolution(
        phrase=f"alternatives_to_{canonical_ref}",
        manufacturer=None,
        candidates=tuple(verified[:5]),
        ranking_basis="tier_peer_or_lower_cost",
        notes=notes,
    )


__all__ = ["CategoryResolution", "resolve_category", "resolve_reference_alternatives"]

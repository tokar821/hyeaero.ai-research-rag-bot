"""Mission-aligned lower-cost alternatives — not random catalog picks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

# Same cabin mission band → lower acquisition tier (catalog-verified only).
_MISSION_ALTERNATIVE_LADDER: Dict[str, Tuple[str, ...]] = {
    "Citation Longitude": (
        "Citation Latitude",
        "Praetor 600",
        "Challenger 350",
        "Gulfstream G280",
    ),
    "Citation Latitude": (
        "Citation CJ4",
        "Praetor 600",
        "Challenger 350",
    ),
    "Gulfstream G650": (
        "Gulfstream G280",
        "Challenger 650",
        "Falcon 2000",
        "Citation Longitude",
    ),
    "Gulfstream G700": (
        "Gulfstream G650",
        "Gulfstream G650ER",
        "Global 6500",
        "Falcon 8X",
    ),
    "Challenger 350": (
        "Citation Latitude",
        "Praetor 600",
        "Citation CJ4",
    ),
    "Challenger 650": (
        "Challenger 350",
        "Citation Longitude",
        "Falcon 2000",
    ),
    "Falcon 8X": (
        "Falcon 7X",
        "Falcon 2000",
        "Challenger 650",
        "Global 6500",
    ),
    "Praetor 600": (
        "Citation Latitude",
        "Citation CJ4",
        "Challenger 350",
    ),
}


@dataclass(frozen=True)
class AlternativeOpportunity:
    model: str
    rationale: str


def _tier_musd(model: str) -> float:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    if model in _ACQUISITION_TIER_MUSD:
        return _ACQUISITION_TIER_MUSD[model]
    profile = AIRCRAFT_PROFILES.get(model) or {}
    return float(profile.get("operating_index") or 0.5) * 25.0


def resolve_alternatives(
    reference_model: str,
    *,
    budget_musd: Optional[float] = None,
    max_items: int = 4,
) -> List[AlternativeOpportunity]:
    """Return same-mission, lower-cost alternatives from verified catalog."""
    lock = lock_comparison_aircraft([reference_model])
    ref = lock.canonical[0] if lock.canonical else reference_model.strip()

    ladder = list(_MISSION_ALTERNATIVE_LADDER.get(ref, ()))
    if not ladder:
        # Fallback: lower operating index in same category.
        ref_cat = (AIRCRAFT_PROFILES.get(ref) or {}).get("category", "")
        scored: List[tuple[float, str]] = []
        ref_tier = _tier_musd(ref)
        for name in AIRCRAFT_PROFILES:
            if name == ref:
                continue
            if ref_cat and (AIRCRAFT_PROFILES.get(name) or {}).get("category") != ref_cat:
                continue
            tier = _tier_musd(name)
            if tier < ref_tier:
                scored.append((tier, name))
        scored.sort(key=lambda x: x[0])
        ladder = [n for _, n in scored[:6]]

    verified: List[str] = []
    for name in ladder:
        lk = lock_comparison_aircraft([name])
        if lk.canonical:
            verified.append(lk.canonical[0])

    if budget_musd is not None:
        verified = [m for m in verified if _tier_musd(m) <= budget_musd * 1.1]

    out: List[AlternativeOpportunity] = []
    ref_tier = _tier_musd(ref)
    for model in verified[:max_items]:
        tier = _tier_musd(model)
        if tier < ref_tier:
            rationale = (
                f"Same general mission band as the {ref}, typically "
                f"${tier:.0f}M+ acquisition tier versus ${ref_tier:.0f}M+ for the {ref}."
            )
        else:
            rationale = f"Credible step-down from the {ref} while staying in verified catalog."
        out.append(AlternativeOpportunity(model=model, rationale=rationale))

    return out


__all__ = ["AlternativeOpportunity", "resolve_alternatives"]

"""
Aircraft truth validator — verified specs only; no speculation on missing data.

Every aircraft cited in recommendations or comparisons must pass validation for:
  - max passengers
  - practical range (nm)
  - runway class
  - baggage volume (cu ft)
  - operating category

Never invent acquisition price, payload capability, runway performance, or nonstop capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from services.aircraft_truth.catalog_supplement import CATALOG_TRUTH_SUPPLEMENT
from services.aircraft_truth.constants import (
    FORBIDDEN_UNVERIFIED_CLAIM_KEYS,
    REQUIRED_TRUTH_FIELDS,
    TRUTH_FIELD_BAGGAGE_VOLUME,
    TRUTH_FIELD_MAX_PASSENGERS,
    TRUTH_FIELD_OPERATING_CATEGORY,
    TRUTH_FIELD_PRACTICAL_RANGE,
    TRUTH_FIELD_RUNWAY_CLASS,
    UNVERIFIED_AIRCRAFT_MESSAGE,
)
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

_RUNWAY_CLASS_LABELS = {
    "short_field": "Short-field capable",
    "regional": "Regional jet runway",
    "super_mid": "Super-midsize runway",
    "large_cabin": "Large-cabin runway",
    "ultra_long": "Ultra-long-range runway",
}


@dataclass(frozen=True)
class VerifiedAircraftFacts:
    model: str
    max_passengers: int
    practical_range_nm: float
    runway_class: str
    baggage_volume_cu_ft: float
    operating_category: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_passengers": self.max_passengers,
            "practical_range_nm": round(self.practical_range_nm, 1),
            "runway_class": self.runway_class,
            "runway_class_label": _RUNWAY_CLASS_LABELS.get(
                self.runway_class, self.runway_class
            ),
            "baggage_volume_cu_ft": round(self.baggage_volume_cu_ft, 1),
            "operating_category": self.operating_category,
        }


@dataclass
class AircraftTruthResult:
    model: str
    verified: bool = False
    facts: Optional[VerifiedAircraftFacts] = None
    missing_fields: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "verified": self.verified,
            "facts": self.facts.to_dict() if self.facts else None,
            "missing_fields": list(self.missing_fields),
            "message": self.message,
        }


def resolve_aircraft_profile(
    model: str,
    profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge catalog profile with verified supplement (supplement does not override core range/pax)."""
    base = dict(profile or AIRCRAFT_PROFILES.get(model) or {})
    supplement = CATALOG_TRUTH_SUPPLEMENT.get(model) or {}
    merged = {**base, **supplement}
    return merged


def _positive_number(value: Any) -> Optional[float]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _positive_int(value: Any) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def extract_verified_facts(
    model: str,
    profile: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[VerifiedAircraftFacts], List[str]]:
    """
    Build verified facts from merged profile. Returns (facts, missing_field_names).
    """
    merged = resolve_aircraft_profile(model, profile)
    missing: List[str] = []

    if not merged:
        return None, list(REQUIRED_TRUTH_FIELDS)

    if not supplement_truth_verified(model, merged):
        missing.append("truth_verified")

    pax = _positive_int(merged.get("pax_max_long_range"))
    if pax is None:
        pax = _positive_int(merged.get("pax_typical"))
    if pax is None:
        missing.append(TRUTH_FIELD_MAX_PASSENGERS)

    practical = _positive_number(merged.get("practical_nm"))
    if practical is None:
        missing.append(TRUTH_FIELD_PRACTICAL_RANGE)

    runway_class = str(merged.get("runway_class") or "").strip().lower()
    if not runway_class:
        missing.append(TRUTH_FIELD_RUNWAY_CLASS)

    baggage = _positive_number(merged.get("baggage_volume_cu_ft"))
    if baggage is None:
        missing.append(TRUTH_FIELD_BAGGAGE_VOLUME)

    category = str(merged.get("category") or "").strip().lower()
    if not category:
        missing.append(TRUTH_FIELD_OPERATING_CATEGORY)

    if missing:
        return None, missing

    return (
        VerifiedAircraftFacts(
            model=model,
            max_passengers=int(pax),
            practical_range_nm=float(practical),
            runway_class=runway_class,
            baggage_volume_cu_ft=float(baggage),
            operating_category=category,
        ),
        [],
    )


def supplement_truth_verified(model: str, merged: Mapping[str, Any]) -> bool:
    """Model must be in catalog with explicit truth_verified flag."""
    if model not in AIRCRAFT_PROFILES:
        return False
    if not CATALOG_TRUTH_SUPPLEMENT.get(model):
        return False
    return bool(merged.get("truth_verified"))


def validate_aircraft_truth(
    model: str,
    profile: Optional[Mapping[str, Any]] = None,
) -> AircraftTruthResult:
    """
    Validate that an aircraft has full verified truth data.

    If validation fails, ``message`` is exactly ``UNVERIFIED_AIRCRAFT_MESSAGE`` — no speculation.
    """
    name = (model or "").strip()
    if not name:
        return AircraftTruthResult(
            model="",
            verified=False,
            missing_fields=list(REQUIRED_TRUTH_FIELDS),
            message=UNVERIFIED_AIRCRAFT_MESSAGE,
        )

    facts, missing = extract_verified_facts(name, profile)
    if facts is not None:
        return AircraftTruthResult(model=name, verified=True, facts=facts, missing_fields=[])

    return AircraftTruthResult(
        model=name,
        verified=False,
        missing_fields=missing,
        message=UNVERIFIED_AIRCRAFT_MESSAGE,
    )


def filter_truth_verified_models(models: Sequence[str]) -> List[str]:
    """Keep only models with complete verified truth data."""
    out: List[str] = []
    for model in models:
        if validate_aircraft_truth(model).verified:
            out.append(model)
    return out


def is_forbidden_unverified_claim(key: str) -> bool:
    return (key or "").strip().lower() in FORBIDDEN_UNVERIFIED_CLAIM_KEYS


def reject_forbidden_claims(claims: Mapping[str, Any]) -> List[str]:
    """
    Return keys present in ``claims`` that must not be stated without verified sourcing.
    """
    blocked: List[str] = []
    for key in claims:
        if is_forbidden_unverified_claim(key):
            blocked.append(key)
    return blocked


def _qualitative_cost(operating_index: float) -> str:
    if operating_index <= 0.45:
        return "Lower direct operating cost band"
    if operating_index <= 0.7:
        return "Mid operating cost band"
    return "Higher direct operating cost band"


def _qualitative_liquidity(resale_score: float) -> str:
    if resale_score >= 0.82:
        return "Strong"
    if resale_score >= 0.72:
        return "Solid"
    return "Thinner"


def format_verified_comparison_snippets(
    facts: VerifiedAircraftFacts,
    *,
    profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Five-dimension comparison copy from verified facts only (no list/acquisition price)."""
    merged = resolve_aircraft_profile(facts.model, profile)
    return {
        "range": f"~{int(facts.practical_range_nm)} nm practical",
        "cabin": f"{facts.operating_category.replace('-', ' ')} class",
        "operating cost": _qualitative_cost(float(merged.get("operating_index") or 0.7)),
        "runway capability": _RUNWAY_CLASS_LABELS.get(
            facts.runway_class, facts.runway_class
        ),
        "liquidity": _qualitative_liquidity(float(merged.get("resale_score") or 0.7)),
    }


def format_verified_spec_block(facts: VerifiedAircraftFacts) -> str:
    """Short verified spec lines for advisor copy."""
    return (
        f"{facts.model}: up to {facts.max_passengers} passengers, "
        f"~{int(facts.practical_range_nm)} nm practical range, "
        f"{facts.baggage_volume_cu_ft:.0f} cu ft baggage, "
        f"{_RUNWAY_CLASS_LABELS.get(facts.runway_class, facts.runway_class)}, "
        f"{facts.operating_category} category."
    )


def unverified_response_for_model(model: str) -> str:
    """User-facing line when model lacks verified data."""
    name = (model or "").strip()
    if name:
        return f"{name}: {UNVERIFIED_AIRCRAFT_MESSAGE}"
    return UNVERIFIED_AIRCRAFT_MESSAGE

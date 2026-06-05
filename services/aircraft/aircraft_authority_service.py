"""
Aircraft Knowledge Authority Layer (AKAL) — single source of truth for aircraft facts.

Data-authority only. Does not alter routing, dispatch, or response shaping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Authoritative alias map (AKAL) — resolves broker tokens to canonical display names.
_AKAL_ALIAS_MAP: Dict[str, str] = {
    "longitude": "Citation Longitude",
    "citation longitude": "Citation Longitude",
    "cessna citation longitude": "Citation Longitude",
    "challenger longitude": "Challenger Longitude",
    "g280": "Gulfstream G280",
    "g 280": "Gulfstream G280",
    "g-280": "Gulfstream G280",
    "gulfstream g280": "Gulfstream G280",
    "cj3+": "Citation CJ3+",
    "cj3 plus": "Citation CJ3+",
    "citation cj3+": "Citation CJ3+",
    "citation cj3 plus": "Citation CJ3+",
    "falcon 8x": "Falcon 8X",
    "dassault falcon 8x": "Falcon 8X",
    "falcon eight x": "Falcon 8X",
    "3500": "Challenger 3500",
    "challenger 3500": "Challenger 3500",
    "cl3500": "Challenger 3500",
    "g650": "Gulfstream G650",
    "g650er": "Gulfstream G650ER",
    "g700": "Gulfstream G700",
    "g 700": "Gulfstream G700",
    "g-700": "Gulfstream G700",
    "gulfstream g700": "Gulfstream G700",
    "global 7500": "Global 7500",
    "falcon 7x": "Falcon 7X",
    "falcon 2000 lxs": "Falcon 2000LXS",
    "falcon 2000lxs": "Falcon 2000LXS",
    "dassault falcon 2000lxs": "Falcon 2000LXS",
    "praetor 600": "Praetor 600",
    "challenger 300": "Challenger 300",
    "citation latitude": "Citation Latitude",
}

# Canonical display name → verified profile lookup key (same-aircraft display aliases only).
_PROFILE_LOOKUP_KEY: Dict[str, str] = {
    "Dassault Falcon 8X": "Falcon 8X",
}

_MANUFACTURER_BY_CANONICAL: Dict[str, str] = {
    "Gulfstream G650": "Gulfstream",
    "Gulfstream G650ER": "Gulfstream",
    "Gulfstream G700": "Gulfstream",
    "Gulfstream G280": "Gulfstream",
    "Gulfstream G550": "Gulfstream",
    "Gulfstream G500": "Gulfstream",
    "Global 7500": "Bombardier",
    "Global 6500": "Bombardier",
    "Challenger 350": "Bombardier",
    "Challenger 3500": "Bombardier",
    "Challenger 650": "Bombardier",
    "Challenger Longitude": "Bombardier",
    "Citation Longitude": "Textron",
    "Citation Latitude": "Textron",
    "Citation CJ4": "Textron",
    "Citation CJ3+": "Textron",
    "Citation CJ2": "Textron",
    "Falcon 8X": "Dassault",
    "Falcon 7X": "Dassault",
    "Falcon 2000": "Dassault",
    "Falcon 2000LXS": "Dassault",
    "Praetor 600": "Embraer",
    "Challenger 300": "Bombardier",
    "Legacy 600": "Embraer",
    "Learjet 75": "Bombardier",
    "Pilatus PC-24": "Pilatus",
}

_STATIC_ENRICHMENT: Dict[str, Dict[str, Any]] = {
    "Citation Longitude": {
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 12,
        "max_cruise_speed": 476,
        "cabin_height_ft": 6.0,
        "cabin_width_ft": 6.4,
        "cabin_length_ft": 25.2,
        "production_start_year": 2019,
        "production_end_year": None,
        "current_in_production": True,
        "baggage_score": 0.85,
    },
    "Challenger 3500": {
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 10,
        "max_cruise_speed": 470,
        "cabin_height_ft": 6.1,
        "cabin_width_ft": 7.2,
        "cabin_length_ft": 28.4,
        "production_start_year": 2022,
        "production_end_year": None,
        "current_in_production": True,
        "baggage_score": 0.75,
    },
    "Citation CJ3+": {
        "passenger_capacity_min": 6,
        "passenger_capacity_max": 9,
        "max_cruise_speed": 416,
        "cabin_height_ft": 4.8,
        "cabin_width_ft": 4.8,
        "cabin_length_ft": 15.8,
        "production_start_year": 2014,
        "production_end_year": 2021,
        "current_in_production": False,
        "baggage_score": 0.5,
    },
    "Falcon 8X": {
        "manufacturer": "Dassault",
        "display_alias": "Dassault Falcon 8X",
    },
    "Falcon 2000LXS": {
        "manufacturer": "Dassault",
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 10,
        "max_cruise_speed": 564,
        "cabin_height_ft": 6.17,
        "cabin_width_ft": 7.67,
        "cabin_length_ft": 25.58,
        "takeoff_distance_ft": 4325,
        "production_start_year": 2014,
        "current_in_production": True,
    },
    "Praetor 600": {
        "manufacturer": "Embraer",
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 10,
        "max_cruise_speed": 466,
        "cabin_height_ft": 6.0,
        "cabin_width_ft": 6.83,
        "cabin_length_ft": 27.5,
        "takeoff_distance_ft": 4717,
        "production_start_year": 2019,
        "current_in_production": True,
    },
    "Challenger 300": {
        "manufacturer": "Bombardier",
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 9,
        "max_cruise_speed": 459,
        "cabin_height_ft": 6.08,
        "cabin_width_ft": 7.17,
        "cabin_length_ft": 28.5,
        "takeoff_distance_ft": 4850,
        "production_start_year": 2004,
        "production_end_year": 2014,
        "current_in_production": False,
    },
    "Challenger 350": {
        "manufacturer": "Bombardier",
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 9,
        "max_cruise_speed": 470,
        "cabin_height_ft": 6.08,
        "cabin_width_ft": 7.17,
        "cabin_length_ft": 28.5,
        "takeoff_distance_ft": 4800,
        "production_start_year": 2014,
        "current_in_production": True,
    },
    "Citation Latitude": {
        "manufacturer": "Textron",
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 9,
        "max_cruise_speed": 446,
        "cabin_height_ft": 6.0,
        "cabin_width_ft": 6.5,
        "cabin_length_ft": 21.83,
        "takeoff_distance_ft": 3900,
        "production_start_year": 2015,
        "current_in_production": True,
    },
    "Gulfstream G280": {
        "manufacturer": "Gulfstream",
        "passenger_capacity_min": 8,
        "passenger_capacity_max": 10,
        "max_cruise_speed": 482,
        "cabin_height_ft": 6.25,
        "cabin_width_ft": 6.83,
        "cabin_length_ft": 25.83,
        "takeoff_distance_ft": 4800,
        "production_start_year": 2012,
        "current_in_production": True,
    },
}

_COMPETITORS: Dict[str, List[str]] = {
    "gulfstream g280": ["Challenger 350", "Citation Longitude", "Praetor 600"],
    "citation longitude": ["Challenger 350", "Praetor 600", "Gulfstream G280"],
    "challenger 350": ["Citation Longitude", "Praetor 600", "Gulfstream G280"],
    "challenger 3500": ["Citation Longitude", "Praetor 600", "Challenger 650"],
    "falcon 8x": ["Gulfstream G650", "Global 6500", "Challenger 650"],
    "falcon 7x": ["Falcon 8X", "Gulfstream G650", "Global 6500"],
    "gulfstream g650": ["Falcon 8X", "Global 7500", "Global 6500"],
    "gulfstream g700": ["Gulfstream G650ER", "Global 7500", "Falcon 8X"],
    "global 7500": ["Gulfstream G650ER", "Falcon 8X"],
    "citation cj3+": ["Phenom 300E", "Pilatus PC-24", "Citation CJ4"],
}

_REPLACEMENTS: Dict[str, List[str]] = {
    "citation longitude": ["Praetor 600", "Challenger 350", "Gulfstream G280"],
    "gulfstream g280": ["Challenger 350", "Citation Longitude"],
    "falcon 8x": ["Falcon 7X", "Global 6500", "Gulfstream G650"],
    "challenger 3500": ["Challenger 650", "Citation Longitude", "Praetor 600"],
}

_CLAIM_RANGE_RE = re.compile(
    r"(?P<model>[A-Za-z0-9][\w\s+\-]{2,40}?)\s+range\s+(?P<val>\d{3,5})\s*nm",
    re.I,
)
_CLAIM_PAX_RE = re.compile(
    r"(?P<model>[A-Za-z0-9][\w\s+\-]{2,40}?)\s+(?P<val>\d{1,2})\s*(?:pax|passengers?|seats?)",
    re.I,
)


@dataclass
class AircraftAuthorityRecord:
    canonical_name: str
    manufacturer: str
    aircraft_category: str
    passenger_capacity_min: int
    passenger_capacity_max: int
    nbaa_range_nm: float
    max_cruise_speed: Optional[float]
    cabin_height: Optional[float]
    cabin_width: Optional[float]
    cabin_length: Optional[float]
    takeoff_distance_ft: Optional[int]
    production_start_year: Optional[int]
    production_end_year: Optional[int]
    current_in_production: bool
    direct_competitors: List[str] = field(default_factory=list)
    replacement_models: List[str] = field(default_factory=list)
    authority_source: str = "catalog_verified"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "manufacturer": self.manufacturer,
            "aircraft_category": self.aircraft_category,
            "passenger_capacity_min": self.passenger_capacity_min,
            "passenger_capacity_max": self.passenger_capacity_max,
            "nbaa_range_nm": self.nbaa_range_nm,
            "max_cruise_speed": self.max_cruise_speed,
            "cabin_height": self.cabin_height,
            "cabin_width": self.cabin_width,
            "cabin_length": self.cabin_length,
            "takeoff_distance_ft": self.takeoff_distance_ft,
            "production_start_year": self.production_start_year,
            "production_end_year": self.production_end_year,
            "current_in_production": self.current_in_production,
            "direct_competitors": list(self.direct_competitors),
            "replacement_models": list(self.replacement_models),
            "authority_source": self.authority_source,
            "confidence": self.confidence,
        }

    def to_profile_dict(self) -> Dict[str, Any]:
        """Mission/comparison-compatible profile dict from authority record."""
        practical = float(self.nbaa_range_nm or 0)
        brochure = round(practical / 0.88, 1) if practical > 0 else 0.0
        return {
            "category": self.aircraft_category,
            "brochure_nm": brochure,
            "practical_nm": practical,
            "pax_typical": self.passenger_capacity_min,
            "pax_max_long_range": self.passenger_capacity_max,
            "runway_ft": 5000,
            "hot_high_score": 0.65,
            "short_field_score": 0.55,
            "operating_index": 0.65,
            "cabin_score": 0.8,
            "baggage_score": 0.75,
            "dispatch_score": 0.82,
            "resale_score": 0.78,
            "pilot_workload": 0.6,
            "ownership_efficiency": 0.62,
            "_aircraft_authority": True,
            "_authority_source": self.authority_source,
        }


@dataclass
class ClaimValidationResult:
    valid: bool
    reason: str = ""
    canonical_name: str = ""
    claimed_value: Optional[float] = None
    authoritative_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "canonical_name": self.canonical_name,
            "claimed_value": self.claimed_value,
            "authoritative_value": self.authoritative_value,
        }


def _normalize_alias_key(raw: str) -> str:
    spoken = re.sub(r"[^\w\s+]", " ", (raw or "").lower())
    return re.sub(r"\s+", " ", spoken).strip()


def resolve_aircraft_alias(
    raw: str,
    *,
    manufacturer: Optional[str] = None,
) -> str:
    """Resolve broker alias to canonical aircraft name."""
    token = (raw or "").strip()
    if not token:
        return ""
    key = _normalize_alias_key(token)
    if key in _AKAL_ALIAS_MAP:
        return _AKAL_ALIAS_MAP[key]
    if manufacturer:
        combined = _normalize_alias_key(f"{manufacturer} {token}")
        if combined in _AKAL_ALIAS_MAP:
            return _AKAL_ALIAS_MAP[combined]
    try:
        from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

        return resolve_canonical_display_name(token) or token
    except Exception:
        return token


def _profile_lookup_key(canonical: str) -> str:
    return _PROFILE_LOOKUP_KEY.get(canonical, canonical)


def _category_label(raw: str) -> str:
    mapping = {
        "light": "light",
        "super-midsize": "super-midsize",
        "super_mid": "super-midsize",
        "large": "large-cabin",
        "large_cabin": "large-cabin",
        "ultra-long": "ultra-long",
        "ultra_long": "ultra-long",
        "turboprop": "turboprop",
    }
    return mapping.get((raw or "").strip().lower(), raw or "large-cabin")


def get_aircraft_authority_record(
    *,
    aircraft_model: str = "",
    alias: str = "",
    manufacturer: Optional[str] = None,
    db: Any = None,
) -> Optional[AircraftAuthorityRecord]:
    """Return authoritative aircraft record or None when unverified."""
    raw = (aircraft_model or alias or "").strip()
    if not raw:
        return None

    canonical = resolve_aircraft_alias(raw, manufacturer=manufacturer)
    if not canonical:
        return None

    lookup_key = _profile_lookup_key(canonical)
    spec = None
    source = "unverified"
    confidence = 0.5

    try:
        from services.data_authority.aircraft_spec_repository import get_verified_spec

        spec = get_verified_spec(lookup_key, db=db)
        if spec is None and lookup_key != canonical:
            spec = get_verified_spec(canonical, db=db)
        if spec is not None:
            source = spec.source
            confidence = 1.0 if spec.verified else 0.85
    except Exception:
        spec = None

    if spec is None:
        from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

        prof = AIRCRAFT_PROFILES.get(lookup_key) or AIRCRAFT_PROFILES.get(canonical)
        if not prof:
            return None
        source = "curated_catalog"
        confidence = 0.85
        pax_typ = int(prof.get("pax_typical") or 8)
        pax_max = int(prof.get("pax_max_long_range") or pax_typ)
        nbaa = float(prof.get("practical_nm") or prof.get("brochure_nm") or 0)
        category = _category_label(str(prof.get("category") or ""))
    else:
        pax_typ = int(spec.pax_typical)
        pax_max = int(spec.pax_max_long_range)
        nbaa = float(spec.practical_nm)
        category = _category_label(spec.category)

    enrich = _STATIC_ENRICHMENT.get(canonical, {})
    takeoff_ft = int(enrich.get("takeoff_distance_ft") or 0)
    if not takeoff_ft:
        try:
            from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

            prof_runway = (AIRCRAFT_PROFILES.get(lookup_key) or AIRCRAFT_PROFILES.get(canonical) or {})
            takeoff_ft = int(prof_runway.get("runway_ft") or 0)
        except Exception:
            takeoff_ft = 0
    mfr = str(enrich.get("manufacturer") or _MANUFACTURER_BY_CANONICAL.get(canonical, ""))
    if not mfr and " " in canonical:
        mfr = canonical.split(None, 1)[0]

    low_key = canonical.lower()
    competitors = list(_COMPETITORS.get(low_key, []))
    replacements = list(_REPLACEMENTS.get(low_key, []))

    return AircraftAuthorityRecord(
        canonical_name=canonical,
        manufacturer=mfr,
        aircraft_category=category,
        passenger_capacity_min=int(enrich.get("passenger_capacity_min") or pax_typ),
        passenger_capacity_max=int(enrich.get("passenger_capacity_max") or pax_max),
        nbaa_range_nm=nbaa,
        max_cruise_speed=enrich.get("max_cruise_speed"),
        cabin_height=enrich.get("cabin_height_ft"),
        cabin_width=enrich.get("cabin_width_ft"),
        cabin_length=enrich.get("cabin_length_ft"),
        takeoff_distance_ft=takeoff_ft or None,
        production_start_year=enrich.get("production_start_year"),
        production_end_year=enrich.get("production_end_year"),
        current_in_production=bool(enrich.get("current_in_production", True)),
        direct_competitors=competitors,
        replacement_models=replacements,
        authority_source=source,
        confidence=confidence,
    )


def get_authority_profile_dict(model: str, *, db: Any = None) -> Optional[Dict[str, Any]]:
    """Profile dict for mission/comparison engines from authority layer."""
    rec = get_aircraft_authority_record(aircraft_model=model, db=db)
    if rec is None:
        return None
    prof = rec.to_profile_dict()
    try:
        from services.data_authority.aircraft_spec_repository import get_verified_spec

        spec = get_verified_spec(_profile_lookup_key(rec.canonical_name), db=db)
        if spec is not None:
            prof.update(
                {
                    k: v
                    for k, v in spec.to_profile_dict().items()
                    if k not in ("_data_authority_source", "_data_authority_verified")
                }
            )
    except Exception:
        pass
    return prof


def validate_aircraft_claim(claim: str, *, db: Any = None) -> ClaimValidationResult:
    """Validate a factual claim against authoritative aircraft data."""
    text = (claim or "").strip()
    if not text:
        return ClaimValidationResult(valid=False, reason="empty_claim")

    m = _CLAIM_RANGE_RE.search(text)
    if m:
        model_raw = m.group("model").strip()
        claimed = float(m.group("val"))
        rec = get_aircraft_authority_record(aircraft_model=model_raw, db=db)
        if rec is None:
            return ClaimValidationResult(
                valid=False,
                reason="unknown_aircraft",
                canonical_name=model_raw,
                claimed_value=claimed,
            )
        auth = rec.nbaa_range_nm
        tolerance = max(200.0, auth * 0.12)
        if abs(claimed - auth) > tolerance:
            return ClaimValidationResult(
                valid=False,
                reason="range_mismatch",
                canonical_name=rec.canonical_name,
                claimed_value=claimed,
                authoritative_value=auth,
            )
        return ClaimValidationResult(
            valid=True,
            reason="range_match",
            canonical_name=rec.canonical_name,
            claimed_value=claimed,
            authoritative_value=auth,
        )

    m = _CLAIM_PAX_RE.search(text)
    if m:
        model_raw = m.group("model").strip()
        claimed = int(m.group("val"))
        rec = get_aircraft_authority_record(aircraft_model=model_raw, db=db)
        if rec is None:
            return ClaimValidationResult(valid=False, reason="unknown_aircraft", canonical_name=model_raw)
        if claimed > rec.passenger_capacity_max:
            return ClaimValidationResult(
                valid=False,
                reason="passenger_capacity_exceeded",
                canonical_name=rec.canonical_name,
                claimed_value=float(claimed),
                authoritative_value=float(rec.passenger_capacity_max),
            )
        return ClaimValidationResult(valid=True, canonical_name=rec.canonical_name)

    if _detect_hallucinated_token(text):
        return ClaimValidationResult(valid=False, reason="hallucinated_model")

    return ClaimValidationResult(valid=True, reason="no_parseable_numeric_claim")


def _detect_hallucinated_token(text: str) -> bool:
    low = text.lower()
    for fake in ("gulfstream g750", "falcon 9x", "citation longitude x"):
        if fake in low:
            return True
    return False


def build_authoritative_comparison_dataset(
    aircraft: Sequence[str],
    *,
    db: Any = None,
) -> Dict[str, Any]:
    """Build normalized comparison table from authority records only."""
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw in aircraft:
        token = str(raw or "").strip()
        if not token:
            continue
        rec = get_aircraft_authority_record(aircraft_model=token, db=db)
        if rec is None:
            missing.append(token)
            continue
        cabin = "—"
        if rec.cabin_height and rec.cabin_width:
            cabin = f"{rec.cabin_height:.1f} ft H × {rec.cabin_width:.1f} ft W"
        rows.append(
            {
                "canonical_name": rec.canonical_name,
                "manufacturer": rec.manufacturer,
                "category": rec.aircraft_category,
                "range_nm": rec.nbaa_range_nm,
                "speed_ktas": rec.max_cruise_speed,
                "passenger_min": rec.passenger_capacity_min,
                "passenger_max": rec.passenger_capacity_max,
                "cabin": cabin,
                "baggage": rec.to_profile_dict().get("baggage_score"),
                "authority_source": rec.authority_source,
                "confidence": rec.confidence,
            }
        )

    status = "OK" if len(rows) >= 2 and not missing else "INSUFFICIENT_DATA"
    return {
        "status": status,
        "aircraft": rows,
        "missing": missing,
        "authority": "aircraft_knowledge_authority",
    }


def build_authoritative_market_context(
    *,
    year: Optional[int] = None,
    model: str = "",
    ask_usd: Optional[float] = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Authoritative market bands for buy-decision context."""
    rec = get_aircraft_authority_record(aircraft_model=model, db=db)
    if rec is None:
        return {"status": "INSUFFICIENT_DATA", "canonical_name": model}

    avg_price: Optional[float] = None
    try:
        from services.data_authority.aircraft_spec_repository import get_verified_spec

        spec = get_verified_spec(_profile_lookup_key(rec.canonical_name), db=db)
        if spec and spec.average_pre_owned_price:
            avg_price = float(spec.average_pre_owned_price)
    except Exception:
        pass

    # Class-based band when DB price absent
    if avg_price is None:
        cat = rec.aircraft_category
        band_mid = {
            "light": 6_000_000.0,
            "super-midsize": 18_000_000.0,
            "large-cabin": 28_000_000.0,
            "ultra-long": 45_000_000.0,
        }.get(cat, 20_000_000.0)
        avg_price = band_mid

    low = avg_price * 0.75
    high = avg_price * 1.35
    age_years = None
    age_position = "unknown"
    if year:
        from datetime import datetime

        age_years = max(0, datetime.now().year - int(year))
        if age_years <= 5:
            age_position = "young"
        elif age_years <= 12:
            age_position = "mid_life"
        else:
            age_position = "mature"

    depreciation_band = "moderate"
    if age_years is not None and age_years > 10:
        depreciation_band = "elevated"

    ask_position = "unknown"
    if ask_usd is not None and ask_usd > 0:
        if ask_usd < low:
            ask_position = "below_market"
        elif ask_usd > high:
            ask_position = "above_market"
        else:
            ask_position = "in_band"

    return {
        "status": "OK",
        "canonical_name": rec.canonical_name,
        "expected_market_band_usd": {"low": low, "mid": avg_price, "high": high},
        "depreciation_band": depreciation_band,
        "age_position": age_position,
        "ask_position": ask_position,
        "authority_source": rec.authority_source,
        "aircraft_category": rec.aircraft_category,
    }


__all__ = [
    "AircraftAuthorityRecord",
    "ClaimValidationResult",
    "build_authoritative_comparison_dataset",
    "build_authoritative_market_context",
    "get_aircraft_authority_record",
    "get_authority_profile_dict",
    "resolve_aircraft_alias",
    "validate_aircraft_claim",
]

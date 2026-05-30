"""
Aircraft specification repository — PostgreSQL-first, fail-closed.

Authoritative order:
  1. ``aviacost_aircraft_details`` (PostgreSQL)
  2. Same-key operational enrichments from curated catalog (runway, hot/high) — never a different model
  3. Curated in-memory profile only when ``DATA_AUTHORITY_STRICT`` is off (local/tests)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

INSUFFICIENT_VERIFIED_AIRCRAFT_DATA = "INSUFFICIENT VERIFIED AIRCRAFT DATA"
INSUFFICIENT_VERIFIED_COMPARISON = "INSUFFICIENT VERIFIED DATA FOR STRUCTURED COMPARISON"

_CATEGORY_MAP = {
    "light jet": "light",
    "light": "light",
    "super mid": "super-midsize",
    "super midsize": "super-midsize",
    "super-mid": "super-midsize",
    "midsize": "super-midsize",
    "large cabin": "large-cabin",
    "large": "large-cabin",
    "heavy": "large-cabin",
    "ultra long range": "ultra-long",
    "ultra-long range": "ultra-long",
    "ulr": "ultra-long",
}


@dataclass(frozen=True)
class VerifiedAircraftSpec:
    canonical_name: str
    source: str  # postgres_aviacost | curated_fallback
    category: str
    practical_nm: float
    brochure_nm: float
    pax_typical: int
    pax_max_long_range: int
    runway_ft: int
    hot_high_score: float
    short_field_score: float
    operating_index: float
    cabin_score: float
    baggage_score: float
    dispatch_score: float
    resale_score: float
    pilot_workload: float
    ownership_efficiency: float
    variable_cost_per_hour: Optional[float] = None
    average_pre_owned_price: Optional[float] = None
    postgres_name: str = ""
    verified: bool = True

    def to_profile_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "brochure_nm": self.brochure_nm,
            "practical_nm": self.practical_nm,
            "pax_typical": self.pax_typical,
            "pax_max_long_range": self.pax_max_long_range,
            "runway_ft": self.runway_ft,
            "hot_high_score": self.hot_high_score,
            "short_field_score": self.short_field_score,
            "operating_index": self.operating_index,
            "cabin_score": self.cabin_score,
            "baggage_score": self.baggage_score,
            "dispatch_score": self.dispatch_score,
            "resale_score": self.resale_score,
            "pilot_workload": self.pilot_workload,
            "ownership_efficiency": self.ownership_efficiency,
            "_data_authority_source": self.source,
            "_data_authority_verified": self.verified,
        }


def _strict_mode() -> bool:
    return (os.getenv("DATA_AUTHORITY_STRICT") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _get_db():
    try:
        from api.main import get_db

        return get_db()
    except Exception:
        try:
            from config.config_loader import Config
            from database.postgres_client import PostgresClient

            cfg = Config.from_env()
            cs = cfg.postgres_connection_string
            if not cs:
                return None
            db = PostgresClient(cs)
            db.execute_query("SELECT 1")
            return db
        except Exception as exc:
            logger.debug("aircraft_spec_repository: no postgres (%s)", exc)
            return None


def _map_category(raw: Optional[str]) -> str:
    key = re.sub(r"\s+", " ", (raw or "").strip().lower())
    if key in _CATEGORY_MAP:
        return _CATEGORY_MAP[key]
    if "ultra" in key:
        return "ultra-long"
    if "super" in key:
        return "super-midsize"
    if "light" in key:
        return "light"
    return "large-cabin"


def _lookup_aviacost_row(db: Any, canonical: str) -> Optional[Dict[str, Any]]:
    from services.aviacost_lookup import lookup_aviacost

    parts = canonical.split(None, 1)
    if len(parts) == 2:
        mfr, mdl = parts[0], parts[1]
    else:
        mfr, mdl = "", canonical
    row = lookup_aviacost(db, manufacturer=mfr, model=mdl)
    if row:
        return row
    return lookup_aviacost(db, model=canonical)


def _curated_profile(canonical: str) -> Optional[Dict[str, Any]]:
    from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

    if canonical in AIRCRAFT_PROFILES:
        return dict(AIRCRAFT_PROFILES[canonical])
    low = canonical.lower()
    for name, spec in AIRCRAFT_PROFILES.items():
        if name.lower() == low:
            return dict(spec)
    return None


def _spec_from_postgres(canonical: str, row: Dict[str, Any]) -> VerifiedAircraftSpec:
    curated = _curated_profile(canonical) or {}
    range_nm = float(row.get("seats_full_range_nm") or 0)
    practical = range_nm * 0.88 if range_nm > 0 else float(curated.get("practical_nm") or 0)
    brochure = range_nm if range_nm > 0 else float(curated.get("brochure_nm") or practical)
    pax = int(row.get("typical_passenger_capacity_max") or curated.get("pax_typical") or 8)
    var_cost = row.get("variable_cost_per_hour")
    preowned = row.get("average_pre_owned_price")
    oi = float(curated.get("operating_index") or 0.65)
    if var_cost is not None:
        try:
            v = float(var_cost)
            oi = min(0.95, max(0.25, v / 12_000.0))
        except (TypeError, ValueError):
            pass
    return VerifiedAircraftSpec(
        canonical_name=canonical,
        source="postgres_aviacost",
        category=_map_category(row.get("category_name")) or str(curated.get("category") or "large-cabin"),
        practical_nm=practical,
        brochure_nm=brochure,
        pax_typical=pax,
        pax_max_long_range=int(curated.get("pax_max_long_range") or pax),
        runway_ft=int(curated.get("runway_ft") or 5000),
        hot_high_score=float(curated.get("hot_high_score") or 0.65),
        short_field_score=float(curated.get("short_field_score") or 0.55),
        operating_index=oi,
        cabin_score=float(curated.get("cabin_score") or 0.75),
        baggage_score=float(curated.get("baggage_score") or 0.7),
        dispatch_score=float(curated.get("dispatch_score") or 0.8),
        resale_score=float(curated.get("resale_score") or 0.75),
        pilot_workload=float(curated.get("pilot_workload") or 0.55),
        ownership_efficiency=float(curated.get("ownership_efficiency") or 0.6),
        variable_cost_per_hour=float(var_cost) if var_cost is not None else None,
        average_pre_owned_price=float(preowned) if preowned is not None else None,
        postgres_name=str(row.get("name") or canonical),
        verified=True,
    )


def _spec_from_curated(canonical: str, prof: Dict[str, Any]) -> VerifiedAircraftSpec:
    return VerifiedAircraftSpec(
        canonical_name=canonical,
        source="curated_fallback",
        category=str(prof.get("category") or "large-cabin"),
        practical_nm=float(prof.get("practical_nm") or prof.get("range_nm") or 0),
        brochure_nm=float(prof.get("brochure_nm") or prof.get("practical_nm") or 0),
        pax_typical=int(prof.get("pax_typical") or 8),
        pax_max_long_range=int(prof.get("pax_max_long_range") or prof.get("pax_typical") or 8),
        runway_ft=int(prof.get("runway_ft") or 5000),
        hot_high_score=float(prof.get("hot_high_score") or 0.65),
        short_field_score=float(prof.get("short_field_score") or 0.55),
        operating_index=float(prof.get("operating_index") or 0.65),
        cabin_score=float(prof.get("cabin_score") or 0.75),
        baggage_score=float(prof.get("baggage_score") or 0.7),
        dispatch_score=float(prof.get("dispatch_score") or 0.8),
        resale_score=float(prof.get("resale_score") or 0.75),
        pilot_workload=float(prof.get("pilot_workload") or 0.55),
        ownership_efficiency=float(prof.get("ownership_efficiency") or 0.6),
        verified=False,
    )


def get_verified_spec(
    model: str,
    *,
    db: Any = None,
) -> Optional[VerifiedAircraftSpec]:
    """
    Resolve aircraft specifications from PostgreSQL first; fail closed when strict.
    Never substitutes a different model (no G500→G650 bridge).
    """
    from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

    canonical = resolve_canonical_display_name(model)
    if not canonical:
        return None

    connection = db if db is not None else _get_db()
    if connection is not None:
        try:
            row = _lookup_aviacost_row(connection, canonical)
            if row:
                return _spec_from_postgres(canonical, row)
        except Exception as exc:
            logger.warning("aviacost spec lookup failed for %s: %s", canonical, exc)

    if _strict_mode():
        return None

    prof = _curated_profile(canonical)
    if prof:
        return _spec_from_curated(canonical, prof)
    return None


def get_verified_spec_profile(model: str, *, db: Any = None) -> Optional[Dict[str, Any]]:
    spec = get_verified_spec(model, db=db)
    return spec.to_profile_dict() if spec else None


def require_verified_specs(
    models: Sequence[str],
    *,
    db: Any = None,
) -> Tuple[List[VerifiedAircraftSpec], List[str]]:
    """Return (verified, missing_canonical_names)."""
    verified: List[VerifiedAircraftSpec] = []
    missing: List[str] = []
    for raw in models:
        token = (raw or "").strip()
        if not token:
            continue
        spec = get_verified_spec(token, db=db)
        if spec is None:
            from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

            missing.append(resolve_canonical_display_name(token) or token)
        else:
            verified.append(spec)
    return verified, missing


def list_verified_model_keys(*, db: Any = None) -> List[str]:
    """Canonical names with verified postgres or (non-strict) curated profiles."""
    from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

    keys: List[str] = []
    for name in AIRCRAFT_PROFILES:
        if get_verified_spec(name, db=db) is not None:
            keys.append(name)
    return keys


__all__ = [
    "INSUFFICIENT_VERIFIED_AIRCRAFT_DATA",
    "INSUFFICIENT_VERIFIED_COMPARISON",
    "VerifiedAircraftSpec",
    "get_verified_spec",
    "get_verified_spec_profile",
    "require_verified_specs",
    "list_verified_model_keys",
]

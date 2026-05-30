#!/usr/bin/env python3
"""Read-only aircraft catalog audit — AIRCRAFT_PROFILES vs PostgreSQL."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from services.catalog.catalog_alias_resolver import (  # noqa: E402
    _DISPLAY_ALIASES,
    resolve_canonical_display_name,
)
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES  # noqa: E402

VALID_CATEGORIES = frozenset(
    {"light", "super-midsize", "large", "large-cabin", "ultra-long", "turboprop"}
)

BENCHMARK: Dict[str, List[str]] = {
    "Ultra Long Range": [
        "Global 7500",
        "Global 6500",
        "Global 6000",
        "Gulfstream G700",
        "Gulfstream G650ER",
        "Gulfstream G650",
        "Gulfstream G600",
        "Gulfstream G500",
        "Falcon 8X",
        "Falcon 7X",
    ],
    "Large Cabin": ["Challenger 650", "Falcon 2000LXS"],
    "Super Mid": [
        "Praetor 600",
        "Praetor 500",
        "Challenger 3500",
        "Citation Longitude",
        "Citation Latitude",
    ],
    "Light / Utility": ["PC-24", "PC-12 NGX", "King Air 360"],
}


def _op_band(oi: float) -> str:
    if oi <= 0:
        return "unknown"
    if oi <= 0.55:
        return "low"
    if oi <= 0.72:
        return "medium"
    if oi <= 0.88:
        return "high"
    return "ultra"


def _acq_band_curated(category: str) -> str:
    mid = {
        "light": 8,
        "super-midsize": 18,
        "large": 28,
        "large-cabin": 28,
        "ultra-long": 45,
        "turboprop": 5,
    }.get(category, 20)
    return f"~${mid}M (curated est.)"


def _acq_band_pg(preowned: Optional[float]) -> str:
    if not preowned:
        return "unknown"
    m = float(preowned) / 1_000_000
    if m < 12:
        return "<$12M"
    if m < 25:
        return "$12-25M"
    if m < 45:
        return "$25-45M"
    return ">$45M"


def _get_db():
    try:
        from config.config_loader import Config
        from database.postgres_client import PostgresClient

        cfg = Config.from_env()
        cs = cfg.postgres_connection_string
        if not cs:
            return None, "POSTGRES_CONNECTION_STRING not set"
        db = PostgresClient(cs)
        db.execute_query("SELECT 1")
        return db, None
    except Exception as exc:
        return None, str(exc)


def _pg_lookup(db: Any, name: str) -> Optional[Dict[str, Any]]:
    from services.aviacost_lookup import lookup_aviacost

    canon = resolve_canonical_display_name(name)
    parts = canon.split(None, 1)
    if len(parts) == 2:
        row = lookup_aviacost(db, manufacturer=parts[0], model=parts[1])
        if row:
            return row
    return lookup_aviacost(db, model=canon) or lookup_aviacost(db, model=name)


def main() -> int:
    db, pg_err = _get_db()

    profiles_rows: List[Dict[str, Any]] = []
    for name, spec in sorted(AIRCRAFT_PROFILES.items()):
        cat = str(spec.get("category") or "")
        oi = float(spec.get("operating_index") or 0)
        profiles_rows.append(
            {
                "aircraft_name": name,
                "category": cat,
                "range_nm": spec.get("practical_nm"),
                "brochure_nm": spec.get("brochure_nm"),
                "seats": spec.get("pax_typical"),
                "seats_max": spec.get("pax_max_long_range"),
                "acquisition_band": _acq_band_curated(cat),
                "operating_cost_band": _op_band(oi),
                "in_aircraft_profiles": True,
            }
        )

    pg_rows: List[Dict[str, Any]] = []
    if db:
        raw = db.execute_query(
            """
            SELECT name, manufacturer_name, category_name,
                   seats_full_range_nm, typical_passenger_capacity_max,
                   variable_cost_per_hour, average_pre_owned_price
            FROM aviacost_aircraft_details
            ORDER BY name
            """
        )
        for r in raw:
            vch = r.get("variable_cost_per_hour")
            pre = r.get("average_pre_owned_price")
            oi_est = min(0.95, max(0.25, float(vch) / 12000.0)) if vch else 0.0
            pg_rows.append(
                {
                    "aircraft_name": r.get("name"),
                    "manufacturer": r.get("manufacturer_name"),
                    "category": r.get("category_name"),
                    "range_nm": float(r["seats_full_range_nm"])
                    if r.get("seats_full_range_nm") is not None
                    else None,
                    "seats": r.get("typical_passenger_capacity_max"),
                    "acquisition_band": _acq_band_pg(
                        float(pre) if pre is not None else None
                    ),
                    "operating_cost_band": _op_band(oi_est) if vch else "unknown",
                    "variable_cost_per_hour": vch,
                    "average_pre_owned_usd": pre,
                    "in_postgres_aviacost": True,
                }
            )

    # Unified coverage for benchmark + profiles
    coverage: List[Dict[str, Any]] = []
    seen: set = set()

    def add_row(
        name: str,
        *,
        category: str = "",
        range_nm: Any = None,
        seats: Any = None,
        acq: str = "",
        op: str = "",
        in_prof: bool = False,
        in_pg: bool = False,
    ) -> None:
        key = name.lower().strip()
        if key in seen:
            return
        seen.add(key)
        coverage.append(
            {
                "aircraft_name": name,
                "category": category or "—",
                "range_nm": range_nm if range_nm is not None else "—",
                "seats": seats if seats is not None else "—",
                "acquisition_band": acq or "—",
                "operating_cost_band": op or "—",
                "in_aircraft_profiles": in_prof,
                "in_postgres_aviacost": in_pg,
            }
        )

    for row in profiles_rows:
        pg = _pg_lookup(db, row["aircraft_name"]) if db else None
        add_row(
            row["aircraft_name"],
            category=row["category"],
            range_nm=row["range_nm"],
            seats=row["seats"],
            acq=row["acquisition_band"],
            op=row["operating_cost_band"],
            in_prof=True,
            in_pg=bool(pg),
        )

    for row in pg_rows:
        nm = str(row["aircraft_name"] or "")
        if nm.lower() in seen:
            continue
        add_row(
            nm,
            category=str(row.get("category") or ""),
            range_nm=row.get("range_nm"),
            seats=row.get("seats"),
            acq=row.get("acquisition_band", ""),
            op=row.get("operating_cost_band", ""),
            in_prof=False,
            in_pg=True,
        )

    # Gap analysis
    missing: List[Dict[str, Any]] = []
    incomplete_bench: List[Dict[str, Any]] = []
    curated_only: List[Dict[str, Any]] = []
    postgres_only: List[Dict[str, Any]] = []
    alias_collisions: List[Dict[str, Any]] = []

    for seg, models in BENCHMARK.items():
        for m in models:
            canon = resolve_canonical_display_name(m)
            in_prof = canon in AIRCRAFT_PROFILES
            pg = _pg_lookup(db, m) if db else None
            in_pg = pg is not None
            ver_src = None
            if db:
                try:
                    from services.data_authority.aircraft_spec_repository import (
                        get_verified_spec,
                    )

                    vs = get_verified_spec(m, db=db)
                    ver_src = vs.source if vs else None
                except Exception:
                    pass

            rec = {
                "segment": seg,
                "requested_name": m,
                "canonical_name": canon,
                "in_aircraft_profiles": in_prof,
                "in_postgres_aviacost": in_pg,
                "postgres_match_name": pg.get("name") if pg else None,
                "verified_source": ver_src,
            }
            if not in_prof and not in_pg:
                missing.append(rec)
            elif in_prof and not in_pg:
                curated_only.append(rec)
            elif not in_prof and in_pg:
                postgres_only.append(rec)

            if in_prof:
                sp = AIRCRAFT_PROFILES[canon]
                if not sp.get("practical_nm") or not sp.get("pax_typical"):
                    incomplete_bench.append(rec)

    for alias, target in sorted(_DISPLAY_ALIASES.items()):
        issues = []
        if target not in AIRCRAFT_PROFILES:
            issues.append("target_missing_from_profiles")
        if "3500" in alias and target == "Challenger 350":
            issues.append("marketing_name_maps_to_different_profile_key")
        if "6000" in alias and target == "Bombardier Global 6000":
            issues.append("alias_target_not_in_profiles")
        if "g600" in alias and target == "Gulfstream G600":
            issues.append("alias_target_not_in_profiles")
        if "g700" in alias:
            issues.append("no_profile_for_g700")
        if issues:
            alias_collisions.append(
                {"alias": alias, "canonical_target": target, "issues": issues}
            )

    invalid_categories = [
        {"aircraft_name": n, "category": AIRCRAFT_PROFILES[n].get("category")}
        for n in AIRCRAFT_PROFILES
        if str(AIRCRAFT_PROFILES[n].get("category") or "") not in VALID_CATEGORIES
    ]

    incomplete_profiles = [
        {
            "aircraft_name": n,
            "missing_fields": [
                f
                for f in (
                    "category",
                    "practical_nm",
                    "pax_typical",
                    "operating_index",
                )
                if not AIRCRAFT_PROFILES[n].get(f)
            ],
        }
        for n in AIRCRAFT_PROFILES
        if any(not AIRCRAFT_PROFILES[n].get(f) for f in ("practical_nm", "pax_typical", "category"))
    ]

    # Postgres rows that fuzzy-match multiple benchmark types
    pg_name_dupes: List[Dict[str, Any]] = []
    by_lower: Dict[str, List[str]] = defaultdict(list)
    for r in pg_rows:
        by_lower[str(r["aircraft_name"] or "").lower()].append(str(r["aircraft_name"]))
    for k, vals in by_lower.items():
        if len(vals) > 1:
            pg_name_dupes.append({"normalized": k, "variants": vals})

    report = {
        "summary": {
            "aircraft_profiles_count": len(AIRCRAFT_PROFILES),
            "postgres_aviacost_count": len(pg_rows),
            "postgres_connection": "ok" if db else pg_err,
            "display_aliases_count": len(_DISPLAY_ALIASES),
            "benchmark_models_checked": sum(len(v) for v in BENCHMARK.values()),
        },
        "coverage_report": coverage,
        "aircraft_profiles_detail": profiles_rows,
        "postgres_aviacost_sample_count": len(pg_rows),
        "gap_report": {
            "missing_aircraft": missing,
            "curated_only_no_postgres": curated_only,
            "postgres_only_no_profile": postgres_only,
            "incomplete_benchmark_entries": incomplete_bench,
            "duplicate_aliases": alias_collisions,
            "invalid_categories": invalid_categories,
            "incomplete_profiles": incomplete_profiles,
            "postgres_duplicate_names": pg_name_dupes[:20],
        },
    }

    out_path = _ROOT / "evals" / "catalog_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

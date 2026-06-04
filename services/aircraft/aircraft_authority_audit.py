"""
Aircraft authority catalog audit — duplicate aliases, conflicts, missing fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from services.aircraft.aircraft_authority_service import (
    _AKAL_ALIAS_MAP,
    get_aircraft_authority_record,
    resolve_aircraft_alias,
)


@dataclass
class AuthorityAuditReport:
    duplicate_aliases: List[str] = field(default_factory=list)
    conflicting_ranges: List[Dict[str, Any]] = field(default_factory=list)
    missing_cabin_dimensions: List[str] = field(default_factory=list)
    missing_competitor_mappings: List[str] = field(default_factory=list)
    unresolved_canonical_names: List[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "duplicate_aliases": list(self.duplicate_aliases),
            "conflicting_ranges": list(self.conflicting_ranges),
            "missing_cabin_dimensions": list(self.missing_cabin_dimensions),
            "missing_competitor_mappings": list(self.missing_competitor_mappings),
            "unresolved_canonical_names": list(self.unresolved_canonical_names),
        }


def run_aircraft_authority_audit(
    *,
    catalog_models: List[str] | None = None,
) -> AuthorityAuditReport:
    """Audit AKAL alias map and catalog coverage."""
    from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

    models = catalog_models or sorted(AIRCRAFT_PROFILES.keys())
    report = AuthorityAuditReport()

    # Duplicate alias targets with conflicting keys
    target_to_keys: Dict[str, List[str]] = {}
    for alias, canonical in _AKAL_ALIAS_MAP.items():
        target_to_keys.setdefault(canonical, []).append(alias)
    for canonical, keys in target_to_keys.items():
        if len(keys) > 8:
            report.duplicate_aliases.append(f"{canonical}: {len(keys)} aliases")

    # Reverse alias collision
    seen_targets: Dict[str, str] = {}
    for alias, canonical in _AKAL_ALIAS_MAP.items():
        if alias in seen_targets and seen_targets[alias] != canonical:
            report.duplicate_aliases.append(f"alias_collision:{alias}")
        seen_targets[alias] = canonical

    canonical_ranges: Dict[str, float] = {}
    for model in models:
        rec = get_aircraft_authority_record(aircraft_model=model)
        if rec is None:
            report.unresolved_canonical_names.append(model)
            continue
        if rec.nbaa_range_nm <= 0:
            report.conflicting_ranges.append(
                {"canonical_name": rec.canonical_name, "issue": "zero_range"}
            )
        prev = canonical_ranges.get(rec.canonical_name)
        if prev is not None and abs(prev - rec.nbaa_range_nm) > 1:
            report.conflicting_ranges.append(
                {
                    "canonical_name": rec.canonical_name,
                    "issue": "range_drift",
                    "values": [prev, rec.nbaa_range_nm],
                }
            )
        canonical_ranges[rec.canonical_name] = rec.nbaa_range_nm

        if not (rec.cabin_height and rec.cabin_width and rec.cabin_length):
            report.missing_cabin_dimensions.append(rec.canonical_name)

        if not rec.direct_competitors and rec.aircraft_category in (
            "super-midsize",
            "large-cabin",
            "ultra-long",
        ):
            report.missing_competitor_mappings.append(rec.canonical_name)

    # Required AKAL aliases must resolve
    required = ("longitude", "g280", "cj3+", "falcon 8x", "challenger 3500")
    for alias in required:
        resolved = resolve_aircraft_alias(alias)
        if not resolved:
            report.unresolved_canonical_names.append(f"alias:{alias}")

    report.ok = not (
        report.conflicting_ranges
        or report.unresolved_canonical_names
        or report.duplicate_aliases
    )
    return report


__all__ = ["AuthorityAuditReport", "run_aircraft_authority_audit"]

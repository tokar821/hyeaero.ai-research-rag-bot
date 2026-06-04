"""Phase 32 — Hallucination audit (catalog truth validation)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from tests.production_validation.validation_runner import ValidationResult

_FAKE_PATTERNS = re.compile(
    r"\b(?:FakeJet|UnknownJet|HyperJet|NotAReal|ZZZ-999|placeholder)\b",
    re.I,
)
_RANGE_CLAIM = re.compile(r"\b(\d{3,5})\s*(?:nm|nautical)\b", re.I)
_PRICE_CLAIM = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:M|MM|million)\b", re.I)


def _catalog_models() -> set[str]:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    tokens = [
        "G650", "Falcon 8X", "Global 7500", "Longitude", "Challenger 3500",
        "Citation CJ3+", "Citation CJ4", "Praetor 600", "PC-24", "G550", "G280",
        "Citation Latitude", "Challenger 350", "G700", "Falcon 7X",
    ]
    return {resolve_aircraft_alias(t).lower() for t in tokens}


def _check_cross_model_substitution(models: List[str]) -> bool:
    """Return True if a known bad substitution pattern is detected."""
    lowered = [m.lower() for m in models]
    pairs_bad = [
        ("citation longitude", "challenger longitude"),
        ("challenger 3500", "challenger 350"),
        ("citation cj3+", "citation cj4"),
    ]
    for a, b in pairs_bad:
        if a in lowered and b in lowered and a != b:
            if ("challenger 3500" in lowered and "challenger 350" in lowered):
                return True
    return False


def audit_single(result: "ValidationResult", answer: str = "") -> List[str]:
    flags: List[str] = []
    text = answer or result.answer_preview or ""

    if _FAKE_PATTERNS.search(text):
        flags.append("fake_aircraft_reference")

    for model in result.authority_models:
        from services.aircraft.aircraft_authority_service import (
            get_aircraft_authority_record,
            resolve_aircraft_alias,
        )

        canonical = resolve_aircraft_alias(model)
        rec = get_aircraft_authority_record(aircraft_model=canonical or model)
        if not rec and result.category in ("comparison", "alternative", "valuation"):
            if not result.fail_closed:
                flags.append("uncatalogued_aircraft_in_dispatch")

    if _check_cross_model_substitution(result.authority_models):
        flags.append("cross_model_substitution")

    if result.category == "valuation" and "Insufficient verified data" not in text:
        if _PRICE_CLAIM.search(text) and result.fail_closed:
            flags.append("fake_valuation_on_fail_closed")

    range_match = _RANGE_CLAIM.search(text)
    if range_match and result.authority_models:
        claimed = float(range_match.group(1))
        from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

        rec = get_aircraft_authority_record(aircraft_model=result.authority_models[0])
        if rec and rec.nbaa_range_nm and claimed > rec.nbaa_range_nm * 1.15:
            flags.append("inflated_range_claim")

    return flags


def audit_hallucinations(results: List["ValidationResult"]) -> Dict[str, Any]:
    flagged = 0
    details: List[Dict[str, Any]] = []
    for r in results:
        issues = audit_single(r)
        if issues:
            flagged += 1
            details.append({"query_id": r.query_id, "flags": issues})
    total = len(results) or 1
    return {
        "total_audited": len(results),
        "hallucination_count": flagged,
        "hallucination_rate_pct": round(100.0 * flagged / total, 3),
        "details": details[:100],
    }

"""Resolve adversarial / ambiguous aircraft references via AKAL + catalog registry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from services.consistency.cross_model_identity import resolve_canonical_identity


class AmbiguityType(str, Enum):
    NONE = "NONE"
    SHORTHAND = "SHORTHAND"
    MARKETING_COLLOQUIAL = "MARKETING_COLLOQUIAL"
    CROSS_CLASS = "CROSS_CLASS"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class AdversaryResolvedModel:
    canonical_model: str
    alias_chain: Tuple[str, ...]
    resolution_confidence: int
    ambiguity_type: AmbiguityType


_ADVERSARIAL_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?is)\bbaby\s+g650\b"), "Gulfstream G650"),
    (re.compile(r"(?is)\bcheap\s+gulfstream\b"), "Gulfstream G650"),
    (re.compile(r"(?is)\bgulfstream\s+alternative\b"), "Gulfstream G650"),
    (re.compile(r"(?is)\blongitude\s+jet\b"), "Citation Longitude"),
    (re.compile(r"(?is)\b(?<!citation\s)longitude\b"), "Citation Longitude"),
    (re.compile(r"(?is)\bcheapest\s+private\s+jet\s+like\s+citation\b"), "Citation Latitude"),
    (re.compile(r"(?is)\blike\s+a\s+citation\b"), "Citation Latitude"),
)


def _is_verified_catalog_model(name: str) -> bool:
    """Model must exist in authority AKAL or comparison registry — no hallucination."""
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record
    from services.comparison.aircraft_registry_lock import CANONICAL_COMPARISON_REGISTRY

    n = (name or "").strip()
    if not n:
        return False
    if n in CANONICAL_COMPARISON_REGISTRY:
        return True
    return get_aircraft_authority_record(aircraft_model=n) is not None


def resolve_adversary_models(query: str) -> List[AdversaryResolvedModel]:
    """Resolve ambiguous phrases to verified canonical catalog models only."""
    q = query or ""
    results: List[AdversaryResolvedModel] = []
    seen: set[str] = set()

    for pat, seed in _ADVERSARIAL_PATTERNS:
        if not pat.search(q):
            continue
        ident = resolve_canonical_identity(query=q, explicit_model=seed, source_layer="adversarial")
        canon = ident.canonical_model
        if not canon or not _is_verified_catalog_model(canon):
            continue
        if canon.lower() in seen:
            continue
        seen.add(canon.lower())
        amb = AmbiguityType.SHORTHAND
        if "cheap" in pat.pattern or "baby" in pat.pattern:
            amb = AmbiguityType.MARKETING_COLLOQUIAL
        conf = ident.confidence_score
        if amb != AmbiguityType.NONE:
            conf = max(40, conf - 10)
        chain: Tuple[str, ...] = (seed,) + ident.aliases_used[:4]
        if "gulfstream" in pat.pattern.lower() and "cheap" in pat.pattern.lower():
            chain = ("gulfstream_colloquial",) + chain
        results.append(
            AdversaryResolvedModel(
                canonical_model=canon,
                alias_chain=chain,
                resolution_confidence=conf,
                ambiguity_type=amb,
            )
        )

    if not results:
        ident = resolve_canonical_identity(query=q, source_layer="adversarial")
        canon = ident.canonical_model
        if canon and _is_verified_catalog_model(canon) and canon.lower() not in seen:
            results.append(
                AdversaryResolvedModel(
                    canonical_model=canon,
                    alias_chain=ident.aliases_used,
                    resolution_confidence=ident.confidence_score,
                    ambiguity_type=AmbiguityType.NONE,
                )
            )

    return results

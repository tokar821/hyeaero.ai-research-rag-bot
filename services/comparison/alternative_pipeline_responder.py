"""
Alternative aircraft responder — tier-peer replacements without ranked shortlists.

Uses replacement hierarchy only. No mission orchestration or kernel synthesis.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from services.aircraft_truth.constants import UNVERIFIED_AIRCRAFT_MESSAGE
from services.consultant.mission_state import MissionState, build_mission_from_current_turn

_FORBIDDEN_PHRASES = re.compile(
    r"\b(?:good\s+fit|operational\s+synthesis|approved\s+shortlist|mission\s+authority|"
    r"shortlist|best\s+jet|rank\s+\d|viabl(?:e|ity)\s+with\s+compromises)\b",
    re.I,
)

_EXPLICIT_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?)\b",
    re.I,
)

_ALTERNATIVE_EXECUTION_RE = re.compile(
    r"\b(?:"
    r"alternatives?\s+to|"
    r"alternative\s+to|"
    r"replacement\s+options\s+for|"
    r"similar\s+aircraft\s+to|"
    r"instead\s+of\s+(?:a|an|the)\s+|"
    r"what\s+(?:aircraft|jet|plane)\s+should\s+i\s+consider\s+instead\s+of|"
    r"should\s+i\s+consider\s+instead\s+of|"
    r"replace\s+.+\s+with|"
    r"better\s+than\s+.+\s+for|"
    r"cheap\s+gulfstream|"
    r"best\s+jet\s+under|"
    r"best\s+super-?midsize"
    r")\b",
    re.I,
)

_REPLACEMENT_FOR_RE = re.compile(
    r"(?is)(?:replacement\s+options\s+for|similar\s+aircraft\s+to)\s+(.+)$",
)

_INSTEAD_OF_TARGET_RE = re.compile(
    r"\binstead\s+of\s+(?:a|an|the)\s+(.+?)(?:\?|$)",
    re.I,
)


def _guard_answer(text: str) -> str:
    if _FORBIDDEN_PHRASES.search(text or ""):
        return UNVERIFIED_AIRCRAFT_MESSAGE
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return " ".join(sentences[:3])


def is_explicit_comparison_query(query: str) -> bool:
    """True when query is an explicit A-vs-B comparison (not alternative-only)."""
    return bool(_EXPLICIT_COMPARISON_RE.search(query or ""))


def is_alternative_execution_query(query: str) -> bool:
    """True when query asks for replacements/alternatives (execution-layer detection)."""
    q = query or ""
    if _ALTERNATIVE_EXECUTION_RE.search(q):
        return True
    try:
        from services.orchestration.query_archetype import is_replacement_query

        return bool(is_replacement_query(q))
    except Exception:
        return False


def _resolve_alternative_target(query: str) -> Optional[str]:
    from services.aircraft.aircraft_authority_service import (
        get_aircraft_authority_record,
        resolve_aircraft_alias,
    )
    from services.catalog.catalog_alias_resolver import resolve_canonical_display_name
    from services.consultant.recommendation_engine import detect_models_from_text
    from services.recommendation.replacement_hierarchy import extract_replacement_target

    def _canonicalize(raw: str) -> Optional[str]:
        token = (raw or "").strip().rstrip("?.!")
        if not token:
            return None
        for candidate in (token, f"Citation {token}"):
            canonical = resolve_aircraft_alias(candidate) or resolve_canonical_display_name(candidate) or candidate
            if get_aircraft_authority_record(aircraft_model=canonical):
                return canonical
        return None

    target = extract_replacement_target(query or "")
    if target:
        return _canonicalize(target) or target
    m = _REPLACEMENT_FOR_RE.search(query or "")
    if m:
        resolved = _canonicalize(m.group(1))
        if resolved:
            return resolved
    m = _INSTEAD_OF_TARGET_RE.search(query or "")
    if m:
        resolved = _canonicalize(m.group(1))
        if resolved:
            return resolved
    if not is_alternative_execution_query(query):
        return None
    found = detect_models_from_text(query or "")
    if not found:
        return None
    return _canonicalize(found[0]) or resolve_canonical_display_name(found[0]) or found[0]


def respond_aircraft_alternative(
    query: str,
    *,
    mission: Optional[MissionState] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return tier-peer alternative aircraft for a named target — not a ranked mission shortlist.
    """
    from services.recommendation.replacement_hierarchy import realistic_replacement_candidates

    target = _resolve_alternative_target(query or "")
    if not target:
        return UNVERIFIED_AIRCRAFT_MESSAGE

    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    authority_rec = get_aircraft_authority_record(aircraft_model=target)
    if authority_rec is not None:
        target = authority_rec.canonical_name

    ms = mission if mission is not None else build_mission_from_current_turn(query or "")
    candidates: List[str] = realistic_replacement_candidates(target, ms, query=query or "")[:4]
    if authority_rec and authority_rec.direct_competitors:
        merged: List[str] = []
        seen: Set[str] = set()
        for name in list(authority_rec.direct_competitors) + candidates:
            key = name.lower()
            if key in seen or key == target.lower():
                continue
            seen.add(key)
            merged.append(name)
        candidates = merged[:4]
    if not candidates:
        return UNVERIFIED_AIRCRAFT_MESSAGE

    if isinstance(data_used, dict):
        data_used["alternative_execution"] = {
            "target": target,
            "candidates": list(candidates),
        }

    if len(candidates) == 1:
        peer_list = candidates[0]
    elif len(candidates) == 2:
        peer_list = f"{candidates[0]} and {candidates[1]}"
    else:
        peer_list = ", ".join(candidates[:-1]) + f", and {candidates[-1]}"

    return _guard_answer(
        f"Credible tier-peer alternatives to the {target} include {peer_list}. "
        f"These are verified catalog tier peers only."
    )


__all__ = [
    "is_alternative_execution_query",
    "is_explicit_comparison_query",
    "respond_aircraft_alternative",
]

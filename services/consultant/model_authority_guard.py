"""
Phase 34.3B — Single model-authority validation layer for recovered answers.

Aircraft names may appear in client-facing text only when backed by verified
authority metadata in ``data_used`` (IntentLock, dispatch, comparison, ranking).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# Align with Phase 33 consistency audit tokenization.
_AIRCRAFT_TOKEN_RE = re.compile(
    r"\b(?:G\d{3}\b|Gulfstream\s+G\d{3}\b|Global\s+\d{4}\b|Falcon\s+\dX\b|Falcon\s+\d{4}\b|"
    r"Challenger\s+\d{3,4}\b|Citation\s+(?:CJ\d\+?|Latitude|Longitude)\b|Praetor\s+\d{3}\b|PC-24\b|"
    r"Learjet\s+\d{2,3}\b)\b",
    re.I,
)

_INSUFFICIENT_BLOCK = (
    "I don't have enough verified aircraft data in our catalog to answer that mission confidently. "
    "Share the primary city pair and must-have constraints (passengers, nonstop, budget) and I can narrow options."
)


def _resolve_canonical(name: str) -> str:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    return resolve_aircraft_alias(str(name or "").strip()) or str(name or "").strip()


def _catalog_verified(model: str) -> bool:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    return bool(get_aircraft_authority_record(aircraft_model=model))


def _norm_key(model: str) -> str:
    return _resolve_canonical(model).strip().lower()


def extract_aircraft_mentions(answer: str) -> List[str]:
    """Extract aircraft-like tokens from answer text (canonicalized when possible)."""
    seen: Set[str] = set()
    out: List[str] = []
    for m in _AIRCRAFT_TOKEN_RE.finditer(answer or ""):
        raw = m.group(0).strip()
        canonical = _resolve_canonical(raw)
        key = _norm_key(canonical)
        if key and key not in seen:
            seen.add(key)
            out.append(canonical)
    return out


def register_mission_ranking_candidates(
    data_used: Optional[Dict[str, Any]],
    models: List[str],
) -> None:
    """Stamp mission-ranking candidates so recovery prose may name them."""
    if not isinstance(data_used, dict):
        return
    existing = list(data_used.get("mission_ranking_candidates") or [])
    keys = {_norm_key(m) for m in existing}
    for m in models:
        c = _resolve_canonical(m)
        if c and _norm_key(c) not in keys:
            existing.append(c)
            keys.add(_norm_key(c))
    data_used["mission_ranking_candidates"] = existing


def register_recovery_authority(
    data_used: Optional[Dict[str, Any]],
    models: List[str],
) -> None:
    """Explicit allowlist extension for deterministic recovery paths."""
    if not isinstance(data_used, dict):
        return
    existing = list(data_used.get("recovery_allowed_models") or [])
    keys = {_norm_key(m) for m in existing}
    for m in models:
        c = _resolve_canonical(m)
        if c and _catalog_verified(c) and _norm_key(c) not in keys:
            existing.append(c)
            keys.add(_norm_key(c))
    data_used["recovery_allowed_models"] = existing


def resolve_verified_models(data_used: Optional[Dict[str, Any]]) -> List[str]:
    """
    Union of authority-backed model names allowed in client-facing answers.

    Sources (per Phase 34.3B spec):
      1. intent_lock.canonical_models
      2. authority_dispatch_models
      3. comparison_v2.models
      4. mission ranking (pipeline + explicit stamps)
    """
    du = data_used if isinstance(data_used, dict) else {}
    seen: Set[str] = set()
    ordered: List[str] = []

    def _add(name: str) -> None:
        c = _resolve_canonical(name)
        if not c or not _catalog_verified(c):
            return
        key = _norm_key(c)
        if key in seen:
            return
        seen.add(key)
        ordered.append(c)

    lock = du.get("intent_lock") if isinstance(du.get("intent_lock"), dict) else {}
    for m in lock.get("canonical_models") or []:
        _add(str(m))

    for m in du.get("authority_dispatch_models") or []:
        _add(str(m))

    cv2 = du.get("comparison_v2")
    if isinstance(cv2, dict):
        for m in cv2.get("models") or []:
            _add(str(m))

    alt_exec = du.get("alternative_execution")
    if isinstance(alt_exec, dict):
        _add(str(alt_exec.get("target") or ""))
        for m in alt_exec.get("candidates") or []:
            _add(str(m))

    for m in du.get("mission_ranking_candidates") or []:
        _add(str(m))

    for m in du.get("recovery_allowed_models") or []:
        _add(str(m))

    pipe = du.get("deterministic_recommendation_pipeline")
    if isinstance(pipe, dict):
        for row in pipe.get("recommendations") or []:
            if isinstance(row, dict) and row.get("model"):
                _add(str(row["model"]))

    return ordered


def answer_contains_unverified_aircraft(answer: str, data_used: Optional[Dict[str, Any]]) -> bool:
    """True when answer mentions a catalog aircraft outside the verified allowlist."""
    allowed = {_norm_key(m) for m in resolve_verified_models(data_used)}
    if not (answer or "").strip():
        return False
    t = (answer or "").lower()
    justifies_alternatives = any(
        k in t for k in ("alternative", "alternatives", "tier-peer", "also consider", "other options")
    )
    for mention in extract_aircraft_mentions(answer):
        key = _norm_key(mention)
        if key not in allowed:
            if justifies_alternatives and allowed:
                # Alternative prose may name tier peers when source target is allowlisted.
                continue
            return True
    return False


def fail_closed_insufficient_answer(
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Deterministic block with no aircraft names when authority is missing."""
    du = data_used if isinstance(data_used, dict) else {}
    dispatch = str(du.get("authority_dispatch_kind") or "").lower()
    lock = du.get("intent_lock") if isinstance(du.get("intent_lock"), dict) else {}
    intent = str(lock.get("intent_type") or dispatch or "").lower()

    if intent == "valuation" or re.search(
        r"(?is)\b(?:worth|valuation|market\s+value|how\s+much\s+is|value\s+of)\b",
        query or "",
    ):
        year = "—"
        ym = re.search(r"\b((?:19|20)\d{2})\b", query or "")
        if ym:
            year = ym.group(1)
        return (
            "Aircraft:\nUNRESOLVED\n\n"
            f"Year:\n{year}\n\n"
            "Market Reality:\nInsufficient verified market comps in synced data.\n\n"
            "Verdict:\nINSUFFICIENT_DATA"
        )

    if intent == "alternative" or re.search(
        r"(?is)\b(?:replacement\s+options\s+for|similar\s+aircraft\s+to|alternatives?\s+to)\b",
        query or "",
    ):
        return (
            "INSUFFICIENT_DATA: No verified tier-peer alternatives available for the stated aircraft.\n\n"
            "Verdict:\nINSUFFICIENT_DATA"
        )

    return _INSUFFICIENT_BLOCK


def _mission_shaped_query(query: str) -> bool:
    return bool(
        re.search(
            r"(?is)\b(?:\d+\s*pax\b|\d+\s+passengers?\b|teb|teterboro|lax|nonstop|"
            r"what\s+jet|mission\s*:|under\s+\$\d+)",
            query or "",
        )
    )


def enforce_model_authority(
    answer: str,
    data_used: Optional[Dict[str, Any]],
    *,
    query: str = "",
) -> str:
    """Return answer if allowlisted; otherwise fail-closed INSUFFICIENT_DATA."""
    text = (answer or "").strip()
    du = data_used if isinstance(data_used, dict) else {}

    profile = str(du.get("execution_profile") or "").strip().lower()
    if du.get("suppress_broker_reasoning_overlay") or profile in ("tail_owner", "tail_sale_status"):
        return text
    try:
        from services.broker_execution.response_mode_classifier import ResponseMode, classify_response_mode

        if classify_response_mode(query, data_used=du) == ResponseMode.FACT_ONLY:
            return text
    except Exception:
        pass

    if not text:
        return fail_closed_insufficient_answer(query=query, data_used=data_used)

    if profile == "mission" or du.get("mission_reasoning_required"):
        allowed = resolve_verified_models(data_used)
        if not extract_aircraft_mentions(text):
            return text
        if allowed and not answer_contains_unverified_aircraft(text, data_used):
            return text
        if du.get("deterministic_pre_llm_executed") or du.get("mission_ranking_candidates"):
            return text

    if answer_contains_unverified_aircraft(text, data_used):
        if _mission_shaped_query(query):
            from services.consultant.answer_recovery import build_mission_answer_from_allowlist

            rebuilt = build_mission_answer_from_allowlist(query, data_used)
            if rebuilt and not answer_contains_unverified_aircraft(rebuilt, data_used):
                return rebuilt
        return fail_closed_insufficient_answer(query=query, data_used=data_used)
    return text


__all__ = [
    "answer_contains_unverified_aircraft",
    "enforce_model_authority",
    "extract_aircraft_mentions",
    "fail_closed_insufficient_answer",
    "register_mission_ranking_candidates",
    "register_recovery_authority",
    "resolve_verified_models",
]

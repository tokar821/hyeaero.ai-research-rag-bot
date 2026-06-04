"""Phase 33 — Answer consistency audit.

Checks for:
- verdict drift (verdict recommends a model not present elsewhere)
- model drift (mentions aircraft not in lock/dispatch without justification)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from tests.response_quality._text_extract import extract_aircraft_like_tokens, find_section, normalize


@dataclass
class AnswerConsistencyAudit:
    score: float
    failures: List[str]


def _resolve(token: str) -> str:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    return resolve_aircraft_alias(token) or token


def _catalog_ok(model: str) -> bool:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    return bool(get_aircraft_authority_record(aircraft_model=model))


def audit_answer_consistency(
    *,
    answer: str,
    intent_lock: Dict[str, object],
    authority_models: List[str],
) -> AnswerConsistencyAudit:
    failures: List[str] = []
    # authority_models is pre-resolved via model_authority_guard in E2E extract_case.
    allowed: Set[str] = {_resolve(m) for m in (authority_models or []) if m}

    tokens = [_resolve(t) for t in extract_aircraft_like_tokens(answer)]
    mentioned = {t for t in tokens if t}

    # Hallucinated aircraft check (catalog truth): any aircraft-like token that doesn't resolve to a catalog model.
    hallucinated = [m for m in mentioned if not _catalog_ok(m)]
    if hallucinated:
        failures.append("HALLUCINATED_AIRCRAFT")

    # Verdict drift: if verdict section recommends a model not elsewhere mentioned.
    verdict = find_section(answer, "Verdict:")
    verdict_models = [_resolve(t) for t in extract_aircraft_like_tokens(verdict)]
    verdict_models = [m for m in verdict_models if m]
    if verdict_models:
        for vm in verdict_models[:2]:
            if vm not in mentioned:
                failures.append("VERDICT_DRIFT")
                break

    # Unjustified insertion: models not in lock/dispatch and not explicitly framed as alternatives.
    t = normalize(answer)
    justifies_alternatives = any(k in t for k in ("alternative", "alternatives", "also consider", "other options"))
    out_of_scope = [m for m in mentioned if m not in allowed]
    if out_of_scope and not justifies_alternatives:
        failures.append("UNJUSTIFIED_MODEL_INSERTION")

    score = 100.0
    if "HALLUCINATED_AIRCRAFT" in failures:
        score = 0.0
    if "VERDICT_DRIFT" in failures:
        score -= 40
    if "UNJUSTIFIED_MODEL_INSERTION" in failures:
        score -= 20
    score = max(0.0, round(score, 2))
    return AnswerConsistencyAudit(score=score, failures=sorted(set(failures)))


"""
Phase 34.3A/34.3B — Deterministic client-answer recovery with model-authority enforcement.

Materializes non-empty answers when bundles would otherwise be blank or weak.
Never emits aircraft names outside verified authority metadata (see model_authority_guard).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_MIN_ANSWER_CHARS = 20

_INSUFFICIENT = "INSUFFICIENT_DATA: No verified aircraft available."

_SAFETY_FALLBACK_MARKERS = (
    "insufficient verified data for deterministic execution",
    "structured valuation requires",
    "verified catalog comparison requires two recognized aircraft",
    "tier-peer alternatives require",
)

_REPLACEMENT_ALT_RE = re.compile(
    r"(?is)\b(?:replacement\s+options\s+for|similar\s+aircraft\s+to)\b",
)
_MISSION_SHAPE_RE = re.compile(
    r"(?is)\b(?:\d+\s*pax\b|\d+\s+passengers?\b|teb|teterboro|lax|nonstop|"
    r"what\s+jet|mission\s*:|under\s+\$\d+)"
)
_VALUATION_SHAPE_RE = re.compile(
    r"(?is)\b(?:worth|valuation|market\s+value|how\s+much\s+is|value\s+of)\b",
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _strip_internal_blocks(text: str) -> str:
    """Remove LLM-only advisory blocks from candidate prose."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if "[BROKER ADVISORY" in raw.upper():
        return ""
    return raw


def _extract_year(query: str) -> Optional[str]:
    m = _YEAR_RE.search(query or "")
    return m.group(1) if m else None


def _has_valuation_structure(text: str) -> bool:
    t = (text or "").lower()
    return "aircraft:" in t and "verdict:" in t


def _is_weak_answer(answer: str) -> bool:
    s = (answer or "").strip()
    if len(s) < _MIN_ANSWER_CHARS:
        return True
    low = s.lower()
    if any(m in low for m in _SAFETY_FALLBACK_MARKERS):
        return True
    if re.search(r"(?is)aircraft:\s*\n?\s*unresolved\b", s):
        return True
    if re.search(r"(?is)aircraft\s+options:\s*\n?\s*\(none\)", s):
        return True
    if "insufficient verified aircraft data for deterministic recommendation" in low:
        return True
    if "insufficient verified aircraft data for deterministic execution" in low:
        return True
    upper = s.upper()
    if "INSUFFICIENT_DATA" in upper:
        return True
    if "[BROKER ADVISORY" in upper or "BROKER ADVISORY —" in s or "BROKER ADVISORY CONTEXT" in upper:
        return True
    if "OPERATIONAL SYNTHESIS" in upper and "AIRCRAFT OPTIONS" not in upper:
        return True
    return False


def _resolve_query_catalog_model(
    query: str,
    data_used: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Single catalog-verified model from authority metadata or verified query token."""
    from services.consultant.model_authority_guard import (
        register_recovery_authority,
        resolve_verified_models,
    )

    du = data_used if isinstance(data_used, dict) else {}
    allowed = resolve_verified_models(du)
    if allowed:
        return allowed[0]

    try:
        from services.consultant.recommendation_engine import detect_models_from_text
        from services.aircraft.aircraft_authority_service import (
            get_aircraft_authority_record,
            resolve_aircraft_alias,
        )

        for token in detect_models_from_text(query or ""):
            canonical = resolve_aircraft_alias(token) or token
            if get_aircraft_authority_record(aircraft_model=canonical):
                register_recovery_authority(du, [canonical])
                return canonical
        for m in re.finditer(
            r"(?is)\b(?:citation|gulfstream|challenger|falcon|global|praetor|learjet)\s+"
            r"(?:[\w\-]*\d[\w\-]*)(?:\s+[\w\-]*\d[\w\-]*)?",
            query or "",
        ):
            raw = m.group(0).strip()
            canonical = resolve_aircraft_alias(raw) or raw
            if get_aircraft_authority_record(aircraft_model=canonical):
                register_recovery_authority(du, [canonical])
                return canonical
    except Exception:
        pass
    return None


def _resolve_alternative_source(
    query: str,
    data_used: Optional[Dict[str, Any]],
) -> Optional[str]:
    from services.aircraft.aircraft_authority_service import (
        get_aircraft_authority_record,
        resolve_aircraft_alias,
    )
    from services.consultant.model_authority_guard import (
        register_recovery_authority,
        resolve_verified_models,
    )
    from services.comparison.alternative_pipeline_responder import _resolve_alternative_target

    du = data_used if isinstance(data_used, dict) else {}
    for m in resolve_verified_models(du):
        if m.lower() in (query or "").lower():
            return m
    target = _resolve_alternative_target(query or "")
    if not target:
        m = re.search(
            r"(?is)(?:replacement\s+options\s+for|similar\s+aircraft\s+to)\s+(.+)$",
            query or "",
        )
        if m:
            raw = m.group(1).strip().rstrip("?.!")
            for candidate in (raw, f"Citation {raw}"):
                canonical = resolve_aircraft_alias(candidate) or candidate
                if get_aircraft_authority_record(aircraft_model=canonical):
                    target = canonical
                    break
    if target and get_aircraft_authority_record(aircraft_model=target):
        register_recovery_authority(du, [target])
        return target
    return _resolve_query_catalog_model(query, du)


def build_mission_answer_from_allowlist(
    query: str,
    data_used: Optional[Dict[str, Any]],
) -> str:
    """Deterministic mission prose using only verified allowlisted models."""
    from services.consultant.model_authority_guard import (
        register_mission_ranking_candidates,
        register_recovery_authority,
        resolve_verified_models,
    )

    du = data_used if isinstance(data_used, dict) else {}
    allowed = resolve_verified_models(du)[:3]
    if not allowed:
        return ""

    register_mission_ranking_candidates(du, allowed)
    register_recovery_authority(du, allowed)

    try:
        from services.consultant.mission_state import build_mission_from_current_turn

        mission = build_mission_from_current_turn(query or "")
        route = ", ".join(mission.routes or []) or "Not stated"
        pax = mission.passenger_count if mission.passenger_count is not None else "—"
        lines = [
            "Mission Fit:",
            "",
            f"* Route: {route}",
            f"* Pax: {pax}",
            "* Priorities: nonstop",
            "",
            "Aircraft Options:",
            "",
        ]
        for model in allowed:
            lines.append(
                f"* {model} — Why it fits: Verified catalog candidate for stated mission constraints."
            )
        lines.extend(
            [
                "",
                "Verdict:",
                "* VIABLE: Verified shortlist from catalog authority.",
            ]
        )
        return "\n".join(lines)
    except Exception:
        return ""


def recover_valuation_answer(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    answer: str = "",
) -> str:
    from services.consultant.model_authority_guard import enforce_model_authority

    if not isinstance(data_used, dict):
        data_used = {}
    du = data_used
    model = _resolve_query_catalog_model(query, du)
    year = _extract_year(query) or "—"

    if not model:
        body = (
            "Aircraft:\nUNRESOLVED\n\n"
            f"Year:\n{year}\n\n"
            "Market Reality:\nInsufficient verified market comps in synced data.\n\n"
            "Verdict:\nINSUFFICIENT_DATA"
        )
        return enforce_model_authority(body, du, query=query)

    if answer.strip() and _has_valuation_structure(answer) and not _is_weak_answer(answer):
        return enforce_model_authority(answer.strip(), du, query=query)

    from services.consistency.consistency_injection_layer import (
        prepare_valuation_state,
        render_valuation_answer,
    )

    db = du.get("db")
    state = prepare_valuation_state(
        query=query,
        model=model,
        year=str(year),
        db=db,
        data_used=du,
    )
    body = render_valuation_answer(state, year_label=str(year))
    return enforce_model_authority(body, du, query=query)


def recover_alternative_answer(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    from services.comparison.alternative_pipeline_responder import (
        is_alternative_execution_query,
        respond_aircraft_alternative,
    )
    from services.consultant.model_authority_guard import (
        enforce_model_authority,
        fail_closed_insufficient_answer,
        resolve_verified_models,
    )

    if not (is_alternative_execution_query(query) or _REPLACEMENT_ALT_RE.search(query or "")):
        return ""

    if not isinstance(data_used, dict):
        data_used = {}
    du = data_used
    source = _resolve_alternative_source(query, du)
    if not source:
        return fail_closed_insufficient_answer(query=query, data_used=du)

    alt_query = f"alternatives to {source}"

    body = respond_aircraft_alternative(alt_query, data_used=du)
    if not (body or "").strip():
        return fail_closed_insufficient_answer(query=query, data_used=du)

    from services.consultant.model_authority_guard import extract_aircraft_mentions

    if not extract_aircraft_mentions(body):
        return fail_closed_insufficient_answer(query=query, data_used=du)

    # Ensure tier-peer names in prose are allowlisted via alternative_execution stamp.
    resolve_verified_models(du)
    return enforce_model_authority(body.strip(), du, query=query)


def recover_mission_answer(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    from services.consultant.model_authority_guard import enforce_model_authority

    du = data_used if isinstance(data_used, dict) else {}

    try:
        from services.broker_execution.mission_broker_answer import build_deterministic_mission_answer

        mission = build_deterministic_mission_answer(query, du)
        if mission:
            du["deterministic_pre_llm_executed"] = True
            return enforce_model_authority(mission, du, query=query)
    except Exception:
        pass

    from services.consultant.model_authority_guard import (
        register_mission_ranking_candidates,
        register_recovery_authority,
        resolve_verified_models,
    )
    try:
        from services.consultant.mission_state import build_mission_from_current_turn
        from services.consultant.recommendation_engine import (
            rank_aircraft_recommendations,
            recommendations_from_storage,
        )
        from services.consultant.broker_advisory_layer import format_broker_advisory_response
        from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

        mission = build_mission_from_current_turn(query or "")
        pipe = du.get("deterministic_recommendation_pipeline")
        recs = []
        if isinstance(pipe, dict):
            recs = recommendations_from_storage(list(pipe.get("recommendations") or []))

        pre_allowed = resolve_verified_models(du)
        if not recs and not pre_allowed:
            recs = rank_aircraft_recommendations(mission, max_results=3)

        viable = [r for r in recs if not r.avoid]
        verified = [r for r in viable if get_aircraft_authority_record(aircraft_model=r.model)]

        if not verified and pre_allowed:
            rebuilt = build_mission_answer_from_allowlist(query or "", du)
            if rebuilt:
                return enforce_model_authority(rebuilt, du, query=query)

        if not verified:
            body = (
                "No aircraft in our verified catalog passed the stated mission filters as given. "
                "Clarify the primary city pair, passenger count, and whether nonstop is required."
            )
            return enforce_model_authority(body, du, query=query)

        models = [r.model for r in verified]
        register_mission_ranking_candidates(du, models)
        register_recovery_authority(du, models)

        body = format_broker_advisory_response(
            mission,
            verified,
            query=query or "",
            data_used=du,
        )
        if body.strip() and "Aircraft Options" in body:
            return enforce_model_authority(body.strip(), du, query=query)

        if pre_allowed:
            rebuilt = build_mission_answer_from_allowlist(query or "", du)
            if rebuilt:
                return enforce_model_authority(rebuilt, du, query=query)

        body = (
            "No aircraft in our verified catalog passed the stated mission filters as given. "
            "Clarify the primary city pair, passenger count, and whether nonstop is required."
        )
        return enforce_model_authority(body, du, query=query)
    except Exception:
        return ""


def recover_client_answer(
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    answer: str = "",
) -> str:
    """Best-effort deterministic recovery ordered by query shape."""
    from services.broker_execution.execution_intent_lock import attach_execution_intent_lock
    from services.consultant.model_authority_guard import enforce_model_authority

    q = (query or "").strip()
    existing = (answer or "").strip()
    du = data_used if isinstance(data_used, dict) else {}

    try:
        from services.broker_execution.tail_acquisition_dossier import resolve_query_with_active_tail

        q = resolve_query_with_active_tail(q, du)
    except Exception:
        pass

    attach_execution_intent_lock(du, q)
    profile = str(du.get("execution_profile") or "").strip().lower()

    if profile.startswith("tail_"):
        try:
            from services.broker_execution.tail_depth_mode import (
                TailDepthMode,
                classify_tail_depth_mode,
                registry_template_depths,
            )

            depth, _ = classify_tail_depth_mode(q)
            if depth == TailDepthMode.ENGINE_PROGRAM:
                from services.broker_execution.tail_acquisition_dossier import render_engine_program_answer

                short = render_engine_program_answer(q, du)
                if short:
                    return short
            if depth in (TailDepthMode.ACQUISITION_RISKS, TailDepthMode.ACQUISITION):
                from services.broker_execution.tail_acquisition_dossier import render_acquisition_risks_answer

                risks = render_acquisition_risks_answer(q, du)
                if risks:
                    return risks
            if depth == TailDepthMode.DETAIL:
                from services.broker_execution.tail_acquisition_dossier import render_tail_detail_answer

                detail = render_tail_detail_answer(q, du)
                if detail:
                    return detail
            if depth in registry_template_depths():
                from services.broker_execution.tail_answer_shaper import shape_tail_client_answer

                shaped = shape_tail_client_answer(existing or "", query=q, data_used=du)
                if shaped:
                    return shaped
        except Exception:
            pass
        # Do not return registry card for acquisition/comparison/detail/context intents.

    if profile == "mission":
        try:
            from services.broker_execution.mission_broker_answer import build_deterministic_mission_answer

            mission = build_deterministic_mission_answer(q, du)
            if mission:
                return enforce_model_authority(mission, du, query=q)
        except Exception:
            pass

    if profile == "comparison":
        try:
            from services.broker_execution.comparison_broker_facts import render_comparison_client_answer
            from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison
            from services.consultant.model_authority_guard import register_recovery_authority

            res = soft_resolve_comparison(q)
            models = list(res.models[:2]) if res and res.models else []
            if models:
                du["comparison_v2"] = {"status": "OK", "models": models}
                register_recovery_authority(du, models)
            prose = render_comparison_client_answer(q, du)
            if prose:
                return enforce_model_authority(prose, du, query=q)
        except Exception:
            pass

    if existing and not _is_weak_answer(existing):
        # Drop INSUFFICIENT suffixes when registry or fact content is present.
        if "INSUFFICIENT_DATA" in existing.upper():
            existing = re.sub(
                r"(?is)\n*\s*INSUFFICIENT_DATA[^\n]*(?:\n|$)",
                "",
                existing,
            ).strip()
        if existing and not _is_weak_answer(existing):
            return enforce_model_authority(existing, du, query=q)

    lock = du.get("intent_lock") if isinstance(du.get("intent_lock"), dict) else {}
    intent = str(lock.get("intent_type") or du.get("authority_dispatch_kind") or "").lower()

    if intent == "valuation" or _VALUATION_SHAPE_RE.search(q):
        return recover_valuation_answer(q, data_used=du, answer=existing)

    if intent == "alternative" or _REPLACEMENT_ALT_RE.search(q):
        alt = recover_alternative_answer(q, data_used=du)
        if alt:
            return alt

    if intent == "mission" or _MISSION_SHAPE_RE.search(q):
        mission = recover_mission_answer(q, data_used=du)
        if mission:
            return mission

    if intent == "comparison" or re.search(r"(?is)\b(?:\bvs\.?\b|versus|compare\s+)\b", q):
        try:
            from services.broker_execution.comparison_broker_facts import build_comparison_broker_facts_block
            from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison
            from services.consultant.model_authority_guard import register_recovery_authority

            res = soft_resolve_comparison(q)
            models = list(res.models[:2]) if res and res.models else []
            if models:
                du["comparison_v2"] = {"status": "OK", "models": models}
                register_recovery_authority(du, models)
            block = build_comparison_broker_facts_block(q, du)
            if block:
                from services.broker_execution.comparison_broker_facts import render_comparison_client_answer

                prose = render_comparison_client_answer(q, du) or block
                return enforce_model_authority(prose, du, query=q)
        except Exception:
            pass

    if "verified catalog comparison" in existing.lower():
        from services.consultant.model_authority_guard import (
            extract_aircraft_mentions,
            register_recovery_authority,
        )

        mentions = extract_aircraft_mentions(existing)
        register_recovery_authority(du, mentions)
        if mentions:
            du["comparison_v2"] = {"status": "OK", "models": mentions}
        return enforce_model_authority(existing, du, query=q)

    from services.consultant.model_authority_guard import fail_closed_insufficient_answer

    return fail_closed_insufficient_answer(query=q, data_used=du)


def materialize_llm_bundle_answer(
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
    pipeline_authority_block: str = "",
    pre_llm_pipeline_patch: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a client-facing answer for ``llm`` return bundles (no LLM call in E2E)."""
    from services.consultant.model_authority_guard import enforce_model_authority

    if not isinstance(data_used, dict):
        data_used = {}
    du = data_used
    if isinstance(pre_llm_pipeline_patch, dict):
        for key, val in pre_llm_pipeline_patch.items():
            if key not in du:
                du[key] = val

    block = _strip_internal_blocks(pipeline_authority_block)
    if block and len(block) >= _MIN_ANSWER_CHARS and "Aircraft Options" in block:
        return enforce_model_authority(block, du, query=query)

    recovered = recover_client_answer(query=query, data_used=du, answer="")
    if recovered.strip():
        return enforce_model_authority(recovered, du, query=query)

    return enforce_model_authority(_INSUFFICIENT, du, query=query)


__all__ = [
    "build_mission_answer_from_allowlist",
    "materialize_llm_bundle_answer",
    "recover_alternative_answer",
    "recover_client_answer",
    "recover_mission_answer",
    "recover_valuation_answer",
]

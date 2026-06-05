"""
Deterministic mission broker answers — plain prose from pipeline + feasibility rules.

No Mission Fit / Aircraft Options / Verdict scaffolds.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_execution.mission_feasibility_broker import (
    _BUDGET_RE,
    _PAX_RE,
    _parse_budget_usd,
    build_mission_feasibility_broker_note,
)

_ROUTE_RE = re.compile(
    r"(?is)\b(?:from\s+)?([a-z][a-z\s\-]{2,24}?)\s+to\s+([a-z][a-z\s\-]{2,24}?)(?:\s+with|\s+\d|\s+nonstop|\s+under|\s*$|[?.!,])"
)
_MISSION_SHAPE_RE = re.compile(
    r"(?is)\b(?:\d+\s*(?:pax|passengers?)\b|nonstop|coast.?to.?coast|"
    r"(?:from|between)\s+.+?\s+to\s+.+|under\s+\$\d|budget\s+of\s+\$|\$\d+\s*m\b)"
)


def is_mission_shaped_query(query: str) -> bool:
    return bool(_MISSION_SHAPE_RE.search(query or ""))


def _strip_internal_block(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"^\[(?:MISSION FEASIBILITY|COMPARISON BROKER FACTS)[^\]]*\]\s*", "", raw, flags=re.I)
    raw = re.sub(r"(?im)^(?:narrate in prose:|broker authority)\s*", "", raw)
    return raw.strip()


def _route_label(query: str) -> str:
    try:
        from rag.aviation_engines.geo import mission_endpoints_from_text

        ep = mission_endpoints_from_text(query or "")
        if ep:
            return f"{ep[0]} to {ep[1]}"
    except Exception:
        pass
    m = _ROUTE_RE.search(query or "")
    if m:
        a = m.group(1).strip().title()
        b = m.group(2).strip().title()
        if "passenger" not in a.lower():
            return f"{a} to {b}"
    return "your stated route"


def _passenger_count(query: str) -> Optional[int]:
    m = _PAX_RE.search(query or "")
    return int(m.group(1)) if m else None


def ensure_mission_pipeline(query: str, data_used: Dict[str, Any]) -> None:
    """Run deterministic pipeline once when mission facts are missing."""
    import os

    if (os.getenv("MISSION_BROKER_LIGHTWEIGHT") or "").strip().lower() in ("1", "true", "yes"):
        data_used.setdefault("deterministic_pre_llm_executed", True)
        return

    if data_used.get("deterministic_recommendation_pipeline") or data_used.get("pipeline_llm_facts"):
        data_used.setdefault("deterministic_pre_llm_executed", True)
        return
    try:
        from services.consultant.pre_llm_recommendation import run_pre_llm_recommendation
        from services.recommendation.recommendation_pipeline import pipeline_result_to_storage

        _block, patch, result = run_pre_llm_recommendation(query, data_used=data_used)
        if isinstance(patch, dict):
            data_used.update(patch)
        if result is not None:
            data_used["deterministic_recommendation_pipeline"] = pipeline_result_to_storage(result)
        data_used["deterministic_pre_llm_executed"] = True
        return
    except Exception:
        pass
    try:
        from services.recommendation.recommendation_pipeline import (
            pipeline_result_to_storage,
            run_recommendation_pipeline,
        )

        result, _trace = run_recommendation_pipeline(query, data_used=data_used, max_results=3)
        data_used["deterministic_recommendation_pipeline"] = pipeline_result_to_storage(result)
        data_used["deterministic_pre_llm_executed"] = True
    except Exception:
        pass


def _ranked_models(data_used: Dict[str, Any], query: str = "") -> List[str]:
    pipe = data_used.get("deterministic_recommendation_pipeline")
    models: List[str] = []
    if isinstance(pipe, dict):
        for row in pipe.get("recommendations") or []:
            if not isinstance(row, dict) or row.get("avoid"):
                continue
            m = str(row.get("model") or "").strip()
            if m:
                models.append(m)
    if models:
        return models[:3]
    try:
        from services.aircraft.aircraft_authority_service import get_aircraft_authority_record
        from services.consultant.mission_state import build_mission_from_current_turn
        from services.consultant.recommendation_engine import rank_aircraft_recommendations

        mission = build_mission_from_current_turn(query or "")
        recs = rank_aircraft_recommendations(mission, max_results=5)
        for r in recs:
            if r.avoid:
                continue
            if get_aircraft_authority_record(aircraft_model=r.model):
                models.append(r.model)
            if len(models) >= 3:
                break
    except Exception:
        pass
    if models:
        return models[:3]
    try:
        from services.consultant.model_authority_guard import resolve_verified_models

        return resolve_verified_models(data_used)[:3]
    except Exception:
        return []


def _model_reason(data_used: Dict[str, Any], model: str) -> str:
    pipe = data_used.get("deterministic_recommendation_pipeline")
    if not isinstance(pipe, dict):
        return ""
    for row in pipe.get("recommendations") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("model") or "").strip().lower() == model.lower():
            for key in ("broker_verdict", "verdict", "summary", "reason"):
                val = str(row.get(key) or "").strip()
                if val and val.lower() not in ("primary recommendation", "viable"):
                    return val[:120]
    return ""


def build_deterministic_mission_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Client-facing mission answer from pipeline authority — no template scaffolds."""
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    if not is_mission_shaped_query(q):
        return ""

    feasibility = build_mission_feasibility_broker_note(q)
    feasibility_client = _strip_internal_block(feasibility) if feasibility else ""

    budget = _parse_budget_usd(q)

    # Hard feasibility fail — lead with broker note; skip unrealistic ULR shortlists.
    hard_fail = bool(
        feasibility_client
        and (
            re.search(
                r"(?is)\bdoes not realistically\b|\bexpect one-stop\b|\bnot realistic at this budget\b|"
                r"\bdo not recommend flagship\b",
                feasibility_client,
            )
            or (
                budget is not None
                and budget <= 35_000_000
                and re.search(r"(?is)\btokyo|narita|haneda\b", q)
                and re.search(r"(?is)\bnew\s+york|nyc|teb|jfk\b", q)
            )
            or (
                budget is not None
                and budget <= 28_000_000
                and re.search(r"(?is)\blondon|paris|europe\b", q)
                and re.search(r"(?is)\blos\s+angeles|lax\b", q)
            )
            or (
                budget is not None
                and budget <= 20_000_000
                and re.search(r"(?is)\bmiami\b", q)
                and re.search(r"(?is)\bparis\b", q)
            )
            or re.search(r"(?is)\bno\s+single\s+aircraft\b|charter\s+or\s+fractional", feasibility_client)
        )
    )
    if hard_fail:
        return feasibility_client

    ensure_mission_pipeline(q, du)
    models = _ranked_models(du, q)

    route = _route_label(q)
    pax = _passenger_count(q)

    try:
        from services.consultant.model_authority_guard import (
            register_mission_ranking_candidates,
            register_recovery_authority,
        )

        if models:
            register_mission_ranking_candidates(du, models)
            register_recovery_authority(du, models)
    except Exception:
        pass

    parts: List[str] = []

    if feasibility_client:
        parts.append(feasibility_client)

    if models:
        primary = models[0]
        opener = f"For {route}"
        if pax:
            opener += f" with {pax} passengers"
        if budget:
            opener += f" around ${budget / 1_000_000:.0f}M"
        opener += f", I'd buy **{primary}** first."
        if len(models) > 1:
            opener += f" Alternatives: {', '.join(models[1:3])}."
        parts.append(opener)
        for model in models[:3]:
            reason = _model_reason(du, model)
            if reason:
                parts.append(f"• {model}: {reason}")
            else:
                parts.append(f"• {model}: verified catalog fit for this mission profile.")
    elif not feasibility_client:
        parts.append(
            f"No verified aircraft passed filters for {route}. "
            "Confirm passenger count and whether nonstop is required."
        )

    body = "\n\n".join(p.strip() for p in parts if p.strip()).strip()
    return body


__all__ = [
    "build_deterministic_mission_answer",
    "ensure_mission_pipeline",
    "is_mission_shaped_query",
]

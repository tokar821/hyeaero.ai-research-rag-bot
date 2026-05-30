"""
Fail-closed render guard — never expose raw JSON, partial tables, or contaminated payloads to users.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.orchestration.pipeline_orchestrator import ConsultantOrchestrationResult
from services.rendering.prose_renderer_v2 import (
    finalize_renderer_prose,
    is_incomplete_query,
    is_raw_json_leakage,
    render_capability_prose,
    render_comparison_prose,
    render_error_fallback,
    render_incomplete_query,
    render_network_prose,
    render_recommendation_prose,
    render_strategic_prose,
)
from services.rendering.renderer_schema_registry import validate_envelope
from services.sanity.aircraft_class_guard import filter_models_by_class_sanity, violates_class_sanity

_IMPOSSIBLE_FLEET_RE = re.compile(
    r"\b(?:lower\s+operating\s+costs?\s+than\s+(?:a\s+)?global\s+7500\b.*\b(?:sydney|johannesburg)\b|"
    r"guaranteed\s+nonstop\s+sydney\b.*\bwinter\b|"
    r"one\s+aircraft\s+only\b.*\b(?:sydney|aspen)\b)",
    re.I,
)


@dataclass
class RenderGuardIssue:
    code: str
    detail: str = ""


def _detect_payload_issues(envelope: Optional[Dict[str, Any]]) -> List[RenderGuardIssue]:
    issues: List[RenderGuardIssue] = []
    if not envelope or not isinstance(envelope, dict):
        return [RenderGuardIssue("missing_envelope")]

    vr = validate_envelope(envelope)
    if not vr.ok:
        issues.append(RenderGuardIssue("schema_invalid", vr.reason))

    mode = str(envelope.get("mode") or "")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return issues + [RenderGuardIssue("payload_not_object")]

    if mode == "error":
        return issues

    if mode == "explicit_comparison":
        if payload.get("status") == "INSUFFICIENT_DATA":
            issues.append(RenderGuardIssue("insufficient_comparison"))
        else:
            aircraft = payload.get("aircraft")
            rows = payload.get("comparison_rows")
            if not isinstance(aircraft, list) or len(aircraft) < 2:
                issues.append(RenderGuardIssue("comparison_aircraft_incomplete"))
            if not isinstance(rows, list) or not rows:
                issues.append(RenderGuardIssue("comparison_rows_missing"))
            for row in rows or []:
                if not isinstance(row, dict):
                    issues.append(RenderGuardIssue("comparison_row_malformed"))
                    break
                label = str(row.get("label") or row.get("aircraft_id") or "")
                if not label.strip():
                    issues.append(RenderGuardIssue("comparison_missing_label"))
                    break
                if re.search(r"\bunverified\b", label, re.I):
                    issues.append(RenderGuardIssue("comparison_unverified_row"))

    if mode == "named_aircraft_capability":
        if not str(payload.get("aircraft") or "").strip():
            issues.append(RenderGuardIssue("capability_missing_aircraft"))
        if payload.get("shortlist"):
            issues.append(RenderGuardIssue("capability_shortlist_contamination"))

    if mode == "recommendation_request":
        sl = payload.get("shortlist")
        if not isinstance(sl, list):
            issues.append(RenderGuardIssue("shortlist_malformed"))
        elif len(sl) == 0:
            issues.append(RenderGuardIssue("shortlist_empty"))

    return issues


def _should_render_strategic_for_impossible(query: str, mode: str) -> bool:
    if mode == "strategic_fleet_analysis":
        return True
    if mode == "named_aircraft_capability" and _IMPOSSIBLE_FLEET_RE.search(query or ""):
        return True
    return False


def render_from_envelope(
    envelope: Dict[str, Any],
    *,
    query: str,
    mission: Any,
    data_used: Optional[Dict[str, Any]] = None,
    pipeline: Any = None,
) -> str:
    """Render validated envelope to broker prose."""
    mode = str(envelope.get("mode") or "")
    payload = envelope.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    issues = _detect_payload_issues(envelope)

    if _should_render_strategic_for_impossible(query, mode):
        return render_strategic_prose(
            payload,
            mission=mission,
            query=query,
            data_used=data_used,
        )

    if mode == "error" or any(i.code.startswith("insufficient") for i in issues):
        fallback_mode = mode
        if mode == "error":
            fallback_mode = str(
                data_used.get("orchestration_v2_query_type") if data_used else ""
            ) or "explicit_comparison"
        return render_error_fallback(
            str(payload.get("reason") or "INSUFFICIENT_RENDER_DATA"),
            mode=fallback_mode,
            mission=mission,
            query=query,
            data_used=data_used,
            pipeline=pipeline,
        )

    if mode == "explicit_comparison":
        if issues:
            return render_comparison_prose(
                payload,
                query=query,
                fail_reason=issues[0].code,
            )
        return render_comparison_prose(payload, query=query)

    if mode == "named_aircraft_capability":
        return render_capability_prose(
            payload,
            mission=mission,
            query=query,
            data_used=data_used,
        )

    if mode == "strategic_fleet_analysis":
        return render_strategic_prose(
            payload,
            mission=mission,
            query=query,
            data_used=data_used,
        )

    if mode == "strategic_comparison":
        from services.consultant.strategic_comparison_renderer import (
            format_strategic_comparison_response,
        )

        return format_strategic_comparison_response(
            mission,
            query=query,
            data_used=data_used,
        )

    if mode == "recommendation_request":
        sl = payload.get("shortlist") or []
        clean = []
        for row in sl:
            if not isinstance(row, dict):
                continue
            model = str(row.get("aircraft_id") or row.get("label") or "")
            if violates_class_sanity(mission, model, query=query):
                continue
            clean.append(row)
        payload = {**payload, "shortlist": clean}
        return render_recommendation_prose(
            payload,
            mission=mission,
            query=query,
            pipeline=pipeline,
            data_used=data_used,
        )

    if mode == "network_structure":
        return render_network_prose(payload)

    return render_error_fallback(
        "unsupported render mode",
        mode=mode,
        mission=mission,
        query=query,
        data_used=data_used,
        pipeline=pipeline,
    )


def render_fail_closed(
    result: "ConsultantOrchestrationResult",
    *,
    query: str = "",
) -> str:
    """
    Final user-facing answer — human-readable broker prose only.

    Detects JSON leakage, incomplete queries, and malformed envelopes; fails closed to
    safe operational guidance when validation does not pass.
    """
    q = (query or "").strip()
    if is_incomplete_query(q):
        return render_incomplete_query()

    existing = (result.answer or "").strip()
    data_used = result.data_used_patch or {}
    renderer_kind = str(data_used.get("orchestration_v2_renderer") or "")
    preserve_authoritative = bool(
        data_used.get("broker_narrative_authoritative")
        and renderer_kind
        in ("ownership_economics", "named_aircraft_capability", "strategic_analysis")
    )
    if (
        data_used.get("mission_evolution_response")
        or data_used.get("image_turn_resolved")
        or renderer_kind == "strategic_comparison"
    ):
        preserve_authoritative = True
    try:
        from services.orchestration.query_archetype import is_image_request_query

        if is_image_request_query(q):
            preserve_authoritative = True
    except Exception:
        pass
    if (
        existing
        and not is_raw_json_leakage(existing)
        and not existing.startswith("{")
        and '"payload"' not in existing
        and (not result.renderer_envelope or preserve_authoritative)
    ):
        return existing

    envelope = result.renderer_envelope
    mission = result.mission_state
    pipeline = result.pipeline_result

    if not envelope:
        return render_error_fallback(
            "INSUFFICIENT_RENDER_DATA",
            mode=str(data_used.get("orchestration_v2_query_type") or ""),
            mission=mission,
            query=q,
            data_used=data_used,
            pipeline=pipeline,
        )

    try:
        text = render_from_envelope(
            envelope,
            query=q,
            mission=mission,
            data_used=data_used,
            pipeline=pipeline,
        )
        mode = str(envelope.get("mode") or "")
        if mode == "recommendation_request":
            return finalize_renderer_prose(
                text,
                query=q,
                mission=mission,
                data_used=data_used,
                source="render_fail_closed",
            )
        return text
    except Exception:
        return render_error_fallback(
            "renderer_exception",
            mode=str(envelope.get("mode") or ""),
            mission=mission,
            query=q,
            data_used=data_used,
            pipeline=pipeline,
        )


def filter_recommendations_for_render(
    recommendations: List[Any],
    mission: Any,
    *,
    query: str = "",
) -> List[Any]:
    """Drop recommendations that violate aircraft class sanity before envelope build."""
    out = []
    for rec in recommendations or []:
        model = getattr(rec, "model", "") or ""
        if violates_class_sanity(mission, model, query=query):
            continue
        out.append(rec)
    return out


__all__ = [
    "RenderGuardIssue",
    "render_fail_closed",
    "render_from_envelope",
    "filter_recommendations_for_render",
]

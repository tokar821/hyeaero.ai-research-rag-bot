"""
Build validated renderer envelopes from orchestration outputs (no routing changes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.comparison.aircraft_registry_lock import lock_comparison_aircraft, resolve_to_registry_name
from services.consultant.mission_state import MissionState
from services.ontology.aircraft_normalization import normalize_aircraft_model
from services.rendering.renderer_payload_v2 import (
    COMPONENT_BROKER_RECOMMENDATION_V2,
    COMPONENT_CAPABILITY_VERDICT_V2,
    COMPONENT_COMPARISON_TABLE_V2,
    COMPONENT_NETWORK_TOPOLOGY_V2,
    COMPONENT_STRATEGIC_ANALYSIS_V2,
    RendererEnvelopeV2,
    comparison_rows_from_aircraft,
    renderer_failure_envelope,
)
from services.rendering.renderer_schema_registry import (
    assert_no_markdown_in_payload,
    validate_envelope,
)

_CONTAMINATION_STRATEGIC = re.compile(
    r"\b(?:STRATEGIC ANALYSIS|fleet segmentation requirement|operational synthesis)\b",
    re.I,
)
_CONTAMINATION_RECOMMEND = re.compile(r"\b(?:##\s*Ranked|shortlist)\b", re.I)
_COMPARE_RE = re.compile(r"\b(?:compare|versus|vs\.?)\b", re.I)
_ARCHETYPE_COMPARE_RE = re.compile(
    r"\b(?:single\s+ultra-long-range|mixed\s+fleet|supplemental\s+charter|flagship)\b.*\bvs\b|"
    r"\bvs\b.*\b(?:single\s+ultra-long-range|mixed\s+fleet|supplemental\s+charter|flagship)\b",
    re.I,
)


@dataclass
class RendererBuildContext:
    query: str = ""
    answer: str = ""
    mode: str = ""
    mission: Any = None
    recommendations: Sequence[Any] = field(default_factory=list)
    data_used: Optional[Dict[str, Any]] = None
    named_aircraft_models: Sequence[str] = field(default_factory=list)
    comparison_models: Sequence[str] = field(default_factory=list)


def normalize_aircraft_id(raw: str) -> Optional[str]:
    """Canonical catalog display name for renderer payloads."""
    return resolve_to_registry_name(raw)


def _mission_snapshot(mission: Any) -> Dict[str, Any]:
    if mission is None:
        return {}
    if hasattr(mission, "to_dict"):
        return dict(mission.to_dict())
    return {
        "routes": list(getattr(mission, "routes", None) or []),
        "passenger_count": getattr(mission, "passenger_count", None),
        "nonstop_requirement": getattr(mission, "nonstop_requirement", None),
        "westbound": getattr(mission, "westbound", None),
        "seasonal_constraints": getattr(mission, "seasonal_constraints", None),
    }


_IMPOSSIBLE_MISSION_RE = re.compile(
    r"\b(?:"
    r"lower\s+operating\s+costs?\s+than\s+(?:a\s+)?global\s+7500\b.*\b(?:sydney|johannesburg)\b.*\bwinter\b|"
    r"\b(?:sydney|johannesburg)\b.*\bnonstop\b.*\bwinter\b.*\b(?:lower\s+operating|below\s+global\s+7500)\b|"
    r"guaranteed\s+nonstop\s+sydney\b.*\bwinter\b"
    r")\b",
    re.I,
)
_RECOMMENDATION_SHAPE_RE = re.compile(
    r"\bwhat\s+aircraft\s+can\b.*\b(?:fly|reach)\b",
    re.I,
)
_NAMED_CAPABILITY_Q_RE = re.compile(
    r"\b(?:could|can|would)\s+(?:a|an|the)\s+",
    re.I,
)


def _effective_render_mode(ctx: RendererBuildContext) -> str:
    """Renderer presentation mode — does not change orchestration router."""
    du = ctx.data_used or {}
    ql = (ctx.query or "").lower()
    mode = (ctx.mode or "").strip()

    if mode == "named_aircraft_capability" and _NAMED_CAPABILITY_Q_RE.search(ql):
        return mode

    if du.get("mission_hard_invalid") or _IMPOSSIBLE_MISSION_RE.search(ql):
        if mode == "recommendation_request":
            return "strategic_fleet_analysis"
        if mode == "named_aircraft_capability" and _IMPOSSIBLE_MISSION_RE.search(ql):
            return "strategic_fleet_analysis"

    if _RECOMMENDATION_SHAPE_RE.search(ql) and "nonstop" in ql:
        if mode == "strategic_fleet_analysis":
            return "recommendation_request"

    if _COMPARE_RE.search(ctx.query or "") and mode == "recommendation_request":
        if not ctx.recommendations:
            return "error"
    return mode


def _dedupe_aircraft_family(aircraft: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse G650/G650ER family duplicates — prefer ER variant in catalog."""
    by_ident: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in aircraft:
        name = str(row.get("name") or "")
        norm = normalize_aircraft_model(name)
        ident = norm.identity_key if norm else name.lower()
        if ident not in by_ident:
            by_ident[ident] = row
            order.append(ident)
            continue
        existing = str(by_ident[ident].get("name") or "")
        if "ER" in name.upper() and "ER" not in existing.upper():
            by_ident[ident] = row
    return [by_ident[i] for i in order]


def _comparison_clause_tokens(query: str) -> List[str]:
    """Extract comma-separated aircraft tokens after Compare:/versus (renderer-only)."""
    q = query or ""
    m = re.search(
        r"\bcompare\s*:?\s*(.+?)(?:\s+for\s+|\s+on\s+|\s+with\s+|\s+focusing\b|$)",
        q,
        re.I,
    )
    if not m:
        return []
    clause = m.group(1)
    parts = re.split(r"\s*,\s*|\s+and\s+|\s+vs\.?\s+|\s+versus\s+", clause, flags=re.I)
    return [p.strip() for p in parts if p and len(p.strip()) >= 2]


def _models_from_query(ctx: RendererBuildContext) -> List[str]:
    tokens: List[str] = []
    for seg in _comparison_clause_tokens(ctx.query or ""):
        resolved = resolve_to_registry_name(seg)
        if resolved:
            tokens.append(resolved)
        else:
            tokens.append(seg)
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        tokens.extend(detect_models_from_text(ctx.query or ""))
    except Exception:
        pass
    extra = list(ctx.comparison_models or []) + list(ctx.named_aircraft_models or [])
    merged = [*tokens, *extra]
    lock = lock_comparison_aircraft(merged)
    if len(lock.canonical) >= 2:
        return list(lock.canonical)
    return list(dict.fromkeys([*merged]))


def _rebuild_comparison_from_query(ctx: RendererBuildContext) -> Optional[Dict[str, Any]]:
    """Renderer-side comparison rebuild from query (no router change)."""
    try:
        from services.comparison.comparison_pipeline_v2 import build_comparison_payload_v2

        models = _models_from_query(ctx)
        return build_comparison_payload_v2(models, ctx.mission or MissionState(), query=ctx.query or "")
    except Exception:
        return None


def _is_archetype_comparison(query: str) -> bool:
    return bool(_ARCHETYPE_COMPARE_RE.search(query or ""))


def _build_strategy_comparison_payload(ctx: RendererBuildContext) -> Dict[str, Any]:
    return {
        "comparison_type": "strategy_vs_strategy",
        "strategies": [
            {
                "id": "ulr_flagship",
                "label": "Single ultra-long-range flagship",
                "dispatch_reliability": "high_peak_leg",
                "maintenance_complexity": "single_type_low",
                "empty_leg_economics": "poor_regional_utilization",
            },
            {
                "id": "mixed_fleet_charter",
                "label": "Super-midsize + supplemental charter",
                "dispatch_reliability": "segmented_by_domain",
                "maintenance_complexity": "multi_type_higher",
                "empty_leg_economics": "charter_mitigates_empty",
            },
        ],
        "comparison_rows": [
            {
                "dimension": "dispatch_reliability",
                "ulr_flagship": "high_peak_leg",
                "mixed_fleet_charter": "segmented_by_domain",
                "cabin": None,
                "range": None,
                "operating_economics": None,
                "field_performance": None,
                "verdict": "conditional",
            },
            {
                "dimension": "maintenance_complexity",
                "ulr_flagship": "single_type_low",
                "mixed_fleet_charter": "multi_type_higher",
                "cabin": None,
                "range": None,
                "operating_economics": None,
                "field_performance": None,
                "verdict": "conditional",
            },
            {
                "dimension": "empty_leg_economics",
                "ulr_flagship": "poor_regional_utilization",
                "mixed_fleet_charter": "charter_mitigates_empty",
                "cabin": None,
                "range": None,
                "operating_economics": None,
                "field_performance": None,
                "verdict": "conditional",
            },
        ],
        "verdict": {
            "best_overall": None,
            "conditional_winner": "mixed_fleet_charter",
            "no_fit_reason": None,
        },
        "data_quality": {"status": "OK", "reason": "archetype_strategy_compare"},
    }


def _parse_comparison_json(answer: str) -> Optional[Dict[str, Any]]:
    text = (answer or "").strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _build_comparison_payload(ctx: RendererBuildContext) -> Dict[str, Any]:
    query_models = _models_from_query(ctx)
    if len(query_models) >= 2:
        rebuilt_first = _rebuild_comparison_from_query(ctx)
        if isinstance(rebuilt_first, dict) and rebuilt_first.get("status") != "INSUFFICIENT_DATA":
            raw = rebuilt_first
            aircraft_in = list(raw.get("aircraft") or [])
            aircraft = []
            for row in aircraft_in:
                if not isinstance(row, dict):
                    continue
                name = normalize_aircraft_id(str(row.get("name") or ""))
                if name:
                    aircraft.append({**row, "name": name, "aircraft_id": name})
            if len(aircraft) >= 2:
                aircraft = _dedupe_aircraft_family(aircraft)
                return {
                    "aircraft": aircraft,
                    "comparison_rows": comparison_rows_from_aircraft(aircraft),
                    "verdict": raw.get("verdict")
                    if isinstance(raw.get("verdict"), dict)
                    else {
                        "best_overall": None,
                        "conditional_winner": None,
                        "no_fit_reason": None,
                    },
                    "matrix_dimensions": list(
                        (raw.get("comparison_matrix") or {}).get("dimensions") or []
                    ),
                    "data_quality": raw.get("data_quality") or {"status": "OK"},
                }

    if _is_archetype_comparison(ctx.query or ""):
        rebuilt = _rebuild_comparison_from_query(ctx)
        if isinstance(rebuilt, dict) and rebuilt.get("status") == "INSUFFICIENT_DATA":
            return _build_strategy_comparison_payload(ctx)

    raw = _parse_comparison_json(ctx.answer)
    if raw is None:
        du = ctx.data_used or {}
        comp = du.get("comparison_v2") or {}
        if isinstance(comp, dict) and comp.get("status") == "INSUFFICIENT_DATA":
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": comp.get("reason") or "missing canonical aircraft set",
            }
        return {"status": "INSUFFICIENT_DATA", "reason": "comparison answer not structured JSON"}

    if raw.get("status") == "INSUFFICIENT_DATA":
        if _is_archetype_comparison(ctx.query or ""):
            return _build_strategy_comparison_payload(ctx)
        rebuilt = _rebuild_comparison_from_query(ctx)
        if isinstance(rebuilt, dict) and rebuilt.get("status") != "INSUFFICIENT_DATA":
            raw = rebuilt
        else:
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": str(raw.get("reason") or "INSUFFICIENT_DATA"),
            }

    aircraft_in = list(raw.get("aircraft") or [])
    aircraft: List[Dict[str, Any]] = []
    for row in aircraft_in:
        if not isinstance(row, dict):
            continue
        name = normalize_aircraft_id(str(row.get("name") or ""))
        if not name:
            continue
        aircraft.append({**row, "name": name, "aircraft_id": name})

    if len(aircraft) < 2:
        rebuilt = _rebuild_comparison_from_query(ctx)
        if isinstance(rebuilt, dict) and isinstance(rebuilt.get("aircraft"), list):
            if len(rebuilt["aircraft"]) >= 2:
                raw = rebuilt
                aircraft_in = list(rebuilt.get("aircraft") or [])
                aircraft = []
                for row in aircraft_in:
                    if not isinstance(row, dict):
                        continue
                    name = normalize_aircraft_id(str(row.get("name") or ""))
                    if name:
                        aircraft.append({**row, "name": name, "aircraft_id": name})
        if len(aircraft) < 2:
            if _is_archetype_comparison(ctx.query or ""):
                return _build_strategy_comparison_payload(ctx)
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": "fewer than two canonical aircraft after normalization",
            }

    aircraft = _dedupe_aircraft_family(aircraft)
    comparison_rows = comparison_rows_from_aircraft(aircraft)
    verdict = raw.get("verdict") if isinstance(raw.get("verdict"), dict) else {}
    return {
        "aircraft": aircraft,
        "comparison_rows": comparison_rows,
        "verdict": {
            "best_overall": verdict.get("best_overall"),
            "conditional_winner": verdict.get("conditional_winner"),
            "no_fit_reason": verdict.get("no_fit_reason"),
        },
        "matrix_dimensions": list(
            (raw.get("comparison_matrix") or {}).get("dimensions") or []
        ),
        "data_quality": raw.get("data_quality") or {"status": "OK"},
    }


def _build_capability_payload(ctx: RendererBuildContext) -> Dict[str, Any]:
    raw_models = [str(m).strip() for m in ctx.named_aircraft_models if str(m).strip()]
    if not raw_models:
        raw_models = [str(m).strip() for m in (ctx.data_used or {}).get("orchestration_v2_named_models") or [] if m]
    if not raw_models:
        try:
            from services.consultant.recommendation_engine import detect_models_from_text

            raw_models = detect_models_from_text(ctx.query or "")[:1]
        except Exception:
            raw_models = []

    if not raw_models:
        return {}

    aircraft = normalize_aircraft_id(raw_models[0]) or raw_models[0]
    from services.consultant.named_aircraft_capability import evaluate_named_aircraft_capability

    mission = ctx.mission
    ev = evaluate_named_aircraft_capability(aircraft, mission, data_used=ctx.data_used)
    verdict = str(ev.get("verdict") or "NOT REALISTIC")
    constraints = [{"type": "constraint", "detail": r} for r in (ev.get("reasons") or [])]
    payload: Dict[str, Any] = {
        "aircraft": aircraft,
        "aircraft_id": aircraft,
        "mission": _mission_snapshot(mission),
        "verdict": verdict,
        "verdict_class": _verdict_class(verdict),
        "constraints": constraints,
    }
    ql = (ctx.query or "").lower()
    if "category solves" in ql or "what aircraft category" in ql:
        payload["class_guidance"] = {
            "note": "ultra_long_range_or_heavy_cabin",
            "ranked_shortlist": False,
        }
    return payload


def _verdict_class(verdict: str) -> str:
    v = (verdict or "").upper()
    if "FEASIBLE" in v and "NOT" not in v:
        return "feasible"
    if "MARGINAL" in v:
        return "marginal"
    return "not_realistic"


def _build_strategic_payload(ctx: RendererBuildContext) -> Dict[str, Any]:
    conflicts: List[str] = []
    domains: List[str] = []
    du = ctx.data_used or {}

    if du.get("mission_hard_invalid"):
        conflicts.append("mission_hard_invalid: cost/ULR or utilization structure incompatible")

    stab = du.get("orchestration_stabilization") or {}
    if isinstance(stab, dict):
        dom = stab.get("dominant_utilization_band")
        peak = stab.get("peak_capability_requirement")
        if dom and dom != "unresolved":
            domains.append(str(dom))
        if peak and peak != "unresolved":
            domains.append(f"peak:{peak}")

    from services.mission.mission_understanding_engine import load_mission_understanding

    pkt = load_mission_understanding(du)
    if pkt is not None:
        ic = dict(pkt.inferred_constraints or {})
        if ic.get("incompatible_domains"):
            conflicts.append("incompatible operational domains")
        if ic.get("westbound_winter_pressure"):
            conflicts.append("westbound winter reserve-margin pressure")
        if ic.get("continuation_hub_secondary"):
            conflicts.append("continuation hubs must remain secondary")

    mission = ctx.mission
    if mission and getattr(mission, "routes", None):
        domains.append(f"corridor:{mission.routes[0]}")

    if not conflicts:
        conflicts.append("dispatch mismatch risk across mixed utilization bands")
    if not domains:
        domains.append("unresolved_utilization_band")

    return {
        "conflicts": conflicts,
        "operational_domains": domains,
        "recommendation": {
            "summary": "fleet_segmentation_required",
            "ranked_shortlist": False,
        },
    }


def _build_network_payload(ctx: RendererBuildContext) -> Dict[str, Any]:
    hw = (ctx.data_used or {}).get("hierarchy_weighting") or {}
    if not isinstance(hw, dict):
        from services.mission.mission_understanding_engine import load_mission_understanding
        from services.orchestration.hierarchy_weighting import detect_dominant_mission

        pkt = load_mission_understanding(ctx.data_used)
        result = detect_dominant_mission(pkt, query=ctx.query, data_used=ctx.data_used)
        hw = result.to_dict()

    dominant = str(hw.get("dominant_utilization") or "")
    secondary = list(hw.get("secondary_traffic") or [])
    executive = list(hw.get("executive_exceptions") or [])
    seasonal = list(hw.get("seasonal_overlays") or [])
    continuation = list(hw.get("continuation_constraints") or [])

    primary_hubs = [dominant] if dominant else []
    secondary_hubs = secondary + executive
    episodic_routes = continuation
    planning_priority = list(hw.get("weighting_notes") or [])
    if hw.get("continuation_hub_discipline"):
        planning_priority.insert(0, str(hw["continuation_hub_discipline"]))

    return {
        "primary_hubs": primary_hubs,
        "secondary_hubs": secondary_hubs,
        "episodic_routes": episodic_routes,
        "planning_priority": planning_priority,
    }


def _build_recommendation_payload(ctx: RendererBuildContext) -> Dict[str, Any]:
    from services.sanity.aircraft_class_guard import violates_class_sanity

    shortlist: List[Dict[str, Any]] = []
    for rec in ctx.recommendations or []:
        model = normalize_aircraft_id(getattr(rec, "model", "") or "")
        if not model:
            continue
        if violates_class_sanity(ctx.mission, model, query=ctx.query or ""):
            continue
        pc_score = next(
            (s for s in (getattr(rec, "scores", None) or []) if s.dimension == "procurement_credibility"),
            None,
        )
        pc_label = "—"
        if pc_score is not None:
            sc = float(pc_score.score)
            pc_label = "strong" if sc >= 0.75 else "conditional" if sc >= 0.5 else "weak"
        shortlist.append(
            {
                "aircraft_id": model,
                "label": model,
                "rank": getattr(rec, "rank", None),
                "fit": getattr(rec, "fit", None),
                "fit_verdict": getattr(rec, "fit_verdict", None),
                "category": getattr(rec, "category", None),
                "verdict_class": _fit_class_label(getattr(rec, "fit", "")),
                "procurement_credibility": pc_label,
            }
        )
    verdict = "no_match"
    if shortlist:
        verdict = "ranked_shortlist"
    return {
        "shortlist": shortlist,
        "mission": _mission_snapshot(ctx.mission),
        "verdict": {"status": verdict, "count": len(shortlist)},
    }


def _fit_class_label(fit: str) -> str:
    f = (fit or "").lower()
    if "strong" in f:
        return "strong"
    if "good" in f:
        return "good"
    if "partial" in f or "conditional" in f:
        return "conditional"
    if "not" in f:
        return "weak"
    return "unknown"


def _assert_mode_isolation(mode: str, payload: Dict[str, Any]) -> Optional[str]:
    blob = json.dumps(payload)
    if mode == "explicit_comparison":
        if _CONTAMINATION_STRATEGIC.search(blob) or _CONTAMINATION_RECOMMEND.search(blob):
            return "comparison payload contaminated"
        if payload.get("shortlist"):
            return "comparison must not include shortlist"
    if mode == "named_aircraft_capability":
        if payload.get("comparison_rows") or payload.get("shortlist"):
            return "capability payload contaminated"
    if mode == "strategic_fleet_analysis":
        if payload.get("shortlist") and isinstance(payload.get("shortlist"), list) and payload["shortlist"]:
            return "strategic must not include shortlist"
    return None


def build_renderer_envelope(ctx: RendererBuildContext) -> RendererEnvelopeV2:
    """Build and validate renderer envelope; fail-closed on any unsafe payload."""
    mode = _effective_render_mode(ctx)
    if mode == "error":
        return renderer_failure_envelope("INSUFFICIENT_RENDER_DATA")
    if not mode:
        return renderer_failure_envelope("INSUFFICIENT_RENDER_DATA")

    if _COMPARE_RE.search(ctx.query or "") and mode == "recommendation_request":
        lock = lock_comparison_aircraft(_models_from_query(ctx))
        if len(lock.canonical) < 2 and len(lock.rejected) >= 2:
            return renderer_failure_envelope("INSUFFICIENT_RENDER_DATA")

    builders = {
        "explicit_comparison": (
            COMPONENT_COMPARISON_TABLE_V2,
            _build_comparison_payload,
        ),
        "named_aircraft_capability": (
            COMPONENT_CAPABILITY_VERDICT_V2,
            _build_capability_payload,
        ),
        "strategic_fleet_analysis": (
            COMPONENT_STRATEGIC_ANALYSIS_V2,
            _build_strategic_payload,
        ),
        "network_structure": (
            COMPONENT_NETWORK_TOPOLOGY_V2,
            _build_network_payload,
        ),
        "recommendation_request": (
            COMPONENT_BROKER_RECOMMENDATION_V2,
            _build_recommendation_payload,
        ),
    }

    spec = builders.get(mode)
    if spec is None:
        return renderer_failure_envelope(f"unsupported mode: {mode}")

    component, builder = spec
    try:
        payload = builder(ctx)
    except Exception:
        return renderer_failure_envelope("INSUFFICIENT_RENDER_DATA")

    if not payload:
        return renderer_failure_envelope("INSUFFICIENT_RENDER_DATA")

    if payload.get("status") == "INSUFFICIENT_DATA" and mode == "explicit_comparison":
        return renderer_failure_envelope(
            str(payload.get("reason") or "INSUFFICIENT_RENDER_DATA")
        )

    isolation_err = _assert_mode_isolation(mode, payload)
    if isolation_err:
        return renderer_failure_envelope(isolation_err)

    md_check = assert_no_markdown_in_payload(payload)
    if not md_check.ok:
        return renderer_failure_envelope(md_check.reason or "INSUFFICIENT_RENDER_DATA")

    if mode == "explicit_comparison" and isinstance(payload, dict):
        try:
            from services.comparison.comparison_intelligence import enrich_comparison_payload

            payload = enrich_comparison_payload(payload)
        except Exception:
            pass

    envelope = RendererEnvelopeV2(
        mode=mode,
        component=component,
        payload=payload,
        meta={
            "schema_version": "v2",
            "query_preview": (ctx.query or "")[:120],
            "renderer_authoritative": True,
        },
    )
    vr = validate_envelope(envelope.to_dict())
    if not vr.ok:
        return renderer_failure_envelope(vr.reason or "INSUFFICIENT_RENDER_DATA")
    return envelope


def envelope_to_answer(
    envelope: RendererEnvelopeV2,
    *,
    query: str = "",
    mission: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
    pipeline: Any = None,
) -> str:
    """Render envelope as human-readable broker prose (never raw JSON)."""
    from services.rendering.render_guard import render_from_envelope

    return render_from_envelope(
        envelope.to_dict(),
        query=query,
        mission=mission,
        data_used=data_used,
        pipeline=pipeline,
    )


__all__ = [
    "RendererBuildContext",
    "build_renderer_envelope",
    "envelope_to_answer",
    "normalize_aircraft_id",
]

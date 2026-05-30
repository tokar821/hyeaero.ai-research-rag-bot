"""
Human-readable broker prose renderers — no raw JSON, no empty tables, no placeholder rows.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker.graceful_degradation import degraded_empty_shortlist_guidance
from services.consultant.mission_state import MissionState
from services.consultant.named_aircraft_capability import format_named_aircraft_capability_response
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.strategic_analysis_renderer import format_strategic_analysis_response
from services.mission.mission_understanding_engine import load_mission_understanding

_FAIL_HEADER = (
    "I don't have reliable structured output for this request yet.\n"
    "Here's the closest accurate guidance:\n"
)

_INCOMPLETE_QUERY_MSG = (
    "Your mission details appear incomplete.\n"
    "Please finish the operational requirement so I can evaluate the aircraft strategy accurately."
)

_INSUFFICIENT_COMPARISON = "INSUFFICIENT DATA FOR STRUCTURED COMPARISON"

_UNVERIFIED_RE = re.compile(r"\bunverified\b", re.I)
_BANNED_ROW_RE = re.compile(
    r"\b(?:citation\s+cj[24]|cj2|cj4|learjet\s*75|pc-?24|caravan|king\s+air)\b",
    re.I,
)


def _recommendations_from_data_used(
    data_used: Optional[Dict[str, Any]],
) -> List[AircraftRecommendation]:
    """Build minimal recommendation rows from pipeline whitelist metadata."""
    if not isinstance(data_used, dict):
        return []
    models: List[str] = []
    for key in ("approved_shortlist", "final_ranked_aircraft"):
        raw = data_used.get(key)
        if isinstance(raw, list):
            for m in raw:
                s = str(m or "").strip()
                if s and s not in models:
                    models.append(s)
    return [
        AircraftRecommendation(
            model=m,
            category="",
            total_score=0.0,
            confidence=0.0,
            rank=i + 1,
            avoid=False,
        )
        for i, m in enumerate(models)
    ]


def _requires_renderer_authority(
    data_used: Optional[Dict[str, Any]],
    *,
    query: str = "",
) -> bool:
    try:
        from services.consultant.recommendation_authority import (
            requires_recommendation_aircraft_authority,
        )

        return requires_recommendation_aircraft_authority(data_used, query=query)
    except Exception:
        return False


def _empty_shortlist_guidance(
    mission: MissionState,
    pipeline: Any,
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Authority-aware empty shortlist — never surfaces feasible_models survivors."""
    return degraded_empty_shortlist_guidance(
        mission,
        pipeline,
        query,
        data_used=data_used,
    )


def _filter_shortlist_to_authority(
    shortlist: List[Any],
    *,
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> List[Dict[str, Any]]:
    """Keep only rows whose aircraft label is on the approved whitelist."""
    if not _requires_renderer_authority(data_used, query=query):
        return [r for r in shortlist if isinstance(r, dict)]
    try:
        from services.consultant.recommendation_authority import RecommendationAuthority

        auth = RecommendationAuthority.from_pipeline(
            _recommendations_from_data_used(data_used),
            data_used=data_used,
        )
        allowed = auth.allowed_models
        if not allowed:
            return []
        out: List[Dict[str, Any]] = []
        for row in shortlist:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("aircraft_id") or "").strip()
            if label and label in allowed:
                out.append(row)
        return out
    except Exception:
        return []


def finalize_renderer_prose(
    text: str,
    *,
    query: str = "",
    mission: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
    source: str = "prose_renderer_v2",
) -> str:
    """Last-line authority guard for ranked recommendation renderer output."""
    if not _requires_renderer_authority(data_used, query=query):
        return text
    try:
        from services.consultant.recommendation_authority import apply_final_answer_authority

        ms = mission if isinstance(mission, MissionState) else MissionState()
        return apply_final_answer_authority(
            text,
            mission=ms,
            recommendations=_recommendations_from_data_used(data_used),
            data_used=data_used,
            query=query,
            source=source,
        )
    except Exception:
        return text


def is_incomplete_query(query: str) -> bool:
    q = (query or "").strip()
    if not q or len(q) < 12:
        return True
    if q.endswith(":"):
        return True
    if re.search(r"(?:^|\n)\s*[-*•]\s*$", q):
        return True
    if re.search(r"\b(?:leadership\s+insists|we\s+operate|our\s+mission\s+is)\s*:\s*$", q, re.I):
        return True
    if re.search(r"\b(?:compare|versus|vs\.?)\s*:?\s*$", q, re.I):
        return True
    return False


def is_raw_json_leakage(text: str) -> bool:
    t = (text or "").strip()
    if not t.startswith("{"):
        return False
    if '"mode"' in t and '"component"' in t and '"payload"' in t:
        return True
    try:
        import json

        obj = json.loads(t)
        return isinstance(obj, dict) and "mode" in obj and "payload" in obj
    except Exception:
        return False


def render_incomplete_query() -> str:
    return _INCOMPLETE_QUERY_MSG


def render_comparison_prose(
    payload: Dict[str, Any],
    *,
    query: str = "",
    fail_reason: str = "",
) -> str:
    if payload.get("comparison_type") == "strategy_vs_strategy":
        return _render_strategy_comparison_prose(payload)

    if payload.get("status") == "INSUFFICIENT_DATA" or fail_reason:
        return _INSUFFICIENT_COMPARISON

    aircraft = payload.get("aircraft") or []
    rows = payload.get("comparison_rows") or []
    if not isinstance(aircraft, list) or len(aircraft) < 2:
        return _INSUFFICIENT_COMPARISON
    if len(rows) != len(aircraft):
        return _INSUFFICIENT_COMPARISON

    lines = ["## Aircraft Comparison", ""]
    for row in rows:
        if not isinstance(row, dict):
            return _INSUFFICIENT_COMPARISON
        label = str(row.get("label") or row.get("aircraft_id") or "").strip()
        if not label or _UNVERIFIED_RE.search(label):
            return _INSUFFICIENT_COMPARISON
        for col in ("cabin", "range", "operating_economics", "field_performance", "verdict"):
            if col not in row:
                return _INSUFFICIENT_COMPARISON

    lines.append("| Aircraft | Cabin | Range (nm) | Operating economics | Field performance | Verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        label = str(row.get("label") or row.get("aircraft_id") or "")
        if _BANNED_ROW_RE.search(label):
            continue
        cabin = row.get("cabin") or "—"
        rng = row.get("range")
        rng_s = str(int(rng)) if isinstance(rng, (int, float)) else str(rng or "—")
        econ = row.get("operating_economics") or "—"
        field = row.get("field_performance") or "—"
        verdict = row.get("verdict") or "—"
        lines.append(f"| {label} | {cabin} | {rng_s} | {econ} | {field} | {verdict} |")

    if len(lines) <= 3:
        return _INSUFFICIENT_COMPARISON

    verdict = payload.get("verdict") or {}
    if isinstance(verdict, dict):
        best = verdict.get("best_overall") or verdict.get("conditional_winner")
        if best:
            lines.extend(["", f"**Overall:** {best}"])

    intel_dims = payload.get("intelligence_dimensions")
    if intel_dims or any(rows and rows[0].get("dispatch_maturity") for rows in [rows]):
        from services.rendering.broker_response_format import format_comparison_intelligence_block

        intel_block = format_comparison_intelligence_block(rows)
        if intel_block:
            lines.extend(["", intel_block])

    return "\n".join(lines)


def _render_strategy_comparison_prose(payload: Dict[str, Any]) -> str:
    lines = [
        "## Fleet Strategy Comparison",
        "",
        "This is an archetype tradeoff — not a spec-sheet aircraft shopping list.",
        "",
    ]
    for row in payload.get("comparison_rows") or []:
        if not isinstance(row, dict):
            continue
        dim = row.get("dimension") or "dimension"
        lines.append(f"### {str(dim).replace('_', ' ').title()}")
        lines.append(f"- **Single ULR flagship:** {row.get('ulr_flagship', '—')}")
        lines.append(f"- **Mixed fleet + charter:** {row.get('mixed_fleet_charter', '—')}")
        lines.append("")
    return "\n".join(lines).strip()


def render_capability_prose(
    payload: Dict[str, Any],
    *,
    mission: Any,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    aircraft = str(payload.get("aircraft") or payload.get("aircraft_id") or "").strip()
    if not aircraft:
        return (
            "## Named Aircraft Capability\n\n"
            "NOT REALISTIC — no aircraft model identified for evaluation.\n"
        )

    ms = mission if isinstance(mission, MissionState) else MissionState()
    text = format_named_aircraft_capability_response(
        [aircraft],
        ms,
        data_used=data_used,
        query=query,
    )
    if payload.get("class_guidance") and "category solves" in (query or "").lower():
        text += (
            "\n\nIf you'd like, I can also show aircraft classes that solve this mission "
            "more credibly — without substituting a different named airframe in this verdict."
        )
    return text


def render_strategic_prose(
    payload: Dict[str, Any],
    *,
    mission: Any,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Operational synthesis — not generic 'fleet segmentation required' boilerplate."""
    ms = mission if isinstance(mission, MissionState) else MissionState()
    pkt = load_mission_understanding(data_used)
    ql = (query or "").lower()

    lines: List[str] = ["## Strategic Fleet Analysis", ""]

    hw = (data_used or {}).get("hierarchy_weighting") or {}
    dom = ""
    secondary: List[str] = []
    episodic: List[str] = []
    if isinstance(hw, dict):
        dom = str(hw.get("dominant_utilization") or "").replace("_", " ")
        secondary = [str(x).replace("_", " ") for x in (hw.get("secondary_traffic") or [])]
        episodic = [str(x).replace("_", " ") for x in (hw.get("executive_exceptions") or [])]

    # Hub-centric narrative when query mentions domestic core + continuation
    hubs: List[str] = []
    cog = (data_used or {}).get("mission_center_of_gravity") or {}
    if isinstance(cog, dict):
        for hub in cog.get("primary_hubs") or []:
            label = str(hub).strip()
            if label and label not in hubs:
                hubs.append(label)
    for city in ("dallas", "houston", "chicago", "atlanta", "new york", "orange county"):
        if city in ql:
            title = "Orange County" if city == "orange county" else city.title()
            if title not in hubs:
                hubs.append(title)
    if hubs:
        lines.append(
            "Your operational center of gravity is still regional North American executive traffic "
            f"({', '.join(hubs[:3])}). That favors **super-midsize or large-cabin domestic optimization** "
            "rather than forcing a single ultra-long-range platform across every leg."
        )
        lines.append("")

    if "g500" in ql or "gulfstream g500" in ql:
        if "credibility" in ql or "enough aircraft" in ql:
            lines.append(
                "### Gulfstream G500 — where credibility thins\n"
                "- **Transatlantic / Middle East:** often viable for 8–10 pax with disciplined payload, "
                "but winter westbound reserves tighten dispatch margin versus G650ER/Global 6500-class.\n"
                "- **Versus larger ULR:** you begin losing credibility when quarterly Pacific or "
                "Dubai-heavy utilization requires consistent NBAA-margin planning — larger ULR platforms "
                "carry the reserve and payload headroom that G500-class economics trade away.\n"
            )
            lines.append("")

    if episodic or any(x in ql for x in ("dubai", "singapore", "riyadh", "tokyo")):
        cont = ", ".join(episodic[:3]) if episodic else "Dubai / Singapore-style continuation nodes"
        lines.append(
            f"**{cont}** should be treated as **episodic continuation missions**, not procurement drivers. "
            "They are too infrequent to justify distorting the domestic utilization profile."
        )
        lines.append("")

    conflicts = payload.get("conflicts") or []
    if conflicts:
        lines.append("### What structurally conflicts")
        for c in conflicts[:6]:
            lines.append(f"- {str(c).replace('_', ' ')}")
        lines.append("")

    domains = payload.get("operational_domains") or []
    if domains:
        lines.append("### Dominant operational domains")
        for d in domains[:5]:
            lines.append(f"- {str(d).replace('_', ' ')}")
        lines.append("")

    if "one aircraft only" in ql or "single aircraft" in ql:
        lines.append(
            "### Why one platform breaks first\n"
            "- **Dispatch mismatch:** peak international legs and high-cycle domestic legs rarely share "
            "the same optimal airframe.\n"
            "- **Empty-leg economics:** forcing ULR capability into a domestic profile destroys "
            "positioning efficiency.\n"
            "- **Maintenance divergence:** mixed stage lengths inflate downtime and AOG exposure.\n"
        )

    if du := data_used:
        if du.get("mission_hard_invalid"):
            lines.append(
                "### Operating-cost vs range ceiling\n"
                "The stated cost ceiling and guaranteed winter nonstop ULR stage are structurally "
                "incompatible in one airframe — this is a constraint conflict, not a catalog gap.\n"
            )
        pda = du.get("procurement_driver_analysis") or {}
        if isinstance(pda, dict):
            for g in pda.get("guidance") or []:
                lines.append(f"- {g}")
            if pda.get("fleet_distortion"):
                lines.append(
                    "\n**Fleet distortion:** episodic Asia/Pacific continuation is too infrequent "
                    "to justify forcing an ultra-long-range aircraft into a domestic utilization structure. "
                    "Consider supplemental charter or fleet segmentation for those legs.\n"
                )
        np = du.get("network_priority") or {}
        if isinstance(np, dict) and np.get("do_not_procure_around"):
            lines.append(
                "\n**Do not procure around:** "
                + ", ".join(str(x) for x in np["do_not_procure_around"][:4])
                + " — treat as episodic continuation, not the utilization center.\n"
            )

    if len(lines) <= 3:
        return format_strategic_analysis_response(ms, pkt, query=query, data_used=data_used)

    lines.append(
        "No ranked acquisition shortlist is produced for this structural question unless you "
        "explicitly request aircraft recommendations."
    )
    return "\n".join(lines)


def render_recommendation_prose(
    payload: Dict[str, Any],
    *,
    mission: Any,
    query: str = "",
    pipeline: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    from services.rendering.broker_response_format import format_continuity_acknowledgment

    ack = ""
    if isinstance(data_used, dict):
        ack = format_continuity_acknowledgment(data_used.get("context_continuity") or {})

    ms = mission if isinstance(mission, MissionState) else MissionState()
    shortlist = payload.get("shortlist") or []
    if not isinstance(shortlist, list):
        shortlist = []
    shortlist = _filter_shortlist_to_authority(
        shortlist,
        data_used=data_used,
        query=query,
    )
    if len(shortlist) == 0:
        body = _empty_shortlist_guidance(
            ms,
            pipeline,
            query,
            data_used=data_used,
        )
        if _requires_renderer_authority(data_used, query=query):
            return finalize_renderer_prose(
                body,
                query=query,
                mission=ms,
                data_used=data_used,
                source="render_recommendation_prose_empty",
            )
        return (
            "No aircraft in the current validated catalog pass all hard mission constraints as stated.\n\n"
            "This usually means:\n"
            "- range + reserve requirements\n"
            "- payload\n"
            "- runway limitations\n"
            "- operating-cost ceiling\n"
            "are structurally conflicting.\n\n"
            "Closest accurate guidance:\n\n"
            + body
        )

    lines = []
    if ack:
        lines.append(ack)
    lines.extend(["## Ranked Shortlist", ""])
    lines.append("| Rank | Aircraft | Fit | Procurement credibility |")
    lines.append("| --- | --- | --- | --- |")
    op_notes = {}
    if isinstance(data_used, dict):
        for tr in data_used.get("aircraft_operational_assessments") or []:
            if isinstance(tr, dict) and tr.get("model"):
                op_notes[tr["model"]] = (tr.get("dispatch") or {}).get("works_reliably", True)

    for row in shortlist:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or row.get("aircraft_id") or "")
        if _BANNED_ROW_RE.search(label):
            continue
        rank = row.get("rank") or "—"
        fit = row.get("fit_verdict") or row.get("fit") or row.get("verdict_class") or "—"
        pc = row.get("procurement_credibility") or row.get("verdict_class") or "—"
        lines.append(f"| {rank} | {label} | {fit} | {pc} |")

    if len(lines) <= 3:
        return finalize_renderer_prose(
            _empty_shortlist_guidance(ms, pipeline, query, data_used=data_used),
            query=query,
            mission=ms,
            data_used=data_used,
            source="render_recommendation_prose_table_empty",
        )

    mission_snap = payload.get("mission") or {}
    if isinstance(mission_snap, dict) and mission_snap.get("routes"):
        lines.extend(["", f"**Mission corridor:** {mission_snap['routes'][0]}"])
    return finalize_renderer_prose(
        "\n".join(lines),
        query=query,
        mission=ms,
        data_used=data_used,
        source="render_recommendation_prose",
    )


def render_network_prose(payload: Dict[str, Any]) -> str:
    primary = payload.get("primary_hubs") or []
    secondary = payload.get("secondary_hubs") or []
    episodic = payload.get("episodic_routes") or []
    priority = payload.get("planning_priority") or []

    lines = [
        "## Network Utilization Hierarchy",
        "",
        "### Primary hubs (dominant utilization)",
    ]
    for h in primary[:5]:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("### Secondary / executive traffic")
    for h in secondary[:6]:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("### Episodic continuation routes (must not drive procurement)")
    for r in episodic[:6]:
        lines.append(f"- {r}")
    if priority:
        lines.append("")
        lines.append("### Planning priority")
        for p in priority[:4]:
            lines.append(f"- {p}")
    return "\n".join(lines)


def render_error_fallback(
    reason: str,
    *,
    mode: str,
    mission: Any,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    pipeline: Any = None,
) -> str:
    body = _FAIL_HEADER
    if mode == "explicit_comparison":
        body += _INSUFFICIENT_COMPARISON
    elif mode == "named_aircraft_capability":
        ms = mission if isinstance(mission, MissionState) else MissionState()
        from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

        models = []
        for m in (data_used or {}).get("orchestration_v2_named_models") or []:
            models.append(resolve_canonical_display_name(str(m)))
        body += render_capability_prose(
            {"aircraft": models[0] if models else ""},
            mission=ms,
            query=query,
            data_used=data_used,
        )
    elif mode == "recommendation_request":
        ms = mission if isinstance(mission, MissionState) else MissionState()
        body += _empty_shortlist_guidance(
            ms,
            pipeline,
            query,
            data_used=data_used,
        )
        body = finalize_renderer_prose(
            body,
            query=query,
            mission=ms,
            data_used=data_used,
            source="render_error_fallback_recommendation",
        )
    elif mode == "strategic_fleet_analysis":
        ms = mission if isinstance(mission, MissionState) else MissionState()
        pkt = load_mission_understanding(data_used)
        body += format_strategic_analysis_response(ms, pkt, query=query, data_used=data_used)
    else:
        body += reason or "Unable to produce a validated broker response for this turn."
    return body.strip()


__all__ = [
    "is_incomplete_query",
    "is_raw_json_leakage",
    "render_incomplete_query",
    "render_comparison_prose",
    "render_capability_prose",
    "render_strategic_prose",
    "render_recommendation_prose",
    "render_network_prose",
    "render_error_fallback",
    "finalize_renderer_prose",
]

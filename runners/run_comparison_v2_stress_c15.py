#!/usr/bin/env python3
"""Comparison v2 structured output stress — C01–C15."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.comparison.comparison_pipeline_v2 import run_comparison_v2
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import detect_models_from_text
from services.orchestration.orchestration_router_v2 import route_orchestration_v2
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

SCENARIOS: List[Dict[str, Any]] = [
    {"id": "C01", "group": "strict", "expected": "OK", "query": (
        "Compare Gulfstream G650ER vs Bombardier Global 7500 vs Dassault Falcon 8X for long-range "
        "executive missions, focusing on winter westbound reliability and cost efficiency."
    )},
    {"id": "C02", "group": "strict", "expected": "OK", "query": (
        "Which performs better for mixed missions: Embraer Praetor 600, Bombardier Challenger 650, "
        "Gulfstream G280 across US domestic + occasional Europe?"
    )},
    {"id": "C03", "group": "strict", "expected": "INSUFFICIENT_DATA", "query": (
        "Compare G500 vs G600 vs Global 5500 in terms of cost per hour, range efficiency, "
        "and passenger comfort for corporate usage."
    )},
    {"id": "C04", "group": "edge", "expected": "INSUFFICIENT_DATA", "query": (
        "Praetor 600 vs Citation Longitude vs Challenger 3500 — which is most viable for "
        "coast-to-coast + occasional transatlantic operations?"
    )},
    {"id": "C05", "group": "edge", "expected": "INSUFFICIENT_DATA", "query": (
        "Global 6000 vs Falcon 8X vs Gulfstream G650 — which is most efficient for "
        "NYC–London winter operations with 10 passengers?"
    )},
    {"id": "C06", "group": "edge", "expected": "INSUFFICIENT_DATA", "query": (
        "Challenger 650 vs Global 6500 vs G500 — compare for mixed domestic + international executive usage."
    )},
    {"id": "C07", "group": "insufficient", "expected": "INSUFFICIENT_DATA", "query": (
        "Compare aircraft for Arctic gravel operations, Caribbean hops, and Asia-Pacific executive travel."
    )},
    {"id": "C08", "group": "insufficient", "expected": "INSUFFICIENT_DATA", "query": (
        'Which aircraft is better: unknown "regional jet class A", super-midsize class, '
        "ultra-long-range class for mixed global operations?"
    )},
    {"id": "C09", "group": "insufficient", "expected": "INSUFFICIENT_DATA", "query": (
        "Compare best aircraft for: low cost, maximum range, short runway access, 12 passengers simultaneously."
    )},
    {"id": "C10", "group": "registry", "expected": "OK", "query": (
        "Compare: Gulfstream G650, G650ER, Global 7500, Bombardier Global Seven Five Zero Zero."
    )},
    {"id": "C11", "group": "registry", "expected": "OK", "query": (
        'Falcon 7X vs Falcon 8X vs Falcon "Eight X Extended Range" — which is operationally superior?'
    )},
    {"id": "C12", "group": "mixed", "expected": "OK", "query": (
        "We mostly fly domestic US routes but occasionally go to Tokyo and London. Compare: "
        "G500 vs Global 7500 vs Falcon 8X — but also tell me if one aircraft strategy is better."
    )},
    {"id": "C13", "group": "mixed", "expected": "OK", "query": (
        "Compare Challenger 650 vs Praetor 600 — and also tell me what fleet strategy we should use."
    )},
    {"id": "C14", "group": "format", "expected": "OK", "query": (
        "Compare G600 vs G650ER vs Global 6000 — output must be strictly structured. "
        "No explanation, no prose, only structured comparison."
    )},
    {"id": "C15", "group": "invalid", "expected": "INSUFFICIENT_DATA", "query": (
        "Compare aircraft that can simultaneously: land on Arctic gravel strips, "
        "fly nonstop Tokyo–LA westbound in winter, carry 14 passengers, "
        "minimize operating cost below midsize jet level."
    )},
]

_CONTAMINATION_RE = re.compile(
    r"\b(?:STRATEGIC ANALYSIS|fleet strategy|network structure|Named Aircraft Capability|"
    r"##\s*Ranked|recommendation)\b",
    re.I,
)
_MARKDOWN_TABLE_RE = re.compile(r"\|\s*Aircraft\s*\|", re.I)
_PROSE_PREFIX_RE = re.compile(r"directional rather than catalog", re.I)


def _mission() -> MissionState:
    return MissionState(passenger_count=8, nonstop_requirement=True)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t.startswith("{"):
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def _evaluate(answer: str, expected: str) -> Dict[str, Any]:
    issues: List[str] = []
    data = _parse_json(answer)
    if data is None:
        return {"pass": False, "issues": ["not_valid_json"], "status": None}

    if _PROSE_PREFIX_RE.search(answer) or _MARKDOWN_TABLE_RE.search(answer):
        issues.append("hybrid_or_legacy_table")
    if _CONTAMINATION_RE.search(answer):
        issues.append("mode_contamination")

    if data.get("status") == "INSUFFICIENT_DATA":
        status = "INSUFFICIENT_DATA"
        if data.get("mode") != "explicit_comparison":
            issues.append("bad_insufficient_mode")
        if not str(data.get("reason") or "").strip():
            issues.append("missing_reason")
    elif data.get("mode") == "explicit_comparison" and isinstance(data.get("aircraft"), list):
        status = "OK"
        dq = (data.get("data_quality") or {}).get("status")
        if dq != "OK":
            issues.append(f"data_quality={dq}")
        if len(data["aircraft"]) < 2:
            issues.append("fewer_than_two_aircraft")
        for row in data["aircraft"]:
            name = str((row or {}).get("name") or "")
            if re.search(r"unverified|placeholder", name, re.I):
                issues.append(f"banned_name:{name}")
    else:
        status = "UNKNOWN"
        issues.append("unknown_payload_shape")

    exp = expected
    pass_ = status == exp and not issues
    return {"pass": pass_, "status": status, "issues": issues, "aircraft_count": len(data.get("aircraft") or [])}


def main() -> None:
    results: List[Dict[str, Any]] = []
    ok = 0
    for sc in SCENARIOS:
        q = sc["query"]
        route = route_orchestration_v2(q)
        models = list(route.preserve_comparison_models) or detect_models_from_text(q)
        direct = run_comparison_v2(
            query=q,
            mission=_mission(),
            compare_models=models,
            mode="explicit_comparison",
        )
        ev_direct = _evaluate(direct, sc["expected"])

        e2e_answer = ""
        e2e_ev: Dict[str, Any] = {}
        try:
            e2e = run_consultant_orchestration(q)
            e2e_answer = (e2e.answer or "").strip()
            e2e_ev = _evaluate(e2e_answer, sc["expected"])
        except Exception as exc:
            e2e_ev = {"pass": False, "issues": [f"e2e_error:{exc}"], "status": None}

        layer_pass = ev_direct["pass"]
        e2e_applicable = route.query_type.value == "explicit_comparison"
        e2e_pass = e2e_ev.get("pass") if e2e_applicable else None
        overall_pass = layer_pass and (e2e_pass is True or e2e_pass is None)

        row = {
            "id": sc["id"],
            "group": sc["group"],
            "expected": sc["expected"],
            "router": route.query_type.value,
            "models": models,
            "layer_pass": layer_pass,
            "layer_status": ev_direct.get("status"),
            "layer_issues": ev_direct.get("issues"),
            "e2e_applicable": e2e_applicable,
            "e2e_pass": e2e_pass,
            "e2e_status": e2e_ev.get("status"),
            "e2e_issues": e2e_ev.get("issues"),
            "answer_preview": (e2e_answer if e2e_applicable else direct)[:280],
        }
        if overall_pass:
            ok += 1
        row["pass"] = overall_pass
        results.append(row)

    out = {
        "summary": {
            "pass": f"{ok}/15",
            "layer_pass": f"{sum(1 for r in results if r['layer_pass'])}/15",
            "e2e_pass_when_routed": (
                f"{sum(1 for r in results if r.get('e2e_pass') is True)}/"
                f"{sum(1 for r in results if r.get('e2e_applicable'))}"
            ),
        },
        "results": results,
    }
    path = _ROOT / "evals" / "comparison_v2_stress_c15_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

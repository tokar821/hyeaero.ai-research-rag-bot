#!/usr/bin/env python3
"""Structured output / renderer stress suite R01–R15."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "R01",
        "query": (
            "Compare the Gulfstream G280, Embraer Praetor 600, and Bombardier Challenger 650 for: "
            "coast-to-coast U.S. travel, occasional westbound Europe missions, operating cost sensitivity."
        ),
        "expect_mode": "explicit_comparison",
        "expect_component": "comparison_table_v2",
        "min_aircraft": 3,
        "max_aircraft": 3,
    },
    {
        "id": "R02",
        "query": (
            "Compare: Global Seven Five Zero Zero, G650ER, Falcon Eight X "
            "for nonstop Los Angeles to Tokyo winter missions."
        ),
        "expect_mode": "explicit_comparison",
        "expect_component": "comparison_table_v2",
        "expect_names": ["Global 7500", "Gulfstream G650ER", "Falcon 8X"],
    },
    {
        "id": "R03",
        "query": "Compare: Bombardier Global 5500, G600, Falcon 6X for executive Europe missions.",
        "expect_mode": ("error", "explicit_comparison"),
        "allow_insufficient": True,
        "fail_closed": True,
    },
    {
        "id": "R04",
        "query": (
            "Could a Falcon 8X reliably fly Los Angeles to London westbound in winter "
            "with 10 passengers and NBAA IFR reserves?"
        ),
        "expect_mode": "named_aircraft_capability",
        "expect_component": "capability_verdict_v2",
        "no_shortlist": True,
        "no_comparison_rows": True,
    },
    {
        "id": "R05",
        "query": (
            "We operate Arctic gravel strips, Caribbean executive routes, New York to London traffic, "
            "seasonal Tokyo continuation flights. Leadership wants one aircraft only. "
            "What structurally breaks first?"
        ),
        "expect_mode": "strategic_fleet_analysis",
        "expect_component": "strategic_analysis_v2",
        "no_shortlist": True,
    },
    {
        "id": "R06",
        "query": (
            "Most annual utilization is Dallas, Houston, Chicago. Executives occasionally continue "
            "to Dubai and Singapore. How should the operational hierarchy actually be represented?"
        ),
        "expect_mode": "network_structure",
        "expect_component": "network_topology_v2",
        "no_shortlist": True,
    },
    {
        "id": "R07",
        "query": (
            "I need a jet for 9 passengers that can reliably fly Los Angeles to London nonstop year-round "
            "while keeping operating costs below Global 7500 levels. What realistically survives after "
            "filtering out marginal-range aircraft?"
        ),
        "expect_mode": ("recommendation_request", "strategic_fleet_analysis"),
        "expect_component": ("broker_recommendation_v2", "strategic_analysis_v2"),
        "needs_shortlist_or_strategic": True,
    },
    {
        "id": "R08",
        "query": (
            "Compare the Falcon 8X and Global 7500, but also explain whether our overall fleet strategy is wrong."
        ),
        "expect_mode": ("explicit_comparison", "strategic_fleet_analysis"),
        "single_mode_only": True,
    },
    {
        "id": "R09",
        "query": "Compare: aircraft_alpha_unknown, aircraft_beta_unknown, aircraft_gamma_unknown",
        "expect_mode": "error",
        "fail_closed": True,
    },
    {
        "id": "R10",
        "query": (
            "Compare: a single ultra-long-range flagship aircraft vs a mixed fleet using super-midsize "
            "aircraft plus supplemental charter. Focus on dispatch reliability, maintenance complexity, "
            "empty-leg economics."
        ),
        "expect_mode": "explicit_comparison",
        "expect_component": "comparison_table_v2",
        "strategy_compare": True,
    },
    {
        "id": "R11",
        "query": (
            "Could a Global 6000 realistically fly San Francisco to Tokyo westbound in winter with 12 passengers? "
            "If not, what aircraft category solves the problem more credibly?"
        ),
        "expect_mode": ("named_aircraft_capability", "recommendation_request"),
        "no_ranked_table": True,
    },
    {
        "id": "R12",
        "query": (
            "Compare the Praetor 600, Challenger 650, and Gulfstream G280 for Aspen winter operations, "
            "coast-to-coast reliability, occasional Europe continuation."
        ),
        "expect_mode": "explicit_comparison",
        "min_aircraft": 3,
    },
    {
        "id": "R13",
        "query": "What aircraft can fly Aspen to London nonstop with 8 passengers in winter?",
        "expect_mode": "recommendation_request",
        "expect_component": "broker_recommendation_v2",
        "needs_structured": True,
    },
    {
        "id": "R14",
        "query": "Compare: Gulfstream G650, G650ER, Global 7500",
        "expect_mode": "explicit_comparison",
        "max_aircraft": 3,
        "expect_unique": True,
    },
    {
        "id": "R15",
        "query": (
            "I want much lower operating costs than a Global 7500, guaranteed nonstop Sydney westbound in winter, "
            "one aircraft only, Aspen capability, 14 passengers."
        ),
        "expect_mode": ("strategic_fleet_analysis", "error"),
        "no_shortlist": True,
        "no_banned_models": ["Citation CJ", "Learjet", "PC-12"],
    },
]

_MARKDOWN_RE = re.compile(r"\|\s*(?:Rank|Aircraft)\s*\|", re.I)
_PROSE_HEADER_RE = re.compile(r"^##\s+", re.M)


def _parse_envelope(answer: str) -> Optional[Dict[str, Any]]:
    t = (answer or "").strip()
    if not t.startswith("{"):
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def _evaluate(sc: Dict[str, Any], env: Optional[Dict[str, Any]], raw: str) -> Dict[str, Any]:
    issues: List[str] = []
    if env is None:
        return {"pass": False, "issues": ["not_json_envelope"], "mode": None}

    mode = env.get("mode")
    component = env.get("component")
    payload = env.get("payload") or {}

    exp_modes = sc.get("expect_mode")
    if isinstance(exp_modes, str):
        exp_modes = (exp_modes,)
    if exp_modes and mode not in exp_modes:
        issues.append(f"mode={mode} expected {exp_modes}")

    exp_comp = sc.get("expect_component")
    if isinstance(exp_comp, str):
        exp_comp = (exp_comp,)
    if exp_comp and component not in exp_comp:
        issues.append(f"component={component} expected {exp_comp}")

    if raw.strip().startswith("{") and '"payload"' in raw:
        issues.append("raw_json_user_answer")
    if _MARKDOWN_RE.search(raw) and sc.get("expect_mode") == "explicit_comparison":
        pass  # comparison prose tables are allowed
    elif _MARKDOWN_RE.search(raw) and mode not in ("explicit_comparison",):
        if "Rank" in raw and mode != "recommendation_request":
            issues.append("markdown_or_prose_leak")

    if sc.get("fail_closed") and mode != "error":
        if not (sc.get("allow_insufficient") and payload.get("status") == "INSUFFICIENT_DATA"):
            issues.append("expected fail_closed error")

    if sc.get("no_shortlist"):
        sl = payload.get("shortlist")
        if isinstance(sl, list) and len(sl) > 0:
            issues.append("unexpected shortlist")

    if sc.get("no_comparison_rows") and payload.get("comparison_rows"):
        issues.append("unexpected comparison_rows")

    names = [a.get("name") or a.get("aircraft_id") for a in (payload.get("aircraft") or []) if isinstance(a, dict)]
    if sc.get("expect_names"):
        for n in sc["expect_names"]:
            if n not in names:
                issues.append(f"missing canonical name {n}")
    if sc.get("min_aircraft") and len(names) < sc["min_aircraft"]:
        issues.append(f"aircraft count {len(names)} < {sc['min_aircraft']}")
    if sc.get("max_aircraft") and len(names) > sc["max_aircraft"]:
        issues.append(f"aircraft count {len(names)} > {sc['max_aircraft']}")
    if sc.get("expect_unique") and len(names) != len(set(names)):
        issues.append("duplicate aircraft entries")

    if sc.get("strategy_compare"):
        if not payload.get("comparison_type") and not payload.get("strategies"):
            if payload.get("status") != "INSUFFICIENT_DATA" and mode == "explicit_comparison":
                if not names:
                    issues.append("missing strategy comparison payload")

    if sc.get("no_ranked_table") and _MARKDOWN_RE.search(raw):
        issues.append("ranked table leak")

    if sc.get("no_banned_models"):
        blob = json.dumps(payload).lower()
        for b in sc["no_banned_models"]:
            if b.lower() in blob:
                issues.append(f"banned model leak {b}")

    if sc.get("needs_structured") and env is None:
        issues.append("needs structured envelope")

    if sc.get("single_mode_only"):
        if "comparison_rows" in json.dumps(payload) and "conflicts" in json.dumps(payload):
            issues.append("mixed comparison and strategic in one payload")

    return {
        "pass": not issues,
        "issues": issues,
        "mode": mode,
        "component": component,
        "aircraft_count": len(names),
        "names": names,
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    ok = 0
    for sc in SCENARIOS:
        r = run_consultant_orchestration(sc["query"])
        env = r.renderer_envelope or _parse_envelope(r.answer)
        ev = _evaluate(sc, env, r.answer)
        if _parse_envelope(r.answer) and '"mode"' in (r.answer or ""):
            ev["pass"] = False
            ev.setdefault("issues", []).append("raw_json_leaked_to_user")
        row = {"id": sc["id"], **ev, "preview": (r.answer or "")[:200]}
        if ev["pass"]:
            ok += 1
        results.append(row)

    out = {"summary": {"pass": f"{ok}/{len(SCENARIOS)}"}, "results": results}
    path = _ROOT / "evals" / "renderer_stress_r15_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

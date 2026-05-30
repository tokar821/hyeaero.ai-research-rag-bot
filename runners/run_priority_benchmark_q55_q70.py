#!/usr/bin/env python3
"""Priority benchmark Q55–Q70 — extended validation batch 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runners.run_priority_benchmark_q19_q54 import (  # noqa: E402
    SCENARIOS as _BASE,
    _EXTRA_CHECKS,
    _evaluate_extended,
)
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402

_EXTRA_CHECKS.update({
    "hong_kong": lambda blob, text: "hong kong" in blob or "hong" in blob,
    "g500": lambda blob, text: "g500" in blob or "gulfstream g500" in blob,
    "three_way_compare": lambda blob, text: (
        ("challenger" in blob or "650" in blob)
        and "praetor" in blob
        and ("longitude" in blob or "citation" in blob)
    ),
    "hierarchy_dubai": lambda blob, text: any(
        x in blob for x in ("hierarchy", "dubai", "episodic", "phoenix", "distort", "primary")
    ),
    "arctic_caribbean": lambda blob, text: any(
        x in blob for x in ("arctic", "caribbean", "structural", "segment", "break", "domain")
    ),
    "falcon_global_compare": lambda blob, text: "falcon 8x" in blob and "global 7500" in blob,
    "fractional_util": lambda blob, text: any(
        x in blob for x in ("fractional", "ownership", "utilization", "hours", "320")
    ),
    "global_5500": lambda blob, text: "global 5500" in blob or "5500" in blob,
    "aspen_london_conflict": lambda blob, text: any(
        x in blob for x in ("aspen", "london", "conflict", "incompatible", "structural")
    ),
    "praetor_image": lambda blob, text: "praetor" in blob and any(
        x in blob for x in ("image", "exterior", "verification", "unable")
    ),
    "n998": lambda blob, text: "n998" in blob.lower() or "tail" in blob,
    "category_balance": lambda blob, text: any(
        x in blob for x in ("category", "super-mid", "large", "shortlist", "dispatch", "resale")
    ),
    "falcon_7x": lambda blob, text: "falcon 7x" in blob or "7x" in text.lower(),
    "riyadh_distortion": lambda blob, text: any(
        x in blob for x in ("riyadh", "distort", "episodic", "miami", "hierarchy", "rational")
    ),
})

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "Q55",
        "priority": 1,
        "query": (
            "We operate Denver → Chicago weekly, Denver → Cabo frequently, occasional Tokyo board meetings. "
            "Leadership wants a single aircraft that can do everything while staying well below Global 7500 operating economics. "
            "What realistically survives once winter Pacific reserve margins are applied?"
        ),
        "expect_mode": ("recommendation_request", "strategic_fleet_analysis"),
        "checks": ["no_raw_json", "winter_or_survive", "broker_prose"],
    },
    {
        "id": "Q56",
        "priority": 2,
        "query": (
            "Could a Gulfstream G500 reliably perform Seattle → Hong Kong westbound in winter "
            "9 passengers NBAA IFR reserves without dispatch compromises becoming routine?"
        ),
        "expect_mode": ("named_aircraft_capability", "strategic_fleet_analysis"),
        "checks": ["no_raw_json", "capability_verdict", "g500", "hong_kong"],
    },
    {
        "id": "Q57",
        "priority": 3,
        "query": (
            "Compare Bombardier Challenger 650, Embraer Praetor 600, Citation Longitude for "
            "executive comfort on 5–6 hour legs, maintenance maturity, pilot workload, "
            "operating discipline, coast-to-coast reliability."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["no_raw_json", "comparison_table", "three_way_compare"],
    },
    {
        "id": "Q58",
        "priority": 4,
        "query": (
            "Most annual utilization is Phoenix, Dallas, Nashville. Executives occasionally continue onward to "
            "Dubai, Singapore, Zurich. Historically procurement became overly centered around Dubai nonstop capability. "
            "How should the operational hierarchy actually be represented?"
        ),
        "expect_mode": ("strategic_fleet_analysis", "network_structure"),
        "checks": ["no_raw_json", "hierarchy_dubai", "network_hierarchy", "episodic_not_driver"],
    },
    {
        "id": "Q59",
        "priority": 5,
        "query": (
            "What structurally breaks first if a company tries using one aircraft for Arctic mining support, "
            "Caribbean executive travel, New York investor missions, San Francisco → Tokyo continuation flights "
            "while insisting on one standardized cabin product?"
        ),
        "expect_mode": ("strategic_fleet_analysis",),
        "checks": ["no_raw_json", "arctic_caribbean", "structural_breaks"],
    },
    {
        "id": "Q60",
        "priority": 6,
        "query": (
            "Create a side-by-side operational comparison between Falcon 8X and Global 7500 focused on "
            "westbound Europe performance, dispatch consistency, airport flexibility, operating cost exposure, "
            "crew fatigue considerations."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["no_raw_json", "comparison_table", "falcon_global_compare"],
    },
    {
        "id": "Q61",
        "priority": 7,
        "query": (
            "We currently charter about 320 hours annually: mostly domestic U.S., quarterly Europe, rare Asia trips. "
            "At what utilization level does fractional ownership stop making sense operationally?"
        ),
        "expect_mode": ("ownership_economics", "strategic_fleet_analysis"),
        "checks": ["no_raw_json", "fractional_util", "ownership_or_structure"],
    },
    {
        "id": "Q62",
        "priority": 8,
        "query": (
            "Could a Global 5500 realistically support Los Angeles → Tokyo westbound winter conditions "
            "10 passengers full executive baggage NBAA reserves without frequent payload penalties?"
        ),
        "expect_mode": ("named_aircraft_capability",),
        "checks": ["no_raw_json", "capability_verdict", "global_5500"],
    },
    {
        "id": "Q63",
        "priority": 9,
        "query": (
            "Compare a single ULR flagship aircraft vs two super-midsize aircraft with supplemental charter for "
            "dispatch resilience, maintenance downtime exposure, empty-leg economics, operational flexibility, scheduling conflicts."
        ),
        "expect_mode": ("explicit_comparison", "strategic_fleet_analysis", "strategic_comparison"),
        "checks": ["no_raw_json", "comparison_or_strategic"],
    },
    {
        "id": "Q64",
        "priority": 10,
        "query": (
            "I want Aspen winter access, nonstop New York → London, lower operating cost than a Gulfstream G650ER, "
            "one-aircraft simplicity. Which operational constraints begin conflicting first?"
        ),
        "expect_mode": ("strategic_fleet_analysis",),
        "checks": ["no_raw_json", "aspen_london_conflict", "conflict_or_structural"],
    },
    {
        "id": "Q65",
        "priority": 11,
        "query": (
            "Show verified exterior-only images of the Praetor 600. Reject generic interiors, unrelated Praetor variants, "
            "stock marketing cabin shots."
        ),
        "checks": ["no_raw_json", "image_or_fail_closed", "praetor_image"],
    },
    {
        "id": "Q66",
        "priority": 12,
        "query": (
            "Find verified images for tail number N998DX. If no exact-aircraft verification exists explicitly state that "
            "then show only the closest confirmed aircraft model."
        ),
        "checks": ["no_raw_json", "tail_or_verification", "n998"],
    },
    {
        "id": "Q67",
        "priority": 13,
        "query": (
            "Which aircraft category gives the best balance of dispatch reliability, boardroom-quality cabin, "
            "disciplined operating economics, strong resale liquidity for a public company flying mostly 3–5 hour executive missions?"
        ),
        "expect_mode": ("recommendation_request", "strategic_fleet_analysis"),
        "checks": ["no_raw_json", "category_balance", "broker_prose"],
    },
    {
        "id": "Q68",
        "priority": 14,
        "query": (
            "Could a Falcon 7X credibly fly San Francisco → Paris westbound winter conditions 8 passengers "
            "NBAA IFR reserves without becoming range-fragile?"
        ),
        "expect_mode": ("named_aircraft_capability",),
        "checks": ["no_raw_json", "capability_verdict", "falcon_7x"],
    },
    {
        "id": "Q69",
        "priority": 15,
        "query": (
            "Compare Gulfstream G500, Global 6500, Falcon 8X for operator friendliness, maintenance ecosystem, "
            "winter transatlantic reliability, passenger productivity, long-term ownership practicality."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["no_raw_json", "comparison_table", "g500"],
    },
    {
        "id": "Q70",
        "priority": 16,
        "query": (
            "We operate Miami Caribbean utilization, Houston energy routes, Orange County executive travel, "
            "occasional Riyadh continuation flights. Leadership keeps insisting Riyadh should drive procurement "
            "because it's strategically important. Is that operationally rational, or is the network hierarchy being distorted?"
        ),
        "expect_mode": ("strategic_fleet_analysis", "network_structure"),
        "checks": ["no_raw_json", "riyadh_distortion", "distortion_or_episodic"],
    },
]


def main() -> None:
    results: List[Dict[str, Any]] = []
    passed = graded = 0
    by_priority: Dict[int, List[bool]] = {}

    for sc in SCENARIOS:
        r = run_consultant_orchestration(sc["query"], conversation_state=None)
        env = r.renderer_envelope
        du = r.data_used_patch or {}
        ok, issues = _evaluate_extended(sc, r.answer or "", du, env)
        graded += 1
        p = int(sc["priority"])
        by_priority.setdefault(p, []).append(ok)
        if ok:
            passed += 1
        results.append({
            "id": sc["id"],
            "priority": p,
            "pass": ok,
            "issues": issues,
            "mode": env.get("mode") if env else None,
            "preview": (r.answer or "")[:280],
        })

    pri_summary = {p: f"{sum(v)}/{len(v)}" for p, v in sorted(by_priority.items())}
    out = {"summary": {"pass": f"{passed}/{graded}", "graded": graded, "by_priority": pri_summary}, "results": results}
    out_path = _ROOT / "evals" / "priority_benchmark_q55_q70_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    failed = [r for r in results if not r.get("pass")]
    if failed:
        print("\n--- FAILURES ---", file=sys.stderr)
        for r in failed:
            print(
                f"{r['id']} P{r['priority']}: {r['issues']} | {r.get('mode')} | {r.get('preview','')[:120]}",
                file=sys.stderr,
            )
    return 0 if passed == graded else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""vNext Routing Stress T01-T15 (unused v2 prompts)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.orchestration.orchestration_router_v2 import route_orchestration_v2
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

SCENARIOS: List[Dict[str, str]] = [
    {"id": "T01", "expected": "strategic_fleet_analysis", "query": (
        "We operate mostly regional US routes but leadership wants something significantly cheaper than a Global 7500 "
        "that can still do Los Angeles → Tokyo westbound in winter with NBAA reserves for 8 passengers. "
        "What aircraft actually survives filtering?"
    )},
    {"id": "T02", "expected": "strategic_fleet_analysis", "query": (
        "Is there any aircraft that can do SFO → Hong Kong nonstop year-round while still keeping operating costs "
        "below Global 7500 levels, or is this structurally impossible?"
    )},
    {"id": "T03", "expected": "strategic_fleet_analysis", "query": (
        "We want to replace a Global 6000 but must keep transatlantic westbound capability in winter for 10 passengers. "
        "What options remain after excluding marginal-range jets?"
    )},
    {"id": "T04", "expected": "network_structure", "query": (
        "We fly Boston, Chicago, Dallas, and Atlanta weekly, but also do occasional Dubai and Singapore trips. "
        "How should the true operational hierarchy be structured?"
    )},
    {"id": "T05", "expected": "network_structure", "query": (
        'Our leadership insists London and Tokyo are "primary hubs," but 80% of flights are domestic US. '
        "What is the correct network priority model?"
    )},
    {"id": "T06", "expected": "network_structure", "query": (
        "We operate Miami, Houston, Denver, and Aspen heavily, with rare Europe trips. "
        "Should Europe still be treated as the dominant planning axis?"
    )},
    {"id": "T07", "expected": "explicit_comparison", "query": (
        "Compare: Bombardier Challenger 650, Embraer Praetor 600, Gulfstream G280 "
        "for coast-to-coast + occasional Europe missions"
    )},
    {"id": "T08", "expected": "explicit_comparison", "query": (
        "What are the tradeoffs between: a single ultra-long-range jet vs "
        "a mixed fleet of super-midsize + charter support?"
    )},
    {"id": "T09", "expected": "explicit_comparison", "query": (
        "Gulfstream G500 vs G600 vs Falcon 8X — which is most efficient for mixed US + Europe utilization?"
    )},
    {"id": "T10", "expected": "named_aircraft_capability", "query": (
        "Can a Falcon 8X realistically do Los Angeles → London westbound in winter with 9 passengers and NBAA reserves?"
    )},
    {"id": "T11", "expected": "named_aircraft_capability", "query": (
        "Is the Embraer Praetor 600 capable of New York → Paris nonstop in winter with full payload?"
    )},
    {"id": "T12", "expected": "named_aircraft_capability", "query": (
        "Can a Bombardier Global 5500 handle SFO → Tokyo westbound year-round for 8 passengers reliably?"
    )},
    {"id": "T13", "expected": "strategic_fleet_analysis", "query": (
        "We operate Arctic oil support, Miami Caribbean hops, and London executive travel. "
        "What structurally breaks if we try to use one aircraft?"
    )},
    {"id": "T14", "expected": "strategic_fleet_analysis", "query": (
        "We want one aircraft to cover Aspen winter, Dallas weekly trips, and occasional Singapore. "
        "Is that structurally feasible or not?"
    )},
    {"id": "T15", "expected": "recommendation_request", "query": (
        "We currently charter 300 hours/year. Leadership wants: lower cost than Global 7500, "
        "nonstop Tokyo capability, strong US domestic efficiency. Should we buy, fractional, or stay charter?"
    )},
]


def main() -> None:
    results: List[Dict[str, Any]] = []
    router_ok = 0
    e2e_ok = 0
    for sc in SCENARIOS:
        route = route_orchestration_v2(sc["query"])
        got = route.query_type.value
        exp = sc["expected"]
        row: Dict[str, Any] = {
            "id": sc["id"],
            "expected": exp,
            "query_type": got,
            "pass_router": got == exp,
            "routing_debug": dict(route.routing_debug),
            "models": list(route.preserve_comparison_models or route.named_aircraft_models),
        }
        if got == exp:
            router_ok += 1
        try:
            e2e = run_consultant_orchestration(sc["query"])
            du = e2e.data_used_patch or {}
            e2e_mode = du.get("orchestration_v2_query_type", "")
            row["e2e_mode"] = e2e_mode
            row["pass_e2e"] = e2e_mode == exp
            row["stabilizer_modified_route"] = (du.get("orchestration_stabilization") or {}).get(
                "stabilizer_modified_route", False
            )
            row["router_vs_e2e_match"] = got == e2e_mode
            if e2e_mode == exp:
                e2e_ok += 1
        except Exception as exc:
            row["e2e_error"] = str(exc)[:180]
        results.append(row)

    summary = {
        "router_pass": f"{router_ok}/15",
        "e2e_pass": f"{e2e_ok}/15",
        "stabilizer_overrides": sum(1 for r in results if r.get("stabilizer_modified_route")),
        "router_e2e_mismatch": sum(1 for r in results if r.get("router_vs_e2e_match") is False),
    }
    out = {"summary": summary, "results": results}
    path = _ROOT / "evals" / "vnext_routing_stress_t15_v2_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

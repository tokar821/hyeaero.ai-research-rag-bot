#!/usr/bin/env python3
"""Smoke-test Orchestration V2 against user-provided scenario prompts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from services.orchestration.orchestration_router_v2 import route_orchestration_v2
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

SCENARIOS = [
    (
        "S01_atlanta_ulr_strategy",
        "We’re headquartered in Atlanta and most annual utilization is still Atlanta ↔ New York ↔ Chicago, "
        "but ownership keeps demanding nonstop winter capability to Johannesburg and Sydney "
        "“without moving up to airline-scale costs.” What aircraft strategy is actually coherent?",
    ),
    (
        "S02_multi_domain_flagship_conflict",
        "Our company operates: mining strips in Northern Quebec, executive traffic into Zurich and London, "
        "seasonal yacht traffic from Miami to the Caribbean, occasional Tokyo continuation trips from San Francisco. "
        "Leadership insists on one flagship aircraft. What structurally conflicts here?",
    ),
    (
        "S03_seattle_paris_westbound_filter",
        "I need an aircraft for 11 passengers that can reliably fly westbound Seattle → Paris year-round "
        "with NBAA IFR reserves, but I specifically want to avoid aircraft that become payload-restricted in winter. "
        "Which models remain realistically viable after filtering?",
    ),
    (
        "S04_compare_g280_650_praetor",
        "Compare: Gulfstream G280, Challenger 650, Praetor 600, for: coast-to-coast executive travel, "
        "occasional Western Europe missions, strong dispatch reliability, "
        "lower operating costs than ultra-long-range jets. Which is the best-balanced operational platform?",
    ),
    (
        "S05_dallas_denver_london_hierarchy",
        "We mostly fly: Dallas ↔ Houston ↔ Denver, occasional Aspen winter access, quarterly London investor trips. "
        "The system previously over-weighted London and kept recommending Global 7500-class aircraft. "
        "What should the real mission hierarchy be?",
    ),
    (
        "S06_mixed_fleet_strategy",
        "Would a mixed fleet strategy make more operational sense than a single aircraft if we operate: "
        "Caribbean regional utilization from Miami, Permian Basin energy operations, executive travel into Frankfurt, "
        "occasional Singapore continuation flights? Focus on dispatch reliability and utilization coherence, not prestige.",
    ),
    (
        "S07_falcon8x_vs_g7500",
        "I’m considering a Falcon 8X instead of a Global 7500 because I care more about operating efficiency "
        "than maximum range. For westbound Los Angeles → London missions with 8 passengers, "
        "what tradeoffs actually matter operationally?",
    ),
    (
        "S08_calgary_arctic_network_hierarchy",
        "We operate between: Calgary energy operations, Arctic gravel strips near Yellowknife, "
        "New York executive headquarters, seasonal Paris and Geneva travel. "
        "Most flight hours are still domestic North America. "
        "How should the network hierarchy be represented without Europe incorrectly becoming the dominant mission?",
    ),
    (
        "S09_ulr_vs_segmented_fleet",
        "Compare the operational realism of: one ultra-long-range flagship aircraft vs "
        "a segmented fleet of super-midsize + regional aircraft for a company flying: "
        "Los Angeles ↔ Tokyo, Miami Caribbean routes, Houston energy operations, short domestic executive corridors. "
        "Which approach produces fewer dispatch failures?",
    ),
    (
        "S10_sf_tokyo_economics_credible",
        "I want lower operating costs than a Global 7500, but I still need reliable nonstop San Francisco → Tokyo "
        "capability westbound in winter for 9 passengers. "
        "Which aircraft are actually credible once marginal-range options are removed?",
    ),
]


def _ascii(s: str) -> str:
    return (s or "").encode("ascii", "ignore").decode("ascii", "ignore")


def main() -> int:
    results = []
    for sid, query in SCENARIOS:
        route = route_orchestration_v2(query)
        orch = run_consultant_orchestration(
            query,
            conversation_state={"history": []},
            data_used={"consultant_response_mode": "mission_advisory"},
            query_intent="mission_feasibility",
        )
        du = orch.data_used_patch or {}
        answer = _ascii(orch.answer or "")
        recs = [r.model for r in (orch.recommendations or [])]
        results.append(
            {
                "id": sid,
                "query_type": route.query_type.value,
                "renderer": route.renderer.value,
                "allow_ranking": route.allow_recommendation_ranking,
                "allow_fallback": route.allow_tier_fallback,
                "allow_synthesis": route.allow_operational_synthesis,
                "locked_compare": list(route.preserve_comparison_models),
                "named_models": list(route.named_aircraft_models),
                "recs": recs,
                "v2_du": du.get("orchestration_v2_query_type"),
                "kernel_blocked": bool(du.get("kernel_synthesis_blocked")),
                "broker_auth": bool(du.get("broker_narrative_authoritative")),
                "hack_v2": bool(du.get("hack_v2_ranking")),
                "answer_head": answer[:500],
                "has_ops_synthesis": "OPERATIONAL SYNTHESIS" in answer,
                "has_strategic": "STRATEGIC ANALYSIS" in answer,
                "has_named_cap": "Named Aircraft Capability" in answer,
                "has_compare_table": "|" in answer and "Compare" in answer or "Structured Comparison" in answer,
                "has_ranked_table": "Ranked Aircraft List" in answer,
            }
        )
        print(f"--- {sid} ---")
        print(f"  route(router): {route.query_type.value} / {route.renderer.value}")
        if du.get("orchestration_v2_query_type") and du.get("orchestration_v2_renderer"):
            print(
                f"  route(stabilized): {du.get('orchestration_v2_query_type')} / {du.get('orchestration_v2_renderer')}"
            )
        print(f"  recs: {recs[:5]}")
        print(f"  synthesis: {'OPERATIONAL SYNTHESIS' in answer}")
        print()

    out = _ROOT / "evals" / "v2_scenario_smoke_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

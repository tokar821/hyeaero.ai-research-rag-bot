#!/usr/bin/env python3
"""
vNext Stress Suite (20 prompts) — Comparison + Structured Output Stabilization Phase
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

env_path = _ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402


SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "Q01_mixed_feasibility_cost_trap",
        "query": (
            "We operate: Boston \u2192 London weekly (10 pax), Chicago \u2192 Aspen winter weekly, "
            "Occasional Dubai trips, Must keep operating cost below Global 6000. "
            "What aircraft class survives after filtering?"
        ),
    },
    {
        "id": "Q02_capability_westbound_stress",
        "query": "Can a Falcon 7X reliably fly Los Angeles \u2192 Paris westbound in January with 9 passengers and NBAA reserves?",
    },
    {
        "id": "Q03_comparison_multi_class",
        "query": "Compare: Praetor 600, Challenger 650, Gulfstream G280 for coast-to-coast US executive missions",
    },
    {
        "id": "Q04_strategic_decomposition_trap",
        "query": (
            "We operate: Arctic gravel strips (Nunavut), Miami Caribbean hops, New York \u2192 London executive travel. "
            "Is a single aircraft structurally viable?"
        ),
    },
    {
        "id": "Q05_range_payload_contradiction",
        "query": "Can a Bombardier Global 5500 do San Francisco \u2192 Tokyo nonstop in winter with 12 passengers?",
    },
    {
        "id": "Q06_mixed_routing_hierarchy",
        "query": (
            "We fly: Houston \u2192 New York weekly, Houston \u2192 Geneva monthly, Occasional Singapore continuation. "
            "How should hierarchy be represented?"
        ),
    },
    {
        "id": "Q07_capability_escalation_cost",
        "query": "What aircraft below Global 7500 cost can still reliably do Los Angeles \u2192 Hong Kong nonstop year-round?",
    },
    {
        "id": "Q08_comparison_ulr_vs_supermid",
        "query": "Compare: Gulfstream G500, Gulfstream G600, Falcon 8X for mixed US + Europe operations",
    },
    {
        "id": "Q09_empty_fallback_prevention",
        "query": "We need: Alaska gravel access, Caribbean ops, Paris executive travel. Recommend a single aircraft.",
    },
    {
        "id": "Q10_named_aircraft_trap",
        "query": "Can a Cessna Citation Longitude fly Miami \u2192 London nonstop with 8 passengers in winter?",
    },
    {
        "id": "Q11_ontology_alias_stress",
        "query": "Compare: Challenger 3500 vs Challenger 650 vs Global 6500",
    },
    {
        "id": "Q12_arctic_ulr_conflict",
        "query": (
            "We operate: Yellowknife gravel strips, Dubai executive travel, Los Angeles \u2192 Tokyo nonstop. "
            "What breaks first structurally?"
        ),
    },
    {
        "id": "Q13_cost_constraint_long_range_trap",
        "query": "We want lower operating cost than G650ER but need San Francisco \u2192 Tokyo westbound winter capability.",
    },
    {
        "id": "Q14_comparison_vs_strategy_ambiguity",
        "query": (
            "We operate: Miami \u2192 Caribbean, Miami \u2192 S\u00e3o Paulo, Houston \u2192 London. "
            "Compare single aircraft vs mixed fleet."
        ),
    },
    {
        "id": "Q15_capability_margin_stress",
        "query": "Can a Global 6000 do Aspen \u2192 London nonstop in winter with 10 passengers?",
    },
    {
        "id": "Q16_routing_misclassification_trap",
        "query": (
            "We have: Dallas \u2192 Chicago \u2192 Atlanta weekly, Occasional Riyadh and Singapore. "
            "What is the true operational hierarchy?"
        ),
    },
    {
        "id": "Q17_heavy_payload_short_runway_conflict",
        "query": "Can a Gulfstream G280 operate consistently from Aspen in winter with full load?",
    },
    {
        "id": "Q18_multi_domain_fleet_breakdown",
        "query": (
            "We operate: Offshore West Africa rigs, Miami Caribbean, Frankfurt executive travel, Singapore continuation. "
            "Is one aircraft viable?"
        ),
    },
    {
        "id": "Q19_comparison_schema_stress",
        "query": "Compare: Global 6500 vs Falcon 8X vs Gulfstream G500 for long-haul + medium-haul mixed utilization",
    },
    {
        "id": "Q20_capability_plus_alt_class",
        "query": (
            "Can a Praetor 600 realistically do San Diego \u2192 Tokyo nonstop with NBAA reserves? "
            "If not, what class is required?"
        ),
    },
]


def _a(s: str) -> str:
    return (s or "").encode("ascii", "ignore").decode("ascii", "ignore")


def _has_strict_table(answer: str) -> bool:
    return (
        "Comparison Type:" in answer
        and "| Aircraft | Range Class | Cabin Class | Economics | Operational Fit | Verdict |" in answer
        and "\n|---|---|---|---|---|---|" in answer
    )


def main() -> int:
    results: List[Dict[str, Any]] = []
    for sc in SCENARIOS:
        q = sc["query"]
        orch = run_consultant_orchestration(
            q,
            conversation_state={"history": []},
            data_used={"consultant_response_mode": "mission_advisory"},
            query_intent="mission_feasibility",
        )
        du = orch.data_used_patch or {}
        answer = _a(orch.answer or "")
        recs = [r.model for r in (orch.recommendations or [])]
        results.append(
            {
                "id": sc["id"],
                "v2_type": du.get("orchestration_v2_query_type"),
                "v2_renderer": du.get("orchestration_v2_renderer"),
                "recs": recs,
                "flags": {
                    "has_ranked_list": "Ranked Aircraft List" in answer,
                    "has_ops_synthesis": "OPERATIONAL SYNTHESIS" in answer,
                    "has_named_capability": "Named Aircraft Capability" in answer,
                    "has_strategic_analysis": "STRATEGIC ANALYSIS" in answer,
                    "has_network_structure": "NETWORK STRUCTURE" in answer,
                    "has_strict_comparison_table": _has_strict_table(answer),
                    "has_insufficient_data": "INSUFFICIENT DATA FOR STRUCTURED COMPARISON" in answer,
                    "contains_unverified": "Unverified" in answer or "UNVERIFIED" in answer,
                    "contains_banned_lightjets": any(x in answer for x in ("Citation CJ2", "Citation CJ4", "Learjet 75")),
                },
                "answer_head": answer[:500],
            }
        )

    out = _ROOT / "evals" / "vnext_stress_suite_20_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for r in results:
        f = r["flags"]
        print(f"--- {r['id']} ---")
        print(f"  route: {r['v2_type']} / {r['v2_renderer']}")
        print(f"  recs: {r['recs'][:4]}")
        print(
            "  flags:",
            "table" if f["has_strict_comparison_table"] else "-",
            "cap" if f["has_named_capability"] else "-",
            "strategic" if f["has_strategic_analysis"] else "-",
            "network" if f["has_network_structure"] else "-",
            "ranked" if f["has_ranked_list"] else "-",
            "opsSynth" if f["has_ops_synthesis"] else "-",
            "insufficient" if f["has_insufficient_data"] else "-",
            "UNVERIFIED" if f["contains_unverified"] else "-",
            "BANNED_LJ" if f["contains_banned_lightjets"] else "-",
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


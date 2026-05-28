#!/usr/bin/env python3
"""
vNext Deep Stress (15 UNUSED questions)
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
        "id": "D01_hierarchy_confusion_trap",
        "query": (
            "We operate: NYC, Chicago, Dallas weekly utilization, occasional Dubai and Singapore investor trips. "
            "Leadership thinks Dubai is the “global hub”. What should the real operational hierarchy be?"
        ),
    },
    {
        "id": "D02_impossible_single_jet",
        "query": (
            "We need one aircraft for: Arctic gravel strips in Nunavut, Miami–Caribbean shuttle, "
            "New York–London winter flights. Is a single platform structurally viable?"
        ),
    },
    {
        "id": "D03_below_ulr_cost_constraint_trap",
        "query": (
            "We want: lower operating cost than Global 7500, nonstop Los Angeles → Tokyo (winter, 8 pax, NBAA reserves). "
            "What aircraft classes survive filtering?"
        ),
    },
    {
        "id": "D04_explicit_comparison_clean_schema",
        "query": (
            "Compare: Challenger 650, Praetor 600, Gulfstream G280 for transcontinental US + occasional Europe"
        ),
    },
    {
        "id": "D05_named_capability_only",
        "query": (
            "Can a Falcon 8X reliably do: Aspen → London in winter, full passenger load (9–10 pax), NBAA reserves?"
        ),
    },
    {
        "id": "D06_network_topology_stress",
        "query": (
            "We operate: Houston oil, Miami Caribbean, Paris executive travel, San Francisco → Tokyo seasonal. "
            "How should the network hierarchy be structured?"
        ),
    },
    {
        "id": "D07_mixed_fleet_vs_flagship",
        "query": (
            "Single Global 6500 vs mixed fleet (super-mid + charter) for: Miami–Caribbean, Houston–NYC, LA–Tokyo. "
            "Focus only on economics and dispatch reliability."
        ),
    },
    {
        "id": "D08_empty_comparison_table_trap",
        "query": (
            "Compare: Global 7500, Gulfstream G500, Falcon 8X for Europe + Asia executive missions"
        ),
    },
    {
        "id": "D09_arctic_offshore_europe_break",
        "query": (
            "We operate: Arctic oil support, West Africa offshore flights, London executive shuttle. "
            "Is one aircraft viable?"
        ),
    },
    {
        "id": "D10_capability_vs_rec",
        "query": (
            "What aircraft can do: San Diego → Hong Kong nonstop with 10 passengers in winter?"
        ),
    },
    {
        "id": "D11_cost_filtered_long_haul",
        "query": (
            "We need: SFO → Paris nonstop, 9 passengers but must reduce operating cost vs Global 7500."
        ),
    },
    {
        "id": "D12_hierarchy_reversal_test",
        "query": (
            "Why is most utilization: Dallas–Chicago–Atlanta but system keeps optimizing for international routes?"
        ),
    },
    {
        "id": "D13_continuation_hub_confusion",
        "query": (
            "We fly: LA → Tokyo, LA → Singapore, Miami → Caribbean. Why is Tokyo being treated as dominant utilization?"
        ),
    },
    {
        "id": "D14_winter_westbound_constraint_trap",
        "query": (
            "Can a Gulfstream G600 reliably do: LA → London westbound in winter with reserves? "
            "If not, what class solves it?"
        ),
    },
    {
        "id": "D15_fleet_segmentation_pressure",
        "query": (
            "We operate: Alaska gravel strips, Geneva executive travel, Miami Caribbean, London–New York shuttle. "
            "Should we use: A) single flagship B) mixed fleet C) hybrid model. Explain structurally."
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
                    "contains_banned_lightjets": any(
                        x in answer for x in ("Citation CJ2", "Citation CJ4", "Learjet 75")
                    ),
                },
                "answer_head": answer[:600],
            }
        )

    out = _ROOT / "evals" / "vnext_deep_stress_15_results.json"
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


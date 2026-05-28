#!/usr/bin/env python3
"""vNext Routing Stress Test — 15 unused questions (routing + optional E2E)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.orchestration.orchestration_router_v2 import (  # noqa: E402
    OrchestrationQueryTypeV2,
    route_orchestration_v2,
)
from services.orchestration.pipeline_orchestrator import (  # noqa: E402
    run_consultant_orchestration,
)

SCENARIOS: List[Dict[str, str]] = [
    {"id": "T01", "category": "cost_ulr", "query": (
        "We need nonstop Tokyo capability from Los Angeles year-round for 9 executives, "
        "but leadership wants operating costs significantly below a Global 7500. "
        "What aircraft realistically remain viable?"
    )},
    {"id": "T02", "category": "cost_ulr", "query": (
        "We fly New York to Dubai and Singapore occasionally, but 80% of hours are domestic. "
        "We want to avoid Global 7500-level costs. What aircraft strategy makes sense?"
    )},
    {"id": "T03", "category": "cost_ulr", "query": (
        "Can we replace a Global 6000 with something cheaper while still guaranteeing "
        "westbound London flights from Chicago in winter with NBAA reserves?"
    )},
    {"id": "T04", "category": "capability", "query": (
        "Can a Falcon 8X safely handle Aspen → London nonstop in winter with 10 passengers "
        "and full fuel reserves?"
    )},
    {"id": "T05", "category": "capability", "query": (
        "Is the Embraer Praetor 600 capable of Los Angeles → Tokyo nonstop year-round under NBAA rules?"
    )},
    {"id": "T06", "category": "capability", "query": (
        "Would a Challenger 3500 realistically complete Miami → Paris nonstop with 8 passengers "
        "in winter operations?"
    )},
    {"id": "T07", "category": "comparison", "query": (
        "Compare Gulfstream G650ER vs Global 7500 vs Falcon 8X for transpacific executive travel."
    )},
    {"id": "T08", "category": "comparison", "query": (
        "Which is better for our company: a super-midsize fleet strategy or a single "
        "ultra-long-range flagship aircraft?"
    )},
    {"id": "T09", "category": "comparison", "query": (
        "G500 vs G600 vs Global 5500 — which actually performs better for mixed domestic "
        "and international usage?"
    )},
    {"id": "T10", "category": "network", "query": (
        "Our executives mainly fly Dallas, Houston, Chicago, Atlanta, but occasionally continue "
        "to Dubai and Singapore. How should this hierarchy actually be structured?"
    )},
    {"id": "T11", "category": "network", "query": (
        "We used to treat London as the operational hub, but most flights are actually domestic. "
        "Is that interpretation wrong?"
    )},
    {"id": "T12", "category": "network", "query": (
        "How should we prioritize Aspen winter access vs New York–London executive travel "
        "in fleet planning?"
    )},
    {"id": "T13", "category": "strategic", "query": (
        "We operate Miami Caribbean routes, Houston energy flights, and Los Angeles–Tokyo trips. "
        "Can one aircraft realistically cover everything?"
    )},
    {"id": "T14", "category": "strategic", "query": (
        "We want one aircraft for Arctic operations, Europe executive travel, and Asia-Pacific missions. "
        "What breaks first?"
    )},
    {"id": "T15", "category": "edge", "query": (
        "What is the best aircraft for my company if we want low cost, maximum range, "
        "short runway access, and 12 passengers?"
    )},
]

_BANNED_CAPABILITY = (
    "alternatives", "better option", "consider", "instead", "recommend",
    "class band", "category", "shortlist", "ranked",
)


def _capability_leak(text: str) -> List[str]:
    low = (text or "").lower()
    return [t for t in _BANNED_CAPABILITY if t in low]


def main() -> None:
    results: List[Dict[str, Any]] = []
    for sc in SCENARIOS:
        q = sc["query"]
        route = route_orchestration_v2(q)
        row: Dict[str, Any] = {
            "id": sc["id"],
            "category": sc["category"],
            "query_type": route.query_type.value,
            "renderer": route.renderer.value,
            "signals": list(route.signals),
            "routing_debug": dict(route.routing_debug),
            "router_v2_final": dict(route.routing_debug),
            "models": list(route.preserve_comparison_models or route.named_aircraft_models),
            "allow_ranking": route.allow_recommendation_ranking,
        }
        try:
            e2e = run_consultant_orchestration(q)
            text = (e2e.answer or "")[:1200]
            du = e2e.data_used_patch or {}
            stab = du.get("orchestration_stabilization") or {}
            row["response_preview"] = text[:500]
            row["capability_leaks"] = _capability_leak(text)
            row["e2e_mode"] = du.get("orchestration_v2_query_type")
            row["stabilizer_modified_route"] = stab.get("stabilizer_modified_route", False)
            row["router_vs_e2e_match"] = row["query_type"] == row["e2e_mode"]
            row["e2e_has_ranked_list"] = bool(
                "## Ranked" in text or "| Rank |" in text
            )
        except Exception as exc:
            row["e2e_error"] = str(exc)[:200]
        results.append(row)

    out = _ROOT / "evals" / "vnext_routing_stress_t15_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

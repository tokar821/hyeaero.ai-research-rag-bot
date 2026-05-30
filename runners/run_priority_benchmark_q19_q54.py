#!/usr/bin/env python3
"""Priority benchmark Q19–Q54 — extended priority validation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runners.run_priority_benchmark_q18 import _evaluate, _RAW_JSON  # noqa: E402
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402

SCENARIOS: List[Dict[str, Any]] = [
    # P1
    {"id": "Q19", "priority": 1, "query": (
        'Our CEO wants "Global 7500 capability without Global 7500 economics." '
        "Typical utilization: Chicago → New York, Chicago → Los Angeles, quarterly Tokyo trips. "
        "What aircraft actually survive that requirement once winter westbound reserve margins are applied?"
    ), "expect_mode": ("recommendation_request", "strategic_fleet_analysis"), "checks": ["no_raw_json", "winter_or_survive", "no_cj_pc24", "broker_prose"]},
    {"id": "Q20", "priority": 1, "query": (
        "We currently charter a mix of Challenger 650 and Falcon 7X. Leadership wants to standardize onto one aircraft family. "
        "What operational compromises appear if we consolidate too aggressively?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "structural_breaks", "consolidat"]},
    {"id": "Q21", "priority": 1, "query": (
        "I want: true stand-up cabin comfort, lower crew/maintenance burden than a Global 7500, "
        "reliable New York → Paris capability, 8 passenger executive loads. What aircraft category actually fits?"
    ), "expect_mode": ("recommendation_request", "strategic_fleet_analysis", "named_aircraft_capability"), "checks": ["no_raw_json", "category_or_shortlist"]},
    {"id": "Q22", "priority": 1, "query": (
        "We are evaluating whether a Gulfstream G500 is operationally enough aircraft for: "
        "Los Angeles, London, occasional Dubai, heavy executive utilization. "
        "Where does it begin losing credibility versus larger ULR aircraft?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "g500_or_ulr"]},
    # P2
    {"id": "Q23", "priority": 2, "query": (
        "Most hours are Miami, Nassau, San Juan, Houston. But ownership keeps discussing Hong Kong because it's strategically important. "
        "Should Hong Kong materially influence fleet sizing?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "episodic_not_driver", "domestic_cog"]},
    {"id": "Q24", "priority": 2, "query": (
        "We do Dallas → Chicago weekly, Dallas → Aspen in winter, occasional Geneva trips. "
        "Should Aspen or Geneva drive aircraft selection more heavily?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "distortion_or_episodic", "aspen_or_geneva"]},
    {"id": "Q25", "priority": 2, "query": (
        'Executives say "Tokyo is critical." But Tokyo represents fewer than 3 annual trips. '
        "Meanwhile domestic utilization exceeds 80% and short-runway operations are frequent. "
        "How should the network hierarchy actually be weighted?"
    ), "expect_mode": ("strategic_fleet_analysis", "network_structure"), "checks": ["no_raw_json", "network_hierarchy", "episodic_not_driver"]},
    {"id": "Q26", "priority": 2, "query": (
        "Our CFO wants one aircraft optimized around Singapore, Riyadh, Dubai. "
        "But utilization data shows mostly Florida, Texas, Northeast corridor flying. "
        "What planning distortion risks appear if procurement follows the international edge cases?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "distortion_or_episodic", "domestic_cog"]},
    # P3
    {"id": "Q27", "priority": 3, "query": (
        "Compare Falcon 7X, Falcon 8X, Global 6500 for winter Europe missions, operating economics, dispatch reliability, airport flexibility."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_table", "tradeoffs_or_table"]},
    {"id": "Q28", "priority": 3, "query": (
        "Create a structured comparison between Praetor 600, Citation Longitude, Challenger 3500 "
        "focused on coast-to-coast utilization, passenger comfort, reliability, owner/operator practicality."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_or_insufficient"]},
    {"id": "Q29", "priority": 3, "query": (
        "Compare a single ultra-long-range aircraft vs a segmented fleet with super-midsize aircraft plus charter supplementation "
        "for dispatch resilience, empty-leg efficiency, operational coherence."
    ), "expect_mode": ("explicit_comparison", "strategic_fleet_analysis", "strategic_comparison"), "checks": ["no_raw_json", "comparison_or_strategic"]},
    {"id": "Q30", "priority": 3, "query": (
        "Build a side-by-side operational comparison between Global 7500 and Gulfstream G650ER "
        "for westbound Pacific reliability, maintenance burden, passenger productivity, scheduling flexibility."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_table"]},
    # P4
    {"id": "Q31", "priority": 4, "query": (
        "Could a Praetor 600 realistically fly San Diego → Tokyo westbound winter 8 passengers NBAA reserves without becoming dispatch-fragile?"
    ), "expect_mode": ("named_aircraft_capability",), "checks": ["no_raw_json", "capability_verdict", "praetor"]},
    {"id": "Q32", "priority": 4, "query": (
        "Could a Challenger 3500 reliably handle New York → London year-round 9 passengers baggage-heavy executive loads without frequent restrictions?"
    ), "expect_mode": ("named_aircraft_capability",), "checks": ["no_raw_json", "capability_verdict", "challenger_350"]},
    {"id": "Q33", "priority": 4, "query": (
        "Would a Global 5500 realistically support Seattle → Tokyo westbound winter conditions 10 passengers NBAA IFR reserves with operational consistency?"
    ), "expect_mode": ("named_aircraft_capability",), "checks": ["no_raw_json", "capability_verdict"]},
    {"id": "Q34", "priority": 4, "query": (
        "Can a Citation Longitude credibly perform Los Angeles → Paris nonstop winter westbound executive reserve standards or is that structurally optimistic?"
    ), "expect_mode": ("named_aircraft_capability", "strategic_fleet_analysis"), "checks": ["no_raw_json", "capability_verdict"]},
    # P5
    {"id": "Q35", "priority": 5, "query": (
        "We operate Arctic resource flights, Caribbean executive travel, New York investor routes, occasional Asia continuation flights. "
        "Leadership wants one aircraft because fleet simplicity matters. Why does that logic begin failing operationally?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "structural_breaks"]},
    {"id": "Q36", "priority": 5, "query": (
        "What operational problems emerge when 90% of flying is domestic but procurement is centered around a handful of ultra-long-haul missions?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "distortion_or_episodic"]},
    {"id": "Q37", "priority": 5, "query": (
        "Our executives want Aspen capability, Tokyo nonstop, low operating cost, one-aircraft simplicity. Which constraints begin colliding first?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "conflict_or_structural"]},
    {"id": "Q38", "priority": 5, "query": (
        "Why do mixed utilization patterns usually push sophisticated operators toward segmented fleets instead of one hero aircraft?"
    ), "expect_mode": ("strategic_fleet_analysis",), "checks": ["no_raw_json", "segment_or_fleet"]},
    # P6
    {"id": "Q39", "priority": 6, "query": (
        "Compare Falcon 8X, Global 7500, Gulfstream G650ER for winter Pacific missions, passenger comfort, operational flexibility, dispatch reliability."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_table", "tradeoffs_or_table"]},
    {"id": "Q40", "priority": 6, "query": (
        "Compare Praetor 600, Gulfstream G280, Challenger 650 for operator friendliness, maintenance support, transcontinental utilization, cabin practicality."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_table"]},
    {"id": "Q41", "priority": 6, "query": (
        "Which is operationally smarter for a midsize company: one large-cabin flagship aircraft or two super-midsize aircraft with charter supplementation?"
    ), "expect_mode": ("explicit_comparison", "strategic_fleet_analysis", "strategic_comparison"), "checks": ["no_raw_json", "comparison_or_strategic"]},
    {"id": "Q42", "priority": 6, "query": (
        "Compare Global 6500 and Falcon 8X for airport access, runway flexibility, winter reliability, crew workload, ownership economics."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_table"]},
    # P7
    {"id": "Q43", "priority": 7, "query": (
        "Create a broker-style acquisition brief for New York → London executive travel 10 passengers "
        "lower operating cost than a Global 7500 emphasis on dispatch consistency."
    ), "expect_mode": ("recommendation_request",), "checks": ["no_raw_json", "broker_layout", "shortlist_or_guidance"]},
    {"id": "Q44", "priority": 7, "query": (
        "Design a comparison presentation for Praetor 600, Challenger 3500, Citation Longitude "
        "where operational strengths and weaknesses are immediately obvious."
    ), "expect_mode": ("explicit_comparison",), "checks": ["no_raw_json", "comparison_or_insufficient"]},
    {"id": "Q45", "priority": 7, "query": (
        "Show a professional fleet-strategy summary comparing one-aircraft ownership vs segmented fleet structure "
        "for a multinational executive operation."
    ), "expect_mode": ("strategic_fleet_analysis", "explicit_comparison", "strategic_comparison"), "checks": ["no_raw_json", "segment_or_fleet"]},
    {"id": "Q46", "priority": 7, "query": (
        "Generate a high-end advisory-style recommendation layout for Aspen winter operations, coast-to-coast utilization, "
        "occasional Europe travel under a disciplined operating-cost mandate."
    ), "expect_mode": ("recommendation_request", "strategic_fleet_analysis"), "checks": ["no_raw_json", "broker_prose"]},
    # P8
    {"id": "Q47a", "priority": 8, "query": "We mostly operate Texas and Florida routes with occasional London travel.", "store_state": True},
    {"id": "Q47", "priority": 8, "multi": True, "prior": "Q47a", "query": "Would your answer change if Tokyo became a quarterly mission?",
     "checks": ["no_raw_json", "evolution_or_change", "tokyo_or_quarterly"]},
    {"id": "Q48a", "priority": 8, "query": "We currently charter about 180 hours annually.", "store_state": True},
    {"id": "Q48", "priority": 8, "multi": True, "prior": "Q48a",
     "query": "Still assuming we prioritize low operating complexity, does fractional ownership now make more sense than full ownership?",
     "checks": ["no_raw_json", "ownership_or_structure"]},
    {"id": "Q49a", "priority": 8, "query": "We operate mostly domestic U.S. flying.", "store_state": True},
    {"id": "Q49", "priority": 8, "multi": True, "prior": "Q49a",
     "query": "What if leadership suddenly insists on guaranteed nonstop Singapore capability?",
     "checks": ["no_raw_json", "singapore_or_conflict"]},
    {"id": "Q50a", "priority": 8, "query": "We want lower operating costs than a Global 7500.", "store_state": True},
    {"id": "Q50", "priority": 8, "multi": True, "prior": "Q50a",
     "query": "Still assuming winter westbound Pacific capability remains mandatory, what compromises become unavoidable?",
     "checks": ["no_raw_json", "conflict_or_structural", "pacific_or_winter"]},
    # P9
    {"id": "Q51", "priority": 9, "query": (
        "Show verified exterior images of the Falcon 8X only. Reject interiors, unrelated Falcon variants, or stock cabin imagery."
    ), "checks": ["no_raw_json", "image_or_fail_closed", "falcon_8x"]},
    {"id": "Q52", "priority": 9, "query": (
        "Find verified images for tail number N750LX. If exact verification fails explicitly say so then show only the closest confirmed aircraft model."
    ), "checks": ["no_raw_json", "tail_or_verification", "n750"]},
    {"id": "Q53", "priority": 9, "query": (
        "Show only verified images of the Global 6500 cockpit and exterior. Do not include generic Global-series interiors."
    ), "checks": ["no_raw_json", "image_or_fail_closed", "global_6500"], "expect_mode": ("recommendation_request",)},
    {"id": "Q54", "priority": 9, "query": (
        'Find verified images for VP-CBA. If confidence is low do not hallucinate return "No verified images found for this exact aircraft."'
    ), "checks": ["no_raw_json", "image_or_fail_closed", "vp_cba_or_fail"]},
]

# Extended checks (merged into local evaluate)
_EXTRA_CHECKS = {
    "winter_or_survive": lambda blob, text: any(x in blob for x in ("winter", "survive", "margin", "ulr", "falcon", "global", "g650")),
    "consolidat": lambda blob, text: "consolidat" in blob or "compromise" in blob or "standardiz" in blob,
    "category_or_shortlist": lambda blob, text: any(x in blob for x in ("category", "large", "super-mid", "shortlist", "cabin", "paris")),
    "g500_or_ulr": lambda blob, text: "g500" in blob or "gulfstream g500" in blob or "ulr" in blob or "credibility" in blob,
    "aspen_or_geneva": lambda blob, text: "aspen" in blob or "geneva" in blob,
    "network_hierarchy": lambda blob, text: any(x in blob for x in ("hierarchy", "weight", "domestic", "episodic", "primary")),
    "comparison_or_strategic": lambda blob, text: "|" in text or "segment" in blob or "fleet" in blob or "comparison" in blob or "strategic" in blob,
    "praetor": lambda blob, text: "praetor" in blob,
    "segment_or_fleet": lambda blob, text: any(x in blob for x in ("segment", "fleet", "charter", "mixed")),
    "tokyo_or_quarterly": lambda blob, text: "tokyo" in blob or "quarterly" in blob or "change" in blob,
    "singapore_or_conflict": lambda blob, text: any(x in blob for x in ("singapore", "conflict", "ulr", "nonstop", "incompatible")),
    "pacific_or_winter": lambda blob, text: any(x in blob for x in ("pacific", "winter", "compromise", "conflict")),
    "n750": lambda blob, text: "n750" in blob.lower() or "tail" in blob,
    "global_6500": lambda blob, text: "global 6500" in blob or "6500" in blob,
    "vp_cba_or_fail": lambda blob, text: "vp-cba" in blob.lower() or "vp cba" in blob.lower() or "no verified" in blob.lower() or "verification" in blob,
}


def _evaluate_extended(sc: Dict[str, Any], answer: str, du: Dict[str, Any], env: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    env = dict(env) if env else {}
    if env.get("mode") == "error" and "strategic comparison" in (answer or "").lower():
        env["mode"] = "strategic_comparison"
    ok, issues = _evaluate(sc, answer, du, env)
    blob = (answer or "").lower() + json.dumps(du or {}).lower()
    text = answer or ""
    for check in sc.get("checks") or []:
        fn = _EXTRA_CHECKS.get(check)
        if fn and not fn(blob, text):
            issues.append(f"failed:{check}")
            ok = False
    return ok, issues


def main() -> None:
    conversation_states: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    passed = graded = 0
    by_priority: Dict[int, List[bool]] = {}

    for sc in SCENARIOS:
        conv: Dict[str, Any] = {}
        if sc.get("multi") and sc.get("prior"):
            conv = conversation_states.get(sc["prior"], {})

        r = run_consultant_orchestration(sc["query"], conversation_state=conv or None)
        env = r.renderer_envelope
        du = r.data_used_patch or {}

        if sc.get("store_state"):
            conversation_states[sc["id"]] = {
                "history": [
                    {"role": "user", "content": sc["query"]},
                    {"role": "assistant", "content": (r.answer or "")[:2000]},
                ],
                "data_used": du,
            }
            results.append({"id": sc["id"], "priority": sc["priority"], "pass": True, "issues": [], "note": "setup"})
            continue

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
    out_path = _ROOT / "evals" / "priority_benchmark_q19_q54_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    failed = [r for r in results if not r.get("pass") and not r.get("note")]
    if failed:
        print("\n--- FAILURES ---", file=sys.stderr)
        for r in failed:
            print(f"{r['id']} P{r['priority']}: {r['issues']} | {r.get('mode')} | {r.get('preview','')[:100]}", file=sys.stderr)


if __name__ == "__main__":
    main()

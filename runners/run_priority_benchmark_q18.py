#!/usr/bin/env python3
"""Priority benchmark Q1–Q18 — report broker-quality signals per question."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "Q1",
        "priority": 1,
        "query": (
            "Your board wants a lower-cost alternative to a Gulfstream G650, but they refuse to lose: "
            "true transatlantic comfort, executive cabin feel, nonstop New York → London capability for "
            "10 passengers. What aircraft realistically remain credible replacements, and which options "
            "fall too far down-market operationally?"
        ),
        "checks": ["no_cj_pc24", "upper_large_or_ulr", "no_raw_json", "broker_prose"],
    },
    {
        "id": "Q2",
        "priority": 1,
        "query": (
            "We currently operate a Bombardier Global 7500 for a founder-led technology company. "
            "Most flights are San Jose → New York, San Jose → Miami, occasional Tokyo investor trips. "
            "Would downsizing into the upper-large-cabin segment make operational sense, or would it "
            "materially reduce dispatch credibility?"
        ),
        "checks": ["founder_or_dispatch", "no_raw_json"],
    },
    {
        "id": "Q3",
        "priority": 2,
        "query": (
            "Most annual utilization is Dallas, Houston, Atlanta. But leadership keeps fixating on "
            "Riyadh, Singapore, Dubai because those trips are strategically important. "
            "How should the actual procurement center-of-gravity be represented?"
        ),
        "expect_mode": ("strategic_fleet_analysis", "network_structure"),
        "checks": ["domestic_cog", "episodic_not_driver", "no_raw_json"],
    },
    {
        "id": "Q4",
        "priority": 2,
        "query": (
            "We fly Orange County → Chicago weekly, Orange County → Aspen during ski season, "
            "occasional London and Tokyo investor trips. Leadership insists the aircraft should be "
            "optimized around Tokyo because it's our longest route. Is that operationally rational, "
            "or is the network being distorted by edge-case missions?"
        ),
        "expect_mode": ("strategic_fleet_analysis",),
        "checks": ["distortion_or_episodic", "not_tokyo_procurement", "no_raw_json"],
    },
    {
        "id": "Q5",
        "priority": 3,
        "query": (
            "Compare Dassault Falcon 8X, Bombardier Global 6500, Gulfstream G500 for "
            "Los Angeles → London westbound winter missions, 8–10 passengers, operating cost discipline, "
            "dispatch reliability. Focus on operational tradeoffs, not prestige."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["comparison_table", "tradeoffs_or_table", "no_raw_json"],
    },
    {
        "id": "Q6",
        "priority": 3,
        "query": (
            "Create a side-by-side comparison between Cessna Citation Longitude and Embraer Legacy 600 "
            "covering cabin usability, baggage practicality, pilot workload, operating economics, "
            "passenger experience."
        ),
        "checks": ["comparison_or_insufficient", "no_raw_json"],
    },
    {
        "id": "Q7",
        "priority": 4,
        "query": (
            "Could a Bombardier Global 6000 realistically perform San Francisco → Tokyo westbound in winter "
            "11 passengers NBAA IFR reserves without frequent payload penalties or fuel stops? "
            "If not, what aircraft class solves the mission more credibly?"
        ),
        "expect_mode": ("named_aircraft_capability",),
        "checks": ["capability_verdict", "no_shortlist_ranking", "class_guidance_optional"],
    },
    {
        "id": "Q8",
        "priority": 4,
        "query": (
            "Could a Dassault Falcon 8X reliably fly Los Angeles → London westbound in winter "
            "10 passengers NBAA IFR reserves without dispatch compromises becoming routine?"
        ),
        "expect_mode": ("named_aircraft_capability",),
        "checks": ["falcon_8x", "capability_verdict", "no_raw_json"],
    },
    {
        "id": "Q9",
        "priority": 5,
        "query": (
            "We operate Caribbean executive shuttles from Miami, Houston energy operations, "
            "New York investor travel, occasional Singapore continuation flights. Leadership wants one "
            "flagship aircraft for branding simplicity. What structurally breaks first, and what operating "
            "structure actually makes sense?"
        ),
        "expect_mode": ("strategic_fleet_analysis",),
        "checks": ["structural_breaks", "no_generic_only", "no_raw_json"],
    },
    {
        "id": "Q10",
        "priority": 5,
        "query": (
            "Our executives want nonstop Asia capability, Aspen winter access, short domestic repositioning "
            "flexibility, lower operating cost than a Global 7500. Why do these requirements start "
            "conflicting operationally?"
        ),
        "expect_mode": ("strategic_fleet_analysis", "named_aircraft_capability"),
        "checks": ["conflict_or_structural", "no_raw_json"],
    },
    {
        "id": "Q11",
        "priority": 6,
        "query": (
            "Compare Embraer Praetor 600, Bombardier Challenger 650, Gulfstream G280 for "
            "dispatch maturity, maintenance ecosystem, coast-to-coast executive utilization, "
            "operational flexibility, long-term resale liquidity."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["comparison_table", "tradeoffs_or_three_aircraft"],
    },
    {
        "id": "Q12",
        "priority": 6,
        "query": (
            "Compare Gulfstream G500, Dassault Falcon 8X, Bombardier Global 6500 for "
            "winter transatlantic reliability, airport flexibility, maintenance burden, "
            "crew workload, long-term operator suitability."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["comparison_table", "tradeoffs_or_table"],
    },
    {
        "id": "Q13",
        "priority": 7,
        "query": (
            "Show a broker-style acquisition summary for 9-passenger Los Angeles → London missions "
            "emphasis on dispatch reliability under $45M acquisition budget. "
            "Professional aircraft brokerage presentation."
        ),
        "checks": ["broker_layout", "no_raw_json", "shortlist_or_guidance"],
    },
    {
        "id": "Q14",
        "priority": 7,
        "query": (
            "Create a clean comparison layout for Falcon 8X, Global 6500, Gulfstream G500 where the user "
            "can instantly understand which aircraft dominates which mission, where compromises appear, "
            "which aircraft is the safest operational recommendation."
        ),
        "expect_mode": ("explicit_comparison",),
        "checks": ["comparison_table", "no_raw_json"],
    },
    {
        "id": "Q15a",
        "priority": 8,
        "multi": False,
        "query": "We currently charter about 250 hours annually between Dallas, New York, and London.",
        "store_state": True,
    },
    {
        "id": "Q15",
        "priority": 8,
        "multi": True,
        "prior": "Q15a",
        "query": (
            "Still assuming we want lower operating costs than a G650, what ownership structure "
            "makes the most sense now?"
        ),
        "checks": ["continuity_g650", "ownership_or_structure", "no_raw_json"],
    },
    {
        "id": "Q16a",
        "priority": 8,
        "multi": False,
        "query": "We mainly operate Miami Caribbean routes with occasional London investor travel.",
        "store_state": True,
    },
    {
        "id": "Q16",
        "priority": 8,
        "multi": True,
        "prior": "Q16a",
        "query": "Would your recommendation change if Aspen winter operations become frequent?",
        "checks": ["aspen_or_winter", "evolution_or_change", "no_raw_json"],
    },
    {
        "id": "Q17",
        "priority": 9,
        "query": (
            "Show verified exterior images of the Bombardier Challenger 3500 only. "
            "Do not show generic cabins or unrelated Challenger variants."
        ),
        "checks": ["image_or_fail_closed", "challenger_350"],
    },
    {
        "id": "Q18",
        "priority": 9,
        "query": (
            "Find verified images for tail number N628TS. If exact-aircraft verification is unavailable "
            "explicitly say so then show only the closest verified aircraft model reference."
        ),
        "checks": ["tail_or_verification", "fail_closed_ok"],
    },
]

_BANNED_LIGHT = re.compile(
    r"\b(?:citation\s+cj[24]|cj2|cj4|pc-?24|learjet\s*75|caravan)\b",
    re.I,
)
_RAW_JSON = re.compile(r'^\s*\{\s*"mode"')
_GENERIC_STRATEGIC = re.compile(
    r"fleet segmentation requirement likely.*dispatch mismatch risk",
    re.I | re.S,
)


def _blob(answer: str, du: Dict[str, Any], env: Optional[Dict[str, Any]]) -> str:
    parts = [answer or ""]
    if env:
        parts.append(json.dumps(env))
    parts.append(json.dumps(du or {}))
    return "\n".join(parts).lower()


def _evaluate(
    sc: Dict[str, Any],
    answer: str,
    du: Dict[str, Any],
    env: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    text = answer or ""
    blob = _blob(text, du, env)
    mode = (env or {}).get("mode") if env else None

    exp_modes = sc.get("expect_mode")
    if exp_modes:
        if isinstance(exp_modes, str):
            exp_modes = (exp_modes,)
        if mode and mode not in exp_modes:
            issues.append(f"mode={mode} expected {exp_modes}")

    for check in sc.get("checks") or []:
        if check == "no_raw_json":
            if _RAW_JSON.match(text.strip()) or ('"payload"' in text and text.strip().startswith("{")):
                issues.append("raw_json_leaked")
        elif check == "no_cj_pc24":
            if _BANNED_LIGHT.search(text) and "down-market" not in text.lower():
                issues.append("banned_light_jet_in_shortlist")
        elif check == "upper_large_or_ulr":
            if not any(
                x in blob
                for x in ("falcon 8x", "global 6500", "g650", "challenger 650", "praetor", "large")
            ):
                if "shortlist" in blob or "| aircraft |" in text.lower():
                    issues.append("missing_credible_replacement_band")
        elif check == "broker_prose":
            if len(text) < 80:
                issues.append("answer_too_short")
        elif check == "domestic_cog":
            if not any(x in blob for x in ("dallas", "houston", "atlanta", "domestic", "center of gravity")):
                issues.append("missing_domestic_cog")
        elif check == "episodic_not_driver":
            if not any(
                x in blob
                for x in ("episodic", "continuation", "secondary", "not procurement", "too infrequent")
            ):
                issues.append("missing_episodic_discipline")
        elif check == "distortion_or_episodic":
            if not any(x in blob for x in ("distort", "episodic", "edge", "weekly", "dominant", "rational")):
                issues.append("missing_distortion_analysis")
        elif check == "comparison_table":
            if "|" not in text and "insufficient" not in text.lower():
                issues.append("missing_comparison_table")
        elif check == "tradeoffs_or_table":
            if "|" not in text and "tradeoff" not in blob and "operational" not in blob:
                issues.append("missing_tradeoffs")
        elif check == "comparison_or_insufficient":
            if "|" not in text and "insufficient" not in text.lower() and "comparison" not in blob:
                issues.append("no_comparison_output")
        elif check == "capability_verdict":
            if not any(x in blob for x in ("feasible", "marginal", "not realistic", "verdict")):
                issues.append("missing_capability_verdict")
        elif check == "falcon_8x":
            if "falcon 8x" not in blob and "8x" not in text.lower():
                issues.append("missing_falcon_8x_eval")
        elif check == "no_shortlist_ranking":
            if re.search(r"\|\s*1\s*\|.*\|\s*citation", text, re.I):
                issues.append("ranked_shortlist_leak")
        elif check == "structural_breaks":
            if not any(x in blob for x in ("break", "segment", "conflict", "structure", "domain")):
                issues.append("missing_structural_analysis")
        elif check == "no_generic_only":
            if _GENERIC_STRATEGIC.search(text) and "center of gravity" not in blob:
                issues.append("too_generic_strategic")
        elif check == "conflict_or_structural":
            if not any(x in blob for x in ("conflict", "incompatible", "structural", "segment")):
                issues.append("missing_conflict_explanation")
        elif check == "broker_layout":
            if not any(x in blob for x in ("shortlist", "rank", "mission", "acquisition", "dispatch")):
                issues.append("missing_broker_summary")
        elif check == "shortlist_or_guidance":
            if not any(x in blob for x in ("shortlist", "no aircraft", "constraint", "guidance", "catalog")):
                issues.append("missing_recommendation_content")
        elif check == "continuity_g650":
            cc = du.get("context_continuity") or {}
            if not (
                "g650" in blob
                or (isinstance(cc, dict) and "g650" in str(cc).lower())
                or "continuing from prior" in text.lower()
                or du.get("continuity_reference_aircraft")
            ):
                issues.append("missing_g650_continuity")
        elif check == "ownership_or_structure":
            if not any(x in blob for x in ("fractional", "ownership", "charter", "whole", "structure", "acquisition")):
                issues.append("missing_ownership_guidance")
        elif check == "aspen_or_winter":
            if "aspen" not in blob and "winter" not in blob:
                issues.append("missing_aspen_evolution")
        elif check == "evolution_or_change":
            if not any(x in blob for x in ("change", "frequent", "field", "runway", "recommend")):
                issues.append("missing_evolution_response")
        elif check == "image_or_fail_closed":
            if not any(
                x in blob
                for x in ("image", "exterior", "challenger 350", "unable", "verification", "aircraft_images")
            ):
                issues.append("missing_image_handling")
        elif check == "challenger_350":
            if "350" not in blob and "challenger" not in blob:
                issues.append("missing_challenger_reference")
        elif check == "tail_or_verification":
            if "n628ts" not in blob.lower() and "tail" not in blob:
                issues.append("missing_tail_handling")

    if sc.get("store_state"):
        return (True, [])  # setup turn

    return (not issues, issues)


def main() -> None:
    conversation_states: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    passed = 0
    graded = 0

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
            results.append(
                {
                    "id": sc["id"],
                    "priority": sc["priority"],
                    "pass": True,
                    "issues": [],
                    "note": "setup_turn",
                    "mode": env.get("mode") if env else None,
                    "preview": (r.answer or "")[:400],
                }
            )
            continue

        ok, issues = _evaluate(sc, r.answer or "", du, env)
        graded += 1
        if ok:
            passed += 1

        results.append(
            {
                "id": sc["id"],
                "priority": sc["priority"],
                "pass": ok,
                "issues": issues,
                "mode": env.get("mode") if env else None,
                "component": env.get("component") if env else None,
                "has_images": bool(r.aircraft_images),
                "context_continuity": du.get("context_continuity"),
                "preview": (r.answer or "")[:500],
            }
        )

    out = {
        "summary": {"pass": f"{passed}/{graded}", "graded": graded},
        "results": results,
    }
    out_path = _ROOT / "evals" / "priority_benchmark_q18_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""
Six unused critical stress scenarios — industrial/ULR, governance, seasonal, multi-region,
payload variance, ULR continuation vs domestic saturation.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
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

SCENARIOS = [
    {
        "id": "extreme_asymmetry_industrial_ulr",
        "title": "Extreme asymmetry + industrial + ULR conflict",
        "query": (
            "We operate between Calgary oil fields, Houston headquarters, and London. "
            "Aircraft often land on short gravel strips, but executives also do quarterly nonstop "
            "London flights. One aircraft must cover everything if possible, but we've had dispatch "
            "failures before."
        ),
        "expect": [
            "industrial_or_field_access",
            "multi_segment_or_portfolio",
            "no_global7500_default",
            "gravel_or_runway_priority",
            "structural_or_fleet_decomposition",
        ],
    },
    {
        "id": "ceo_override_mixed_fleet",
        "title": "CEO override + mixed fleet demand tension",
        "query": (
            "Our CEO requires nonstop New York–Dubai capability. However, 80% of flights are "
            "2–3 hour domestic hops with 4 executives. We previously owned a large jet that was "
            "inefficient for daily use."
        ),
        "expect": [
            "governance_or_ceo_tension",
            "continuation_or_ulr_segment",
            "not_blind_global_only",
            "ownership_or_utilization_reasoning",
        ],
    },
    {
        "id": "seasonal_mountain_global_network",
        "title": "Seasonal mountain ops + global network",
        "query": (
            "We fly Los Angeles to Tokyo and Singapore regularly, but in winter we also run constant "
            "Aspen, Jackson Hole, and Telluride rotations. Our last aircraft struggled in ski season "
            "and caused multiple diversions."
        ),
        "expect": [
            "structural_or_dual_domain",
            "mountain_segment",
            "ulr_or_pacific_segment",
            "no_single_jet_collapse",
            "winter_or_seasonal",
        ],
    },
    {
        "id": "multi_continental_caribbean_reliability",
        "title": "Multi-continental + short runway islands + reliability-first",
        "query": (
            "We operate between Miami, São Paulo, Madrid, and multiple Caribbean islands with short "
            "runways. Reliability is more important than cabin luxury, and we've had dispatch issues "
            "with larger jets before."
        ),
        "expect": [
            "multi_corridor_routes",
            "multi_segment",
            "dispatch_reliability_emphasis",
            "no_one_jet_solves_all",
            "caribbean_or_short_runway",
        ],
    },
    {
        "id": "heavy_team_payload_variance",
        "title": "Heavy team movement + inconsistent payload profile",
        "query": (
            "We move teams ranging from 3 executives to 14-person deal groups between Chicago, New York, "
            "and Europe. Flights vary weekly, and we sometimes need cargo capacity for equipment."
        ),
        "expect": [
            "multi_segment_or_europe_band",
            "payload_or_pax_variance",
            "no_scalar_pax_only",
            "no_false_single_airliner",
        ],
    },
    {
        "id": "ulr_continuation_domestic_saturation",
        "title": "ULR continuation + domestic saturation contradiction",
        "query": (
            "We fly Boston, Chicago, and San Francisco frequently with small teams, but twice a month "
            "the founder flies nonstop to Abu Dhabi. We want simplicity and ideally one aircraft, but "
            "our previous jet was either too big or too limited."
        ),
        "expect": [
            "continuation_or_middle_east",
            "domestic_or_multi_city",
            "single_aircraft_preference_acknowledged",
            "structural_or_continuation_logic",
            "not_forced_global7500_only",
        ],
    },
]


@dataclass
class EvalResult:
    id: str
    title: str
    answer: str
    packet: Dict[str, Any]
    kernel: Dict[str, Any]
    fleet: Dict[str, Any]
    structural: Dict[str, Any]
    checks: Dict[str, bool] = field(default_factory=dict)
    anti_patterns: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def score(self) -> tuple[int, int]:
        passed = sum(1 for v in self.checks.values() if v)
        return passed, len(self.checks)


def _a(text: str) -> str:
    return (text or "").lower()


def evaluate(sc: Dict[str, Any], answer: str, pkt: Dict, kernel: Dict, fleet: Dict, structural: Dict) -> EvalResult:
    a = _a(answer)
    inf = pkt.get("inferred_constraints") or {}
    bands = [b.lower() for b in (pkt.get("fallback_operational_band") or [])]
    routes = pkt.get("explicit_constraints", {}).get("routes") or []
    struct_req = bool(
        structural.get("required")
        or kernel.get("structural_decomposition")
        or inf.get("incompatible_mission_bands")
    )
    has_segments = "operational segments:" in a
    has_portfolio = (
        "per-segment" in a
        or "multi-domain" in a
        or "fleet structure" in a
        or fleet.get("multi_aircraft_required")
        or struct_req
    )
    mentions_g7500_alone = bool(
        re.search(r"global\s*7500.*(?:only|primary|best|start with|recommend)", a)
        or (
            a.count("global 7500") >= 1
            and "per-segment" not in a
            and not has_portfolio
            and "structurally invalid" not in a
        )
    )
    global_options_without_segments = (
        "aircraft options:" in a and not has_segments and not has_portfolio
    )
    one_jet_solves = bool(
        re.search(r"one (?:aircraft|jet).*(?:everything|all|covers)", a)
        and "invalid" not in a
        and "structurally" not in a
    )

    r = EvalResult(
        id=sc["id"],
        title=sc["title"],
        answer=answer,
        packet=pkt,
        kernel=kernel,
        fleet=fleet,
        structural=structural,
    )
    r.anti_patterns = {
        "global7500_default_risk": mentions_g7500_alone,
        "global_options_flatten": global_options_without_segments,
        "one_jet_solves_all": one_jet_solves,
        "missing_kernel_block": "operational synthesis (authoritative)" not in a,
    }

    checks: Dict[str, bool] = {}

    if sc["id"] == "extreme_asymmetry_industrial_ulr":
        checks["industrial_or_field_access"] = any(
            w in a for w in ("industrial", "field-access", "field access", "gravel", "unpaved", "short-strip")
        ) or inf.get("industrial_airport_access")
        checks["multi_segment_or_portfolio"] = has_segments or has_portfolio
        checks["no_global7500_default"] = not mentions_g7500_alone or has_portfolio
        checks["gravel_or_runway_priority"] = any(
            w in a for w in ("gravel", "runway", "strip", "field", "dispatch")
        )
        checks["structural_or_fleet_decomposition"] = struct_req or bool(fleet.get("multi_aircraft_required"))

    elif sc["id"] == "ceo_override_mixed_fleet":
        checks["governance_or_ceo_tension"] = any(
            w in a for w in ("ceo", "domestic", "80%", "utilization", "inefficient", "daily")
        )
        checks["continuation_or_ulr_segment"] = any(
            "continuation" in b or "ulr" in b or "middle east" in b for b in bands
        ) or any(w in a for w in ("dubai", "continuation", "ulr"))
        checks["not_blind_global_only"] = not mentions_g7500_alone or has_segments
        checks["ownership_or_utilization_reasoning"] = any(
            w in a for w in ("owned", "ownership", "utilization", "inefficient", "daily", "domestic")
        )

    elif sc["id"] == "seasonal_mountain_global_network":
        checks["structural_or_dual_domain"] = struct_req or has_portfolio
        checks["mountain_segment"] = any(
            w in a for w in ("aspen", "jackson", "telluride", "mountain", "ski")
        )
        checks["ulr_or_pacific_segment"] = any(
            w in a for w in ("tokyo", "singapore", "ulr", "ultra-long", "pacific")
        )
        checks["no_single_jet_collapse"] = not one_jet_solves
        checks["winter_or_seasonal"] = "winter" in a or "ski" in a

    elif sc["id"] == "multi_continental_caribbean_reliability":
        checks["multi_corridor_routes"] = (
            len(routes) >= 2
            or sum(1 for city in ("miami", "sao paulo", "são paulo", "madrid") if city in a) >= 2
        )
        checks["multi_segment"] = has_segments or len(bands) >= 2
        checks["dispatch_reliability_emphasis"] = "dispatch" in a or "reliabilit" in a
        checks["no_one_jet_solves_all"] = not one_jet_solves and (
            has_portfolio or "structurally invalid" in a or "single platform" in a
        )
        checks["caribbean_or_short_runway"] = any(
            w in a for w in ("caribbean", "short runway", "short-runway", "island")
        )

    elif sc["id"] == "heavy_team_payload_variance":
        checks["multi_segment_or_europe_band"] = has_segments or any("europe" in b or "transatlantic" in b for b in bands)
        checks["payload_or_pax_variance"] = any(
            w in a for w in ("14", "3 executive", "cargo", "equipment", "deal group", "passenger")
        ) or (pkt.get("explicit_constraints") or {}).get("passengers") not in (None, 0)
        checks["no_scalar_pax_only"] = not (
            re.search(r"\b\d+\s+passengers?\b", a)
            and "14" not in a
            and "range" not in a
            and "vary" not in a
        ) or any(w in a for w in ("range", "vary", "3", "14", "cargo"))
        checks["no_false_single_airliner"] = "737" not in a or "narrow" in a

    elif sc["id"] == "ulr_continuation_domestic_saturation":
        checks["continuation_or_middle_east"] = any(
            w in a for w in ("abu dhabi", "continuation", "middle east", "ulr")
        )
        checks["domestic_or_multi_city"] = any(
            w in a for w in ("boston", "chicago", "san francisco", "domestic")
        ) or len(routes) >= 2
        checks["single_aircraft_preference_acknowledged"] = any(
            w in a for w in ("one aircraft", "simplicity", "ideally one", "single platform")
        )
        checks["structural_or_continuation_logic"] = (
            any("continuation" in b for b in bands) or has_segments or struct_req
        )
        checks["not_forced_global7500_only"] = not mentions_g7500_alone or has_segments

    r.checks = checks
    r.notes = [
        f"structural={struct_req}",
        f"segments={has_segments}",
        f"portfolio={has_portfolio}",
        f"routes={routes[:4]}",
        f"bands={len(bands)}",
        f"kernel_auth={kernel.get('authorized_models')}",
        f"anti={ {k: v for k, v in r.anti_patterns.items() if v} }",
    ]
    return r


def main() -> int:
    results: List[EvalResult] = []
    print("CRITICAL UNUSED SCENARIO REPORT\n" + "=" * 60)
    for sc in SCENARIOS:
        print(f"\n### {sc['title']}")
        data_used: Dict[str, Any] = {"consultant_response_mode": "mission_advisory"}
        orch = run_consultant_orchestration(
            sc["query"],
            conversation_state={"history": []},
            data_used=data_used,
            query_intent="mission_feasibility",
        )
        if isinstance(orch.data_used_patch, dict):
            data_used.update(orch.data_used_patch)
        pkt = data_used.get("mission_understanding_packet") or {}
        kernel = data_used.get("mission_authority_kernel") or {}
        fleet = data_used.get("fleet_composition_plan") or {}
        structural = data_used.get("structural_decomposition") or {}
        if not isinstance(pkt, dict):
            pkt = {}
        if not isinstance(kernel, dict):
            kernel = {}
        if not isinstance(fleet, dict):
            fleet = {}
        if not isinstance(structural, dict):
            structural = {}

        ev = evaluate(sc, orch.answer or "", pkt, kernel, fleet, structural)
        results.append(ev)
        passed, total = ev.score()
        status = "PASS" if passed == total and not any(ev.anti_patterns.values()) else "PARTIAL" if passed >= total - 1 else "FAIL"
        print(f"Status: {status} ({passed}/{total} checks)")
        print(f"Checks: {ev.checks}")
        if any(ev.anti_patterns.values()):
            print(f"Anti-patterns: {ev.anti_patterns}")
        print(f"Notes: {ev.notes}")
        print(f"\nPreview:\n{(orch.answer or '')[:700]}...\n")

    summary = {
        "results": [
            {
                "id": r.id,
                "title": r.title,
                "checks": r.checks,
                "anti_patterns": r.anti_patterns,
                "notes": r.notes,
                "answer_preview": (r.answer or "")[:1500],
            }
            for r in results
        ]
    }
    out = _ROOT / "evals" / "critical_unused_scenario_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    total_pass = sum(r.score()[0] for r in results)
    total_checks = sum(r.score()[1] for r in results)
    anti_hits = sum(1 for r in results if any(r.anti_patterns.values()))
    print(f"\nAGGREGATE: {total_pass}/{total_checks} checks passed; {anti_hits}/6 scenarios with anti-patterns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

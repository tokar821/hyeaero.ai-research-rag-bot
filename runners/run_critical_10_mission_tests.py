"""
Ten critical unused mission tests — extraction, pre-ranking representation,
structural reasoning, ranking hygiene.
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

if (_ROOT / ".env").exists():
    load_dotenv(_ROOT / ".env")

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402

SCENARIOS = [
    {
        "id": "arctic_ulr_remote",
        "title": "Arctic + ULR + remote ops",
        "query": (
            "We move 6–10 engineers between Anchorage, remote Arctic oil platforms, Calgary, "
            "and occasionally London. Some sites require gravel runway operations in winter darkness. "
            "CEO insists on nonstop London capability. Is a single aircraft realistic?"
        ),
        "cities": ("anchorage", "calgary", "london"),
        "continuation": ("london",),
        "industrial": True,
        "governance": True,
    },
    {
        "id": "governance_domestic_ceo_prestige",
        "title": "Extreme governance conflict",
        "query": (
            "We fly 85% short hops (NYC–Boston–DC) with 3–5 executives, but the CEO insists on "
            "nonstop Tokyo and Dubai access quarterly. We previously bought a Global 6000 and it was "
            "massively underutilized. What structure makes sense now?"
        ),
        "cities": ("new york", "boston", "tokyo", "dubai"),
        "continuation": ("tokyo", "dubai"),
        "industrial": False,
        "governance": True,
    },
    {
        "id": "multi_continental_pax_cargo",
        "title": "Multi-continental + pax + cargo",
        "query": (
            "We transport between 4 and 16 people depending on mission: engineers, deal teams, and "
            "equipment between Houston, São Paulo, Lagos, and Frankfurt. Cargo space matters more than cabin. "
            "What aircraft strategy actually works?"
        ),
        "cities": ("houston", "sao paulo", "são paulo", "lagos", "frankfurt"),
        "continuation": (),
        "industrial": False,
        "governance": False,
        "cargo": True,
        "pax_range": (4, 16),
    },
    {
        "id": "mountain_desert_transatlantic",
        "title": "Mountain + desert + transatlantic",
        "query": (
            "We operate from Denver, Riyadh, Zurich, and Telluride. Winter ski access and Middle East "
            "heat performance both matter. Previous aircraft struggled in high-altitude airports. "
            "How should we structure the fleet?"
        ),
        "cities": ("denver", "riyadh", "zurich", "telluride"),
        "continuation": ("riyadh",),
        "industrial": False,
        "governance": False,
    },
    {
        "id": "short_islands_europe_asia_ulr",
        "title": "Short runway islands + Europe + Asia ULR",
        "query": (
            "We fly Miami–Caribbean islands with 3,000 ft runways, but also do Miami–Singapore and "
            "Miami–Paris regularly. We want one aircraft if possible but reliability matters more than luxury."
        ),
        "cities": ("miami", "singapore", "paris"),
        "continuation": ("singapore", "paris"),
        "industrial": False,
        "governance": False,
        "short_runway": True,
    },
    {
        "id": "industrial_mining_africa_eu",
        "title": "Industrial mining + Africa + EU HQ",
        "query": (
            "We move geologists into unpaved mining strips in West Africa and Canada, but executives fly "
            "regularly to London and Zurich. Aircraft must survive harsh runway conditions and still cross "
            "the Atlantic."
        ),
        "cities": ("london", "zurich"),
        "continuation": (),
        "industrial": True,
        "governance": False,
    },
    {
        "id": "hub_imbalance_continuation",
        "title": "Executive hub + hidden continuations",
        "query": (
            "We are based in NYC. Daily flights are NYC–Chicago–SF with 4 executives. But the founder also "
            "flies nonstop to Abu Dhabi and sometimes Singapore. The rest of the company never leaves "
            "North America. What structure fits?"
        ),
        "cities": ("new york", "chicago", "san francisco", "abu dhabi", "singapore"),
        "continuation": ("abu dhabi", "singapore"),
        "industrial": False,
        "governance": True,
    },
    {
        "id": "cargo_heavy_stol_intercontinental",
        "title": "Cargo-heavy + STOL + intercontinental",
        "query": (
            "We transport high-value equipment and 6–12 personnel between Calgary, Houston, remote drilling "
            "sites, and Madrid. Some legs require STOL capability. Others require intercontinental range."
        ),
        "cities": ("calgary", "houston", "madrid"),
        "continuation": (),
        "industrial": True,
        "governance": False,
        "cargo": True,
        "pax_range": (6, 12),
    },
    {
        "id": "failed_prior_ski_asia",
        "title": "Failed prior ownership + ski + Asia",
        "query": (
            "We previously owned a large long-range jet but dispatch reliability into Aspen, Jackson Hole, "
            "and European winter airports caused repeated failures. Now we need both ski access and Asia capability."
        ),
        "cities": ("aspen", "jackson"),
        "continuation": (),
        "industrial": False,
        "governance": False,
    },
    {
        "id": "dual_fleet_single_aircraft_pressure",
        "title": "Dual fleet + forced single aircraft",
        "query": (
            "We have two competing demands: executives flying 2–3 hour domestic US hops daily, and quarterly "
            "missions to Riyadh, Dubai, and Singapore. Leadership insists on one aircraft only. Is that actually possible?"
        ),
        "cities": ("riyadh", "dubai", "singapore"),
        "continuation": ("riyadh", "dubai", "singapore"),
        "industrial": False,
        "governance": True,
    },
]


@dataclass
class ScenarioReport:
    id: str
    title: str
    answer: str
    extraction: Dict[str, bool] = field(default_factory=dict)
    representation: Dict[str, bool] = field(default_factory=dict)
    structure: Dict[str, bool] = field(default_factory=dict)
    ranking: Dict[str, bool] = field(default_factory=dict)
    data_snapshot: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def grade(self) -> str:
        all_checks = (
            list(self.extraction.values())
            + list(self.representation.values())
            + list(self.structure.values())
            + list(self.ranking.values())
        )
        if not all_checks:
            return "?"
        passed = sum(1 for v in all_checks if v)
        total = len(all_checks)
        anti = sum(1 for v in self.ranking.values() if v is False)
        if passed == total and anti == 0:
            return "PASS"
        if passed >= total - 1:
            return "PARTIAL"
        return "FAIL"


def _eval(sc: Dict[str, Any], answer: str, du: Dict[str, Any]) -> ScenarioReport:
    a = (answer or "").lower()
    pkt = du.get("mission_understanding_packet") or {}
    inf = pkt.get("inferred_constraints") or {}
    exp = pkt.get("explicit_constraints") or {}
    pre = du.get("pre_ranking_representation") or {}
    gov = du.get("mission_governance") or {}
    industrial = du.get("industrial_airport_profile") or {}
    route_graph = du.get("mission_route_graph") or {}
    kernel = du.get("mission_authority_kernel") or {}
    structural = du.get("structural_decomposition") or {}
    fleet = du.get("fleet_composition_plan") or {}

    routes = exp.get("routes") or route_graph.get("route_labels") or []
    routes_l = " ".join(str(r).lower() for r in routes)
    dist = exp.get("passenger_distribution") or pre.get("passenger_distribution") or {}

    rep = ScenarioReport(id=sc["id"], title=sc["title"], answer=answer or "")

    # A. Extraction
    query_text = sc.get("query", "")
    from services.mission.mission_place_index import city_captured, normalize_place_key
    from services.mission.models import MissionProfile, Route

    prof_check = MissionProfile()
    for lbl in routes:
        r = Route.from_label(str(lbl))
        if r:
            prof_check.routes.append(r)
    places_captured = exp.get("places_captured") or []
    cities_ok = all(
        city_captured(c, prof_check, text=query_text)
        or any(
            normalize_place_key(c) in normalize_place_key(p)
            for p in places_captured
        )
        for c in sc.get("cities", ())
    )
    rep.extraction["all_key_cities_captured"] = cities_ok

    cont_needed = sc.get("continuation") or ()
    if cont_needed:
        cont_ok = any(c in routes_l for c in cont_needed) or any(
            c in " ".join(route_graph.get("inferred_leg_labels") or []).lower()
            for c in cont_needed
        )
    else:
        cont_ok = len(routes) >= 1
    rep.extraction["continuation_legs_in_graph"] = cont_ok

    if sc.get("industrial"):
        rep.extraction["industrial_or_runway_detected"] = bool(
            industrial.get("active")
            or inf.get("industrial_airport_access")
            or any(w in a for w in ("gravel", "unpaved", "mining", "industrial", "stol", "runway"))
        )
    else:
        rep.extraction["industrial_or_runway_detected"] = True

    if sc.get("short_runway"):
        rep.extraction["short_runway_signal"] = any(
            w in a for w in ("3000", "short runway", "short-runway", "caribbean")
        )
    else:
        rep.extraction["short_runway_signal"] = True

    # B. Representation
    if sc.get("pax_range"):
        lo, hi = sc["pax_range"]
        rep.representation["pax_is_distribution"] = bool(
            dist.get("is_variable")
            or (dist.get("min_pax") == lo and dist.get("max_pax") == hi)
            or inf.get("passenger_load_variable")
        )
    else:
        rep.representation["pax_is_distribution"] = bool(
            dist.get("is_variable") or dist.get("min_pax") or True
        )

    if sc.get("cargo"):
        rep.representation["cargo_flagged"] = bool(
            dist.get("cargo_required") or inf.get("runway_over_cabin") or "cargo" in a
        )
    else:
        rep.representation["cargo_flagged"] = True

    if sc.get("governance"):
        rep.representation["governance_detected"] = bool(
            gov.get("utilization_mission_conflict")
            or gov.get("ceo_ulr_mandate")
            or inf.get("governance_asymmetry")
            or inf.get("defer_global_shortlist")
        )
    else:
        rep.representation["governance_detected"] = True

    rep.representation["pre_ranking_applied"] = bool(du.get("pre_ranking_applied"))

    # C. Structural
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
    single_invalid = (
        "structurally invalid" in a
        or "single platform" in a
        or "not realistic" in a
        or "cannot share one platform" in a
        or inf.get("incompatible_mission_bands")
    )

    rep.structure["multi_segment_or_portfolio"] = has_segments or has_portfolio
    rep.structure["structural_or_fleet_decomposition"] = struct_req or has_portfolio
    rep.structure["single_aircraft_invalidated_when_needed"] = (
        not sc.get("governance") and not "single aircraft" in sc["query"].lower()
    ) or single_invalid or has_portfolio

    # D. Ranking hygiene
    g7500_alone = bool(
        re.search(r"global\s*7500.*(?:only|primary|best)", a)
        or (
            "aircraft options:" in a
            and "global 7500" in a
            and "per-segment" not in a
            and not inf.get("defer_global_shortlist")
        )
    )
    one_jet_bad = bool(
        re.search(r"one (?:aircraft|jet).*(?:everything|all|covers)", a)
        and "invalid" not in a
        and "structurally" not in a
    )
    premature = bool(
        "aircraft options:" in a
        and not has_segments
        and not struct_req
        and not inf.get("defer_global_shortlist")
    )

    rep.ranking["no_global7500_default"] = not g7500_alone
    rep.ranking["no_one_jet_collapse"] = not one_jet_bad
    rep.ranking["no_premature_shortlist"] = not premature
    rep.ranking["kernel_authoritative_block"] = "operational synthesis (authoritative)" in a

    rep.data_snapshot = {
        "routes": routes[:6],
        "inferred_continuations": route_graph.get("inferred_leg_labels"),
        "passenger_distribution": dist,
        "governance": gov,
        "industrial": industrial,
        "structural_required": structural.get("required"),
        "defer_global_shortlist": inf.get("defer_global_shortlist"),
        "recommend_aircraft": pkt.get("recommend_aircraft"),
    }
    rep.notes = [
        f"routes={len(routes)}",
        f"struct={struct_req}",
        f"defer_rank={inf.get('defer_global_shortlist')}",
    ]
    return rep


def main() -> int:
    reports: List[ScenarioReport] = []
    print("CRITICAL 10 MISSION TEST REPORT\n" + "=" * 70)

    for sc in SCENARIOS:
        du: Dict[str, Any] = {"consultant_response_mode": "mission_advisory"}
        orch = run_consultant_orchestration(
            sc["query"],
            conversation_state={"history": []},
            data_used=du,
            query_intent="mission_feasibility",
        )
        if isinstance(orch.data_used_patch, dict):
            du.update(orch.data_used_patch)

        rep = _eval(sc, orch.answer or "", du)
        reports.append(rep)
        grade = rep.grade()
        print(f"\n## {sc['id']} — {sc['title']} [{grade}]")
        print(f"A Extraction:      {rep.extraction}")
        print(f"B Representation:  {rep.representation}")
        print(f"C Structure:       {rep.structure}")
        print(f"D Ranking hygiene: {rep.ranking}")
        print(f"Snapshot: {json.dumps(rep.data_snapshot, indent=2)[:500]}...")
        print(f"Preview: {(orch.answer or '')[:400]}...")

    out = _ROOT / "evals" / "critical_10_mission_results.json"
    payload = {
        "results": [
            {
                "id": r.id,
                "title": r.title,
                "grade": r.grade(),
                "extraction": r.extraction,
                "representation": r.representation,
                "structure": r.structure,
                "ranking": r.ranking,
                "data_snapshot": r.data_snapshot,
                "notes": r.notes,
                "answer_preview": (r.answer or "")[:2000],
            }
            for r in reports
        ]
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    grades = [r.grade() for r in reports]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: PASS={grades.count('PASS')} PARTIAL={grades.count('PARTIAL')} FAIL={grades.count('FAIL')}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

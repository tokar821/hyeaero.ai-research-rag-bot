"""Geographic + Route Intelligence — NEW 10 Critical test questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    attach_mission_understanding,
    build_mission_understanding,
)
from services.mission.pre_ranking_representation import apply_pre_ranking_representation
from services.mission.mission_authority_kernel import build_mission_authority_kernel


QUERIES: List[Tuple[str, str, List[str], List[str]]] = [
    (
        "Q1 Arctic + Industrial + EU Executive conflict",
        "We operate oil extraction sites in Nunavut and Northern Alaska gravel strips, but also "
        "fly executives Houston → London and Houston → Frankfurt monthly nonstop. Winter dispatch "
        "failures keep happening. Should this be one aircraft or multiple?",
        ["houston", "london", "frankfurt", "nunavut", "gravel"],
        [
            "remote drilling sites -> london",
            "nunavut -> paris",
            "gravel -> frankfurt",
            "houston -> northern alberta",
        ],
    ),
    (
        "Q2 Australia Mining + Asia + Europe overlay",
        "We run mining logistics in Western Australia (Perth, Pilbara) and also move executives "
        "Perth → Singapore → London, sometimes routing through Middle East fuel stops. "
        "What structure fits?",
        ["perth", "pilbara", "singapore", "london"],
        ["perth -> paris", "singapore -> aspen", "pilbara -> zurich"],
    ),
    (
        "Q3 Florida + Caribbean + Middle East continuation trap",
        "We fly executives from Miami to Caribbean islands (Bahamas, Turks & Caicos) but also do "
        "Miami → Dubai → Riyadh nonstop rotations. Is this one mission or two?",
        ["miami", "caribbean", "dubai", "riyadh"],
        ["dubai -> caribbean", "riyadh -> caribbean", "tokyo -> caribbean", "doha -> miami"],
    ),
    (
        "Q4 Texas Energy + West Africa + EU executive conflict",
        "We operate Houston → Permian Basin drilling fields, and also fly executives "
        "Houston → Lagos → Paris → Zurich regularly. How should this network be structured?",
        ["houston", "permian", "lagos", "paris", "zurich"],
        [
            "remote drilling -> paris",
            "permian -> zurich",
            "paris -> desert",
            "nigeria -> geneva",
        ],
    ),
    (
        "Q5 Multi-hub Asia-Pacific + Ski conflict",
        "We operate between Los Angeles, Tokyo, Seoul, Singapore, but also move teams into "
        "Aspen, Jackson Hole, and Banff ski regions in winter. What aircraft structure makes sense?",
        ["los angeles", "tokyo", "seoul", "singapore", "aspen", "jackson hole", "banff"],
        [
            "aspen -> tokyo",
            "aspen -> singapore",
            "aspen -> dubai",
            "tokyo -> caribbean",
            "singapore -> aspen",
        ],
    ),
    (
        "Q6 Europe-heavy + Africa industrial + Arctic overlay",
        "We fly Frankfurt → Zurich → London executive rotations, but also support West African "
        "mining + Northern Canada Arctic gravel strip operations. Is one aircraft feasible?",
        ["frankfurt", "zurich", "london", "gravel", "west africa"],
        [
            "remote drilling -> london",
            "gravel -> paris",
            "west africa -> zurich",
            "frankfurt -> permian",
        ],
    ),
    (
        "Q7 Founder dominance + asymmetric utilization",
        "A founder requires New York → Singapore nonstop monthly, but 80% of flying is "
        "New York → Boston → Washington DC → Chicago shuttle hops. Previously we had an "
        "oversized jet sitting idle.",
        ["new york", "singapore", "boston", "washington", "chicago"],
        ["dubai -> new york", "singapore -> new york", "boston -> singapore"],
    ),
    (
        "Q8 South America + Middle East + EU triangle",
        "We operate São Paulo ↔ Lagos ↔ Frankfurt, and sometimes insert Dubai fuel stops "
        "for executive continuity. How should this mission be structured?",
        ["sao paulo", "lagos", "frankfurt", "dubai"],
        ["lagos -> caribbean", "são paulo -> caribbean", "dubai -> new york"],
    ),
    (
        "Q9 Mountain + Desert + Transatlantic split",
        "We fly executives between Denver, Salt Lake City, and Aspen ski regions, but also "
        "operate Denver → Dubai → London transatlantic executive missions. What structure fits?",
        ["denver", "salt lake", "aspen", "dubai", "london"],
        [
            "aspen -> dubai",
            "aspen -> london",
            "salt lake -> dubai",
            "denver -> caribbean",
        ],
    ),
    (
        "Q10 Extreme multi-domain stress (full topology test)",
        "We operate across Houston oil fields, Northern Canada Arctic gravel strips, "
        "Miami Caribbean rotations, London executive HQ, and Singapore Asia hub. "
        "We previously tried a single aircraft and had winter + range failures. "
        "What is structurally valid?",
        ["houston", "gravel", "miami", "caribbean", "london", "singapore"],
        [
            "aspen -> singapore",
            "aspen -> dubai",
            "dubai -> caribbean",
            "singapore -> caribbean",
            "gravel -> paris",
            "miami -> london",
        ],
    ),
]


def _run_pipeline(q: str) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    data_used: Dict[str, Any] = {}
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    pkt = build_mission_understanding(q, profile, mission, broker_memory=None)
    attach_mission_understanding(data_used, pkt)
    profile, mission, pkt = apply_pre_ranking_representation(
        q, profile, mission, pkt, data_used=data_used
    )
    attach_mission_understanding(data_used, pkt)
    kernel = build_mission_authority_kernel(
        mission, pkt, profile, data_used=data_used, recommendations=[], query=q
    )
    segments = [
        {
            "label": s.label,
            "kind": getattr(s.kind, "value", str(s.kind)),
            "routes": list(s.route_labels or [])[:4],
        }
        for s in (kernel.segments or [])
    ]
    meta = {
        "decomposition": bool(kernel.structural_decomposition),
        "proof": (data_used.get("mission_structure_resolution") or {}).get("proof_source"),
        "suppressed": (data_used.get("recommendation_suppression") or {}).get(
            "suppress_aircraft_specificity"
        ),
        "geo": data_used.get("geographic_route_intelligence") or {},
        "dir": data_used.get("route_directionality") or {},
        "topo": data_used.get("route_topology_validation") or {},
        "graph": data_used.get("geographic_graph_authority") or {},
        "segments": segments,
    }
    routes = list(profile.route_labels() or mission.routes or [])
    return routes, data_used, meta


def _check(title: str, routes: List[str], meta: Dict, expect: List[str], reject: List[str]) -> Dict:
    blob = " ".join(routes).lower()
    expect_hits = [e for e in expect if e in blob]
    expect_miss = [e for e in expect if e not in blob]
    reject_hits = [x for x in reject if x in blob]
    if reject_hits:
        status = "FAIL"
    elif len(expect_miss) > max(1, len(expect) // 2):
        status = "PARTIAL"
    elif expect_miss:
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "title": title,
        "status": status,
        "routes": routes,
        "expect_hits": expect_hits,
        "expect_miss": expect_miss,
        "reject_hits": reject_hits,
        **meta,
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    for title, query, expect, reject in QUERIES:
        routes, du, meta = _run_pipeline(query)
        r = _check(title, routes, meta, expect, reject)
        results.append(r)
        counts[r["status"]] += 1

    print("=" * 92)
    print("GEOGRAPHIC + ROUTE INTELLIGENCE — NEW 10 CRITICAL TEST REPORT")
    print(f"SUMMARY: PASS={counts['PASS']} PARTIAL={counts['PARTIAL']} FAIL={counts['FAIL']}")
    print("=" * 92)

    for r in results:
        print()
        print(f"## {r['title']} [{r['status']}]")
        print(f"Routes ({len(r['routes'])}): {'; '.join(r['routes'])}")
        print(
            f"Decomposition: {r['decomposition']} | proof: {r['proof']} | suppressed: {r['suppressed']}"
        )
        geo = r.get("geo") or {}
        if geo.get("regions_activated"):
            print(f"Geo regions: {', '.join(geo['regions_activated'])}")
        if geo.get("routes_added"):
            print(f"Geo added: {'; '.join(geo['routes_added'][:6])}")
        dir_r = r.get("dir") or {}
        if dir_r.get("removed"):
            print(f"Direction removed: {'; '.join(dir_r['removed'])}")
        topo = r.get("topo") or {}
        if topo.get("removed_routes"):
            print(f"Topology removed: {'; '.join(topo['removed_routes'])}")
        if r["expect_hits"]:
            print(f"Expected OK: {', '.join(r['expect_hits'])}")
        if r["expect_miss"]:
            print(f"Expected missing: {', '.join(r['expect_miss'])}")
        if r["reject_hits"]:
            print(f"FAIL — rejected present: {', '.join(r['reject_hits'])}")
        if r.get("segments"):
            print("Segments:")
            for s in r["segments"][:5]:
                rt = "; ".join(s.get("routes") or []) or "(none)"
                print(f"  - {s.get('label')} ({s.get('kind')}): {rt}")

    out = _BACKEND / "evals" / "geo_intel_new_10_results.json"
    out.write_text(
        json.dumps({"summary": counts, "results": results}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

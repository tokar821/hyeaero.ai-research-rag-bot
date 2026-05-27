"""Geographic + Route Intelligence — FINAL 10 test questions."""

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
from services.mission.mission_authority_kernel import (
    build_mission_authority_kernel,
)


QUERIES: List[Tuple[str, str, List[str], List[str]]] = [
    (
        "Q1 Arctic + industrial + transatlantic",
        "We operate between Houston, Northern Alberta oil fields, and London. Some sites use "
        "gravel strips near the Arctic. How should the routing be structured?",
        ["houston", "london", "northern alberta", "yellowknife", "remote gravel", "calgary"],
        ["houston -> northern alberta", "paris -> arctic", "remote drilling sites -> london"],
    ),
    (
        "Q2 EU executive preservation under industrial",
        "We run drilling operations in Texas and Nigeria, but executives regularly fly between "
        "Paris, Geneva, and Houston. What does the route graph look like?",
        ["houston", "paris", "geneva", "lagos"],
        ["remote drilling -> paris", "nigeria -> geneva", "paris -> desert"],
    ),
    (
        "Q3 Arctic + Canada hub validation",
        "We move crews between Calgary, Yellowknife, and Nunavut, while also flying executives "
        "to Frankfurt. How should this network be organized?",
        ["calgary", "yellowknife", "nunavut", "frankfurt"],
        ["houston -> nunavut", "yellowknife -> london"],
    ),
    (
        "Q4 Cross-domain ghost (Asia to Caribbean)",
        "We operate Los Angeles to Tokyo, Singapore to Dubai, and also run Caribbean island "
        "operations out of Miami. How should routing be structured?",
        ["los angeles -> tokyo", "miami", "caribbean", "singapore", "dubai"],
        ["tokyo -> caribbean", "dubai -> caribbean", "singapore -> caribbean", "los angeles -> caribbean"],
    ),
    (
        "Q5 Industrial Texas + Europe mis-anchoring",
        "We move equipment from Permian Basin sites in Texas to Northern Africa and also send "
        "executives to Paris and Zurich. What is the correct routing structure?",
        ["houston", "paris", "zurich", "permian", "northern africa"],
        ["paris -> desert", "zurich -> desert", "remote drilling sites -> paris"],
    ),
    (
        "Q6 Hub priority + inversion stress",
        "Our base is New York, but we also operate Dubai to London and Singapore to Dubai routes. "
        "How should the system structure this?",
        ["new york", "london", "singapore", "dubai"],
        ["dubai -> new york", "singapore -> new york", "doha -> new york"],
    ),
    (
        "Q7 Arctic + Europe + missing ontology",
        "We fly between Arctic drilling sites in Northern Canada and also operate frequent flights "
        "between Calgary and Frankfurt. How should routing be organized?",
        ["calgary", "frankfurt", "remote gravel", "yellowknife", "northern alberta"],
        ["remote drilling sites -> frankfurt", "arctic industrial -> paris"],
    ),
    (
        "Q8 Florida corridor + ME continuation",
        "We operate Miami to Palm Beach and Orlando, but also fly nonstop Miami to Riyadh and "
        "occasionally Abu Dhabi. What is the correct structure?",
        ["miami", "palm beach", "riyadh", "abu dhabi"],
        ["madrid -> caribbean", "doha -> miami", "riyadh -> miami"],
    ),
    (
        "Q9 Australia mining + EU executive",
        "We run mining operations in Perth and Pilbara, but executives also fly between Frankfurt "
        "and Zurich. How should routing be modeled?",
        ["perth", "frankfurt", "zurich"],
        ["singapore -> aspen", "perth -> paris"],
    ),
    (
        "Q10 Multi-region overload + ghost prevention",
        "We previously tried a network covering Houston, Lagos, Paris, Singapore, Aspen, Dubai, "
        "and Caribbean islands, but routing became inconsistent. How should this be decomposed?",
        ["houston", "lagos", "paris", "singapore", "aspen", "dubai", "caribbean"],
        ["aspen -> singapore", "aspen -> dubai", "singapore -> aspen", "tokyo -> caribbean", "dubai -> caribbean"],
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
    print("GEOGRAPHIC + ROUTE INTELLIGENCE — FINAL 10 TEST REPORT")
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
        if geo.get("routes_rebalanced"):
            print(f"Geo rebalanced: {'; '.join(geo['routes_rebalanced'])}")
        dir_r = r.get("dir") or {}
        if dir_r.get("swapped"):
            print(f"Direction swapped: {'; '.join(dir_r['swapped'])}")
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

    out = _BACKEND / "evals" / "geo_intel_final_10_results.json"
    out.write_text(
        json.dumps({"summary": counts, "results": results}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

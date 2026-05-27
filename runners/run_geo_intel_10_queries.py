"""Run 10 unused geography + route intelligence test queries and emit structured report."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

from run_phase2_user_queries import run_one

QUERIES: List[Tuple[str, str, List[str], List[str]]] = [
    (
        "Q1 Industrial + Arctic + Transatlantic",
        "We operate between Houston, Calgary oil fields, and London. Some sites are gravel strips in "
        "Northern Canada. Can a single aircraft realistically handle this network?",
        ["houston", "calgary", "london", "arctic", "drilling", "remote"],
        ["paris -> arctic", "london -> arctic industrial"],
    ),
    (
        "Q2 Australia mining + Europe executive",
        "We move mining teams from Perth to remote Pilbara sites, but executives also fly regularly "
        "to Frankfurt and Zurich. How should we structure routing?",
        ["perth", "frankfurt", "zurich"],
        ["singapore -> aspen", "perth -> paris"],
    ),
    (
        "Q3 Florida corridor + ME continuation",
        "Our flights are mostly Miami to Palm Beach and Orlando, but the CEO also flies nonstop "
        "Miami to Riyadh and occasionally Dubai. What does the route structure look like?",
        ["miami", "palm beach", "riyadh", "dubai"],
        ["madrid -> caribbean", "doha -> miami"],
    ),
    (
        "Q4 Asia-Pacific hub confusion",
        "We operate Los Angeles to Tokyo, Tokyo to Singapore, and also seasonal ski operations "
        "into Aspen and Jackson Hole. How should the network be organized?",
        ["los angeles -> tokyo", "aspen", "jackson"],
        ["singapore -> aspen", "aspen -> tokyo", "aspen -> singapore"],
    ),
    (
        "Q5 West Africa industrial + EU executive",
        "We run operations between Lagos mining regions, Houston HQ, and Frankfurt executive travel. "
        "Some flights continue through Paris. What is the correct routing structure?",
        ["houston", "lagos", "frankfurt", "paris"],
        ["lagos -> aspen", "west africa -> zurich"],
    ),
    (
        "Q6 Canada Arctic + US domestic corridor",
        "We fly between Calgary, Yellowknife, and Houston, but also do regular New York to Chicago "
        "shuttle flights. How should these be connected operationally?",
        ["calgary", "houston", "new york", "chicago"],
        ["yellowknife -> london", "calgary -> riyadh"],
    ),
    (
        "Q7 Hub inversion trap",
        "Our main base is New York, but we also operate Singapore to Dubai and Dubai to London routes. "
        "How should we structure this system?",
        ["new york", "singapore", "dubai", "london"],
        ["doha -> new york", "singapore -> new york"],
    ),
    (
        "Q8 Desert logistics + European capital bias",
        "We transport equipment between Texas desert drilling sites and Northern Africa, but executives "
        "also fly to Paris and Geneva. How should the routing graph be structured?",
        ["houston", "dallas", "desert", "paris", "geneva"],
        ["paris -> desert", "dubai -> desert"],
    ),
    (
        "Q9 Caribbean + South America ambiguity",
        "We operate between Miami, São Paulo, and Caribbean islands with short runways, but also "
        "occasionally fly to Madrid. What is the correct hub structure?",
        ["miami", "sao paulo", "caribbean", "madrid"],
        ["madrid -> caribbean", "sao paulo -> zurich"],
    ),
    (
        "Q10 Multi-region overload (ghost edge stress)",
        "We previously tried a single network covering Los Angeles, Aspen, Tokyo, Singapore, Dubai, "
        "and Caribbean islands, but routing became inconsistent and unreliable. How should this be decomposed?",
        ["los angeles", "aspen", "tokyo", "singapore", "dubai", "caribbean"],
        ["aspen -> singapore", "aspen -> dubai", "caribbean -> dubai", "singapore -> aspen"],
    ),
]


def _blob(items: List[str]) -> str:
    return " ".join(items).lower()


def _check(title: str, r: Dict[str, Any], expect: List[str], reject: List[str]) -> Dict[str, Any]:
    routes = r.get("routes") or []
    blob = _blob(routes)
    geo = r.get("geo_enrichment") or {}
    dir_r = (r.get("data_used") or {}).get("route_directionality") or {}
    topo = r.get("topology_removed") or []

    expect_hits = [e for e in expect if e in blob]
    expect_miss = [e for e in expect if e not in blob]
    reject_hits = [x for x in reject if x in blob]

    sr = r.get("structure_resolution") or {}
    sup = r.get("suppression") or {}

    # Verdict
    if reject_hits:
        status = "FAIL"
    elif len(expect_miss) > len(expect) // 2:
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
        "decomposition": r.get("structural_decomposition"),
        "proof": sr.get("proof_source"),
        "suppressed": sup.get("suppress_aircraft_specificity"),
        "geo_regions": geo.get("regions_activated") or [],
        "geo_added": geo.get("routes_added") or [],
        "geo_rebalanced": geo.get("routes_rebalanced") or [],
        "dir_swapped": dir_r.get("swapped") or [],
        "dir_removed": dir_r.get("removed") or [],
        "topology_removed": topo,
        "segments": [
            f"{s.get('label')} ({s.get('kind')}): {'; '.join(s.get('routes') or [])[:80]}"
            for s in (r.get("segments") or [])
        ],
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    for title, query, expect, reject in QUERIES:
        raw = run_one(title, query)
        # attach data_used for directionality (run_one doesn't expose it)
        raw["data_used"] = {}  # run_one doesn't return data_used; re-fetch from geo only
        report = _check(title, raw, expect, reject)
        results.append(report)

    # Re-run with data_used for directionality detail
    from services.consultant.mission_state import MissionState
    from services.mission.mission_extractor import extract_mission
    from services.mission.mission_understanding_engine import (
        attach_mission_understanding,
        build_mission_understanding,
    )
    from services.mission.pre_ranking_representation import apply_pre_ranking_representation

    for i, (title, query, expect, reject) in enumerate(QUERIES):
        du: Dict = {}
        profile = extract_mission(query)
        mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
        pkt = build_mission_understanding(query, profile, mission, broker_memory=None)
        attach_mission_understanding(du, pkt)
        profile, mission, pkt = apply_pre_ranking_representation(
            query, profile, mission, pkt, data_used=du
        )
        raw = run_one(title, query)
        raw["data_used"] = du
        raw["routes"] = list(profile.route_labels() or mission.routes or [])
        raw["geo_enrichment"] = du.get("geographic_route_intelligence") or {}
        raw["topology_removed"] = (du.get("route_topology_validation") or {}).get(
            "removed_routes"
        ) or []
        results[i] = _check(title, raw, expect, reject)

    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for r in results:
        counts[r["status"]] += 1

    print("=" * 92)
    print("GEOGRAPHY + ROUTE INTELLIGENCE — 10 QUERY REPORT")
    print(f"SUMMARY: PASS={counts['PASS']} PARTIAL={counts['PARTIAL']} FAIL={counts['FAIL']}")
    print("=" * 92)

    for r in results:
        print()
        print(f"## {r['title']} [{r['status']}]")
        print(f"Routes ({len(r['routes'])}): {'; '.join(r['routes'][:8])}")
        if len(r["routes"]) > 8:
            print(f"  ... +{len(r['routes']) - 8} more")
        print(f"Decomposition: {r['decomposition']} | proof: {r['proof']} | suppressed: {r['suppressed']}")
        if r["geo_regions"]:
            print(f"Geo regions: {', '.join(r['geo_regions'])}")
        if r["geo_added"]:
            print(f"Geo added: {'; '.join(r['geo_added'][:5])}")
        if r["geo_rebalanced"]:
            print(f"Geo rebalanced: {'; '.join(r['geo_rebalanced'])}")
        if r["dir_swapped"]:
            print(f"Direction swapped: {'; '.join(r['dir_swapped'])}")
        if r["dir_removed"]:
            print(f"Direction removed: {'; '.join(r['dir_removed'])}")
        if r["topology_removed"]:
            print(f"Topology removed: {'; '.join(r['topology_removed'])}")
        if r["expect_hits"]:
            print(f"Expected present: {', '.join(r['expect_hits'])}")
        if r["expect_miss"]:
            print(f"Expected missing: {', '.join(r['expect_miss'])}")
        if r["reject_hits"]:
            print(f"REJECTED PRESENT: {', '.join(r['reject_hits'])}")
        print("Segments:")
        for seg in r["segments"][:4]:
            print(f"  - {seg}")

    out_path = "evals/geo_intel_10_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": counts, "results": results}, f, indent=2)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

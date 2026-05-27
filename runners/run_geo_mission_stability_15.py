"""Geo + Mission Stability — 15 critical unused queries.

Validates the Geographic + Route Intelligence stabilization layer output:
`data_used["mission_graph_stabilized"]`

Checks (strict, deterministic):
- No node loss for named cities/regions in the query (restored if missing)
- No hub collapse (multiple hubs remain multiple hub clusters)
- No continuation hub dominance (Dubai/Doha/Singapore/Frankfurt never appear as hubs)
- No implicit re-anchoring (invalid_reanchors_detected must be empty)
- Directionality preserved for explicit arrow edges (A -> B must exist; B -> A must not)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import build_mission_understanding
from services.mission.pre_ranking_representation import apply_pre_ranking_representation


CONTINUATION_HUBS = {"dubai", "doha", "singapore", "frankfurt"}


QUERIES: List[Tuple[str, str, Dict[str, Any]]] = [
    (
        "Q1 Arctic + multi-hub stress (Nunavut anchor)",
        "We operate between Nunavut gravel strips, Calgary oil fields, and London executive HQ, "
        "with occasional executive flights to Frankfurt and Paris. Winter reliability is critical. "
        "What is the correct mission structure?",
        {"expect_nodes": ["nunavut", "calgary", "london", "frankfurt", "paris"]},
    ),
    (
        "Q2 Multi-continent hub conflict (NY / LA / Miami)",
        "We run flights from New York, Los Angeles, and Miami, connecting to London, Tokyo, and "
        "São Paulo, with variable routing through Dubai and Singapore. How should the route graph "
        "be structured without collapsing hubs?",
        {"expect_nodes": ["new york", "los angeles", "miami", "london", "tokyo", "sao paulo", "dubai", "singapore"], "expect_hubs": ["new york", "los angeles", "miami"]},
    ),
    (
        "Q3 Industrial Africa + Europe + Middle East continuation",
        "We operate Lagos offshore rigs, Houston oil logistics, and executive travel between Zurich, "
        "Frankfurt, and Dubai. What is the correct geographic structure?",
        {"expect_nodes": ["lagos", "houston", "zurich", "frankfurt", "dubai"]},
    ),
    (
        "Q4 Australia mining + Asia + Europe (Perth anchor)",
        "We run mining operations in Perth and Pilbara, move teams to Singapore and Tokyo, and "
        "occasionally route executives through Frankfurt and London. How should this be represented?",
        {"expect_nodes": ["perth", "pilbara", "singapore", "tokyo", "frankfurt", "london"], "expect_hubs": ["perth"]},
    ),
    (
        "Q5 Arctic Canada + US East Coast + Europe mix",
        "We operate in Yellowknife, Nunavut, Calgary, plus executive HQ in New York, with seasonal "
        "flights to London and Paris. What does the mission graph look like?",
        {"expect_nodes": ["yellowknife", "nunavut", "calgary", "new york", "london", "paris"], "expect_hubs": ["new york", "calgary", "yellowknife"]},
    ),
    (
        "Q6 Caribbean + ME continuation trap",
        "We fly between Miami and Caribbean islands, but executives also travel between Miami, Dubai, "
        "and Riyadh, with occasional Europe connections. How should hubs and continuations be treated?",
        {"expect_nodes": ["miami", "caribbean", "dubai", "riyadh"]},
    ),
    (
        "Q7 South America + Africa + Europe tri-domain",
        "We operate between São Paulo, Lagos, and Frankfurt, with additional executive travel between "
        "Dubai, London, and Zurich. What is the correct structure?",
        {"expect_nodes": ["sao paulo", "lagos", "frankfurt", "dubai", "london", "zurich"]},
    ),
    (
        "Q8 Asia-Pacific + US West Coast + ski domain conflict",
        "We operate Los Angeles, San Francisco, Seattle, plus flights to Tokyo, Seoul, Singapore, "
        "and seasonal ski operations in Aspen, Banff, and Jackson Hole. How should this system be structured?",
        {"expect_nodes": ["los angeles", "san francisco", "seattle", "tokyo", "seoul", "singapore", "aspen", "banff", "jackson hole"], "expect_hubs": ["los angeles", "san francisco", "seattle"]},
    ),
    (
        "Q9 Hub inversion trap (continuation ambiguity)",
        "We operate between New York and Singapore, sometimes via Dubai, while also running domestic "
        "executive flights between New York, Boston, and Chicago. What is the correct interpretation of hubs vs continuation?",
        {"expect_nodes": ["new york", "singapore", "dubai", "boston", "chicago"], "expect_hubs": ["new york"]},
    ),
    (
        "Q10 Industrial Texas + West Africa + Europe",
        "We operate Permian Basin (Texas), West African offshore rigs, and executive HQ in Houston, "
        "Paris, and Zurich. How should this be structured geographically?",
        {"expect_nodes": ["permian", "west africa", "houston", "paris", "zurich"], "expect_hubs": ["houston"]},
    ),
    (
        "Q11 Arctic + industrial + Asia-Pacific overload",
        "We operate across Nunavut gravel strips, Houston oil fields, and flights to Tokyo, Singapore, "
        "and London, with winter dispatch issues in Canada. What is the mission structure?",
        {"expect_nodes": ["nunavut", "houston", "tokyo", "singapore", "london"], "expect_hubs": ["houston"]},
    ),
    (
        "Q12 Multi-hub collapse test (5+ hubs)",
        "We operate in Miami, New York, Houston, Los Angeles, and London, with destinations including "
        "Dubai, Singapore, Frankfurt, Tokyo, and São Paulo. How should this be represented without hub collapse?",
        {"expect_nodes": ["miami", "new york", "houston", "los angeles", "london", "dubai", "singapore", "frankfurt", "tokyo", "sao paulo"], "expect_hubs": ["miami", "new york", "houston", "los angeles", "london"]},
    ),
    (
        "Q13 Directionality stress test (reversal trap)",
        "We operate: Houston → Lagos; Frankfurt → Houston; New York → London; London → Dubai. "
        "How should directionality be preserved?",
        {
            "expect_nodes": ["houston", "lagos", "frankfurt", "new york", "london", "dubai"],
            "explicit_edges": [("houston", "lagos"), ("frankfurt", "houston"), ("new york", "london"), ("london", "dubai")],
        },
    ),
    (
        "Q14 Node loss stress test (remote geography)",
        "We operate in Nunavut, Yellowknife, Pilbara, Perth, Lagos, and Frankfurt, with executive HQ in "
        "London and New York. What is the full mission graph?",
        {"expect_nodes": ["nunavut", "yellowknife", "pilbara", "perth", "lagos", "frankfurt", "london", "new york"]},
    ),
    (
        "Q15 Extreme mixed-domain final boss",
        "We operate: Arctic gravel strips (Nunavut) / Texas oil fields / Miami Caribbean operations / "
        "London executive HQ / Singapore / Tokyo Asia-Pacific routes / occasional Dubai continuation routing. "
        "How should this system be structured without collapsing into a single hub?",
        {"expect_nodes": ["nunavut", "miami", "caribbean", "london", "singapore", "tokyo", "dubai", "houston"], "expect_hubs_min": 2},
    ),
]


def _blob_nodes(g: Dict[str, Any]) -> str:
    return " ".join(str(n) for n in (g.get("nodes") or [])).lower()


def _edges_set(g: Dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for e in (g.get("edges") or []):
        if not isinstance(e, dict):
            continue
        o = str(e.get("origin") or "").strip()
        d = str(e.get("destination") or "").strip()
        if o and d:
            out.add((o.lower(), d.lower()))
    return out


def _hub_list(g: Dict[str, Any]) -> List[str]:
    hubs: List[str] = []
    for c in (g.get("hub_clusters") or []):
        if isinstance(c, dict) and c.get("hub"):
            hubs.append(str(c["hub"]).lower())
    return hubs


def _run_one(title: str, q: str, checks: Dict[str, Any]) -> Dict[str, Any]:
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    du: Dict[str, Any] = {}
    pkt = build_mission_understanding(q, profile, mission, broker_memory=None, history=None, data_used=du)
    profile, mission, pkt = apply_pre_ranking_representation(q, profile, mission, pkt, data_used=du)
    g = du.get("mission_graph_stabilized") or {}

    nodes_blob = _blob_nodes(g)
    edges = _edges_set(g)
    hubs = _hub_list(g)

    failures: List[str] = []

    # Re-anchoring must never occur
    if (g.get("invalid_reanchors_detected") or []):
        failures.append("invalid_reanchors_detected")

    # Continuation hubs must never be hubs
    for h in hubs:
        if h in CONTINUATION_HUBS:
            failures.append(f"continuation_hub_as_hub:{h}")

    # Node preservation: all expected tokens must appear in nodes list
    missing_nodes: List[str] = []
    for token in checks.get("expect_nodes") or []:
        if token.lower() not in nodes_blob:
            missing_nodes.append(token)
    if missing_nodes:
        failures.append(f"missing_nodes:{', '.join(missing_nodes)}")

    # Multi-hub preservation: ensure expected hubs are present as independent hubs
    expected_hubs = [h.lower() for h in (checks.get("expect_hubs") or [])]
    if expected_hubs:
        for h in expected_hubs:
            if h not in hubs:
                failures.append(f"missing_hub_cluster:{h}")

    if checks.get("expect_hubs_min") is not None:
        try:
            min_h = int(checks["expect_hubs_min"])
            if len(hubs) < min_h:
                failures.append(f"hub_clusters_too_few:{len(hubs)}<{min_h}")
        except Exception:
            pass

    # Explicit directionality edges must exist and not be reversed
    for o, d in (checks.get("explicit_edges") or []):
        if (o, d) not in edges:
            failures.append(f"missing_explicit_edge:{o}->{d}")
        if (d, o) in edges:
            failures.append(f"reversed_edge_present:{d}->{o}")

    status = "PASS" if not failures else "FAIL"
    return {
        "title": title,
        "status": status,
        "hub_count": len(hubs),
        "hubs": hubs[:8],
        "node_count": len(g.get("nodes") or []),
        "edge_count": len(g.get("edges") or []),
        "failures": failures,
        "dropped_nodes_restored": g.get("dropped_nodes_restored") or [],
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    counts = {"PASS": 0, "FAIL": 0}

    for title, q, checks in QUERIES:
        r = _run_one(title, q, checks)
        results.append(r)
        counts[r["status"]] += 1

    print("=" * 92)
    print("GEO + MISSION STABILITY — 15 CRITICAL UNUSED TEST REPORT")
    print(f"SUMMARY: PASS={counts['PASS']} FAIL={counts['FAIL']}")
    print("=" * 92)
    for r in results:
        print()
        print(f"## {r['title']} [{r['status']}]")
        print(f"Nodes={r['node_count']} Edges={r['edge_count']} HubClusters={r['hub_count']} Hubs={', '.join(r['hubs'])}")
        if r["dropped_nodes_restored"]:
            print(f"Dropped nodes restored: {', '.join(r['dropped_nodes_restored'])}")
        if r["failures"]:
            print(f"FAIL reasons: {', '.join(r['failures'])}")

    out = _BACKEND / "evals" / "geo_mission_stability_15_results.json"
    out.write_text(json.dumps({"summary": counts, "results": results}, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()


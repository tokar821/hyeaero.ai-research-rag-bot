from __future__ import annotations

"""
Ad-hoc Phase 2 validation runner for user-provided queries.

Prints:
  - structural verdict + suppression policy
  - rendered segments + segment authority (type/confidence)
  - top of kernel synthesis block
"""

from typing import Dict, List, Tuple

from services.consultant.mission_state import MissionState
from services.mission.mission_authority_kernel import (
    build_mission_authority_kernel,
    render_kernel_aircraft_section,
    render_kernel_synthesis,
)
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    attach_mission_understanding,
    build_mission_understanding,
)
from services.mission.pre_ranking_representation import apply_pre_ranking_representation


def _safe_recs():
    try:
        from services.consultant.recommendation_engine import AircraftRecommendation

        def _rec(model: str, category: str = "ultra-long"):
            return AircraftRecommendation(
                model=model,
                category=category,
                total_score=0.8,
                confidence=0.7,
                rank=1,
                fit="Strong fit",
                avoid=False,
            )

        return [_rec("Global 7500"), _rec("G650ER"), _rec("Falcon 8X")]
    except Exception:
        return []


QUERIES: List[Tuple[str, str]] = [
    (
        "Northeast-Florida + founder Singapore",
        "Most of our flying is between the Northeast corridor and Florida with 4 executives, but the founder "
        "insists on nonstop Singapore twice monthly. We previously owned a large-cabin jet that sat idle domestically. "
        "Is a single-aircraft strategy operationally rational?",
    ),
    (
        "Perth mining + Singapore + London",
        "We move mining engineers between Perth, remote Australian extraction strips, and Singapore, while leadership "
        "also flies quarterly nonstop to London. Reliability into rough field conditions matters more than cabin luxury.",
    ),
    (
        "LA ski triangle + Tokyo Seoul",
        "Our company regularly flies between Los Angeles, Salt Lake City, Jackson Hole, and Aspen during winter, "
        "but we also have recurring Tokyo and Seoul executive missions. The previous aircraft struggled badly "
        "in ski-season operations.",
    ),
    (
        "South Florida Caribbean + Riyadh",
        "We operate short Caribbean island hops from South Florida, but ownership also wants nonstop Riyadh "
        "capability several times a year. Dispatch reliability and runway access matter more than prestige branding.",
    ),
    (
        "Variable bankers Chicago hub network",
        "We move 3 to 16 bankers between Chicago, Frankfurt, Dubai, and São Paulo, often carrying presentation "
        "equipment and technical cargo. Passenger loads vary dramatically week to week.",
    ),
    (
        "Texas desert drilling + Paris Geneva",
        "Most utilization is Texas energy-sector flying into remote desert drilling sites, but executives also "
        "require nonstop Paris and Geneva capability. We've had repeated winter dispatch problems with larger jets.",
    ),
    (
        "California Nevada + Doha Abu Dhabi",
        "The family office mostly flies short California and Nevada trips with 2-5 passengers, but the principal "
        "wants nonstop Doha and Abu Dhabi capability without operating multiple aircraft if possible.",
    ),
    (
        "Houston Lagos West Africa mining",
        "We operate between Houston, Lagos, and remote West African mining strips, while executives frequently "
        "continue onward to Zurich and Frankfurt. The current aircraft performs poorly on difficult field conditions.",
    ),
    (
        "NY-London + Colorado Utah ski",
        "Our leadership team flies New York to London constantly, but ski-season operations into Colorado and Utah "
        "mountain airports create major operational disruptions every winter. What structure actually fits?",
    ),
    (
        "One flagship multi-domain failure",
        "We previously tried operating one flagship jet for Los Angeles, Singapore, Aspen, Caribbean islands, "
        "and occasional Dubai missions. The aircraft became operationally compromised across nearly every domain. "
        "How should we rethink the network?",
    ),
]


def run_one(title: str, query: str) -> Dict:
    data_used: Dict = {}
    profile = extract_mission(query)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)

    pkt = build_mission_understanding(query, profile, mission, broker_memory=None)
    attach_mission_understanding(data_used, pkt)

    profile, mission, pkt = apply_pre_ranking_representation(
        query, profile, mission, pkt, data_used=data_used
    )
    attach_mission_understanding(data_used, pkt)

    kernel = build_mission_authority_kernel(
        mission,
        pkt,
        profile,
        data_used=data_used,
        recommendations=_safe_recs(),
        query=query,
    )
    synth = render_kernel_synthesis(kernel)
    aircraft = render_kernel_aircraft_section(kernel, [])

    out = {
        "title": title,
        "routes": list(profile.route_labels() or mission.routes or []),
        "geo_enrichment": data_used.get("geographic_route_intelligence") or {},
        "topology_removed": (data_used.get("route_topology_validation") or {}).get(
            "removed_routes"
        )
        or [],
        "structural_decomposition": bool(kernel.structural_decomposition),
        "structure_resolution": data_used.get("mission_structure_resolution") or {},
        "suppression": data_used.get("recommendation_suppression") or {},
        "segments": [],
        "synthesis_preview": "\n".join((synth or "").splitlines()[:24]),
        "aircraft_section": aircraft,
    }
    for seg in kernel.segments:
        auth = (
            (seg.constraints or {}).get("segment_authority")
            if isinstance(seg.constraints, dict)
            else None
        )
        out["segments"].append(
            {
                "label": seg.label,
                "kind": getattr(seg.kind, "value", str(seg.kind)),
                "routes": list(seg.route_labels or []),
                "authority_type": (auth or {}).get("authority_type")
                if isinstance(auth, dict)
                else None,
                "confidence": (auth or {}).get("confidence") if isinstance(auth, dict) else None,
            }
        )
    return out


def main() -> None:
    for title, query in QUERIES:
        r = run_one(title, query)
        sr = r["structure_resolution"]
        sup = r["suppression"]
        print("=" * 92)
        print(r["title"])
        print(
            "structural_decomposition:",
            r["structural_decomposition"],
            "| decomposition_required:",
            sr.get("decomposition_required"),
            "| proof:",
            sr.get("proof_source"),
        )
        print(
            "suppression:",
            sup.get("suppress_aircraft_specificity"),
            "| reason:",
            sup.get("reason"),
        )
        print("routes:", "; ".join(r.get("routes") or []) or "(none)")
        geo = r.get("geo_enrichment") or {}
        if geo.get("routes_added"):
            print("geo_added:", "; ".join(geo.get("routes_added") or []))
        if geo.get("routes_rebalanced"):
            print("geo_rebalanced:", "; ".join(geo.get("routes_rebalanced") or []))
        if r.get("topology_removed"):
            print("topology_removed:", "; ".join(r.get("topology_removed") or []))
        print("segments:")
        for s in r["segments"]:
            routes = "; ".join((s.get("routes") or [])[:2]) or "(none)"
            print(
                " -",
                s.get("label"),
                "|",
                s.get("kind"),
                "|",
                routes,
                "| auth=",
                s.get("authority_type"),
                "conf=",
                s.get("confidence"),
            )
        # Keep output compact for terminal logs
        print("SYNTHESIS_HEADER:")
        preview_lines = (r["synthesis_preview"] or "").splitlines()
        for ln in preview_lines[:10]:
            print(ln)
        if r["aircraft_section"]:
            print("AIRCRAFT_SECTION_HEADER:")
            for ln in (r["aircraft_section"] or "").splitlines()[:8]:
                print(ln)


if __name__ == "__main__":
    main()


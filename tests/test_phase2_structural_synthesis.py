"""Phase 2 — segment authority, structural verdict, recommendation suppression."""

from __future__ import annotations

import re

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_authority_kernel import (
    build_mission_authority_kernel,
    render_kernel_synthesis,
)
from services.mission.mission_graph import SegmentKind, build_mission_graph
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.phase2_structural_synthesis import apply_phase2_structural_synthesis
from services.mission.segment_authority import build_segment_authority


def _profile(mission: MissionState):
    from services.mission.models import MissionProfile, Route

    p = MissionProfile(passengers=mission.passenger_count)
    for lbl in mission.routes or []:
        r = Route.from_label(lbl)
        if r:
            p.routes.append(r)
    return p


def _rec(model: str) -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category="ultra-long",
        total_score=0.8,
        confidence=0.7,
        rank=1,
        fit="Strong fit",
        avoid=False,
    )


def test_no_ghost_segments_nyc_domestic_dubai():
    """NYC domestic + Dubai — no Mountain, Industrial, or Pacific ULR ghosts."""
    mission = MissionState(
        routes=[
            "New York -> Chicago",
            "Chicago -> San Francisco",
            "New York -> Dubai",
        ],
        passenger_count=4,
    )
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Middle East ULR continuation band",
            "Multi-leg ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
            "Domestic field-access executive band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    prof = _profile(mission)
    graph = build_mission_graph(pkt, prof, mission, query="CEO flies NYC to Dubai nonstop; company uses domestic hops.")
    graph, _, authorities, _ = apply_phase2_structural_synthesis(
        graph, pkt, prof, mission, query="CEO flies NYC to Dubai nonstop; company uses domestic hops."
    )
    kinds = {s.kind for s in graph.segments}
    assert SegmentKind.MOUNTAIN_FIELD not in kinds
    assert SegmentKind.INDUSTRIAL_FIELD not in kinds
    assert SegmentKind.PACIFIC_ULR not in kinds
    for auth in authorities:
        if auth.segment_name.lower().find("pacific") >= 0:
            assert auth.renderable is False
    synthesis = render_kernel_synthesis(
        build_mission_authority_kernel(mission, pkt, prof, recommendations=[], query="")
    )
    assert "Mountain Field" not in synthesis
    assert "Pacific Ulr" not in synthesis.lower() or "pacific ulr" not in synthesis.lower()


def test_structural_consistency_no_single_domain_when_decomposed():
    mission = MissionState(
        routes=["Los Angeles -> Aspen", "Los Angeles -> Tokyo"],
        passenger_count=6,
    )
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Multi-leg ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
    )
    prof = _profile(mission)
    du: dict = {}
    kernel = build_mission_authority_kernel(
        mission,
        pkt,
        prof,
        data_used=du,
        recommendations=[_rec("Global 7500")],
    )
    assert kernel.structural_decomposition
    synthesis = render_kernel_synthesis(kernel)
    assert "single operational domain" not in synthesis.lower()
    resolution = du.get("mission_structure_resolution") or {}
    assert resolution.get("decomposition_required") is True


def test_every_rendered_segment_has_route_authority():
    mission = MissionState(
        routes=["New York -> Dubai", "New York -> Boston"],
        passenger_count=3,
    )
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=["Middle East ULR continuation band"],
    )
    prof = _profile(mission)
    graph = build_mission_graph(pkt, prof, mission)
    graph, _, authorities, _ = apply_phase2_structural_synthesis(graph, pkt, prof, mission)
    for seg in graph.segments:
        auth = build_segment_authority(seg)
        assert auth.renderable
        assert auth.source_routes or auth.source_constraints
        assert len(seg.route_labels) > 0


def test_recommendation_suppression_blocks_generic_dump():
    mission = MissionState(
        routes=["Los Angeles -> Aspen", "Aspen -> Tokyo"],
        passenger_count=4,
    )
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"incompatible_mission_bands": True},
        fallback_operational_band=[
            "Multi-leg ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
        ],
    )
    prof = _profile(mission)
    kernel = build_mission_authority_kernel(
        mission,
        pkt,
        prof,
        recommendations=[_rec("Global 7500"), _rec("G650ER"), _rec("Falcon 8X")],
    )
    from services.mission.mission_authority_kernel import render_kernel_aircraft_section

    section = render_kernel_aircraft_section(kernel, [])
    assert "Global 7500" not in section
    assert not re.search(r"\bG650(?:ER)?\b", section)
    assert "Falcon 8X" not in section


def test_governance_conflict_renders_in_synthesis():
    mission = MissionState(
        routes=["New York -> Dubai", "New York -> Chicago", "Chicago -> San Francisco"],
        passenger_count=4,
    )
    pkt = MissionUnderstandingPacket(
        operational_synthesis=(
            "CEO nonstop ULR mandate vs dominant short domestic utilization — "
            "portfolio governance required."
        ),
        fallback_operational_band=[
            "Middle East ULR continuation band",
            "Transatlantic super-mid / heavy-cabin executive band",
        ],
        inferred_constraints={
            "ceo_ulr_mandate": True,
            "domestic_utilization_dominant": True,
            "founder_company_asymmetry": True,
        },
    )
    prof = _profile(mission)
    kernel = build_mission_authority_kernel(mission, pkt, prof, recommendations=[])
    synthesis = render_kernel_synthesis(kernel)
    assert "governance" in synthesis.lower() or "utilization" in synthesis.lower()

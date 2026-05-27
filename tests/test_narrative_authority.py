"""Mission representation + narrative authority regressions."""

from __future__ import annotations

import re

from services.consultant.broker_advisory_layer import format_broker_advisory_response
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_graph import build_mission_graph, SegmentKind
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    attach_mission_understanding,
    bands_are_incompatible,
    needs_portfolio_synthesis,
)
from services.mission.narrative_authority import (
    SYNTHESIS_BLOCK_MARKER,
    build_narrative_authority_payload,
    compose_authoritative_advisory,
    dedupe_advisory_body,
    dedupe_recommendation_models,
    enforce_narrative_authority_in_answer,
    render_narrative_authority,
)
from services.mission.structural_decomposition import needs_structural_decomposition


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


def test_exactly_one_synthesis_marker_in_render():
    mission = MissionState(passenger_count=8, routes=["NYC -> London", "London -> Dubai"])
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Peak planning centers on Middle East nonstop continuation.",
        fallback_operational_band=[
            "Transatlantic super-mid / heavy-cabin executive band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    payload = build_narrative_authority_payload(mission, pkt, query="CEO Dubai")
    text = render_narrative_authority(payload)
    assert text.count(SYNTHESIS_BLOCK_MARKER) == 1
    assert "Operational segments:" in text
    assert "Operational bands:" not in text


def test_no_duplicate_synthesis_on_compose():
    mission = MissionState(passenger_count=10, routes=["TEB -> London", "TEB -> Aspen"])
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Incompatible bands — portfolio required.",
        fallback_operational_band=[
            "Transatlantic ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
    )
    du = attach_mission_understanding({}, pkt)
    body = compose_authoritative_advisory(
        mission,
        pkt,
        [_rec("Gulfstream G650ER"), _rec("Pilatus PC-24")],
        query="Aspen and London",
        data_used=du,
    )
    assert body.count(SYNTHESIS_BLOCK_MARKER) == 1
    assert body.count("Fleet Structure:") <= 1


def test_ulr_continuation_band_scoped_to_segment():
    mission = MissionState(
        passenger_count=8,
        routes=["Dallas -> New York", "New York -> London", "London -> Dubai"],
    )
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic super-mid / heavy-cabin executive band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    graph = build_mission_graph(pkt, _profile(mission), mission)
    cont_seg = next(
        (s for s in graph.segments if s.kind == SegmentKind.ULR_CONTINUATION),
        None,
    )
    assert cont_seg is not None
    assert "continuation" in cont_seg.operational_band.lower()
    assert any("Dubai" in r or "London" in r for r in cont_seg.route_labels)


def _profile(mission: MissionState):
    from services.mission.models import MissionProfile, Route

    p = MissionProfile(passengers=mission.passenger_count)
    for lbl in mission.routes or []:
        r = Route.from_label(lbl)
        if r:
            p.routes.append(r)
    return p


def test_aspen_london_structural_decomposition():
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
    )
    proof = needs_structural_decomposition(pkt)
    assert proof.required
    assert needs_portfolio_synthesis("", pkt)


def test_domestic_europe_not_structural_decomposition():
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic super-mid / heavy-cabin executive band",
            "Multi-leg ultra-long-range executive band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    assert not bands_are_incompatible(pkt.fallback_operational_band)
    proof = needs_structural_decomposition(pkt)
    assert not proof.required
    assert not needs_portfolio_synthesis("", pkt)


def test_dedupe_recommendation_models():
    recs = [_rec("G650ER"), _rec("G650ER"), _rec("Global 7500")]
    out = dedupe_recommendation_models(recs)
    assert len(out) == 2


def test_llm_merge_cannot_drop_authority_block():
    mission = MissionState(passenger_count=8, routes=["NYC -> London"])
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Transatlantic executive with modest utilization.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
    )
    payload = build_narrative_authority_payload(mission, pkt)
    llm_only = "Aircraft Options:\n\n* Gulfstream G280 — good fit.\n\nVerdict:\n\n* VIABLE"
    enforced = enforce_narrative_authority_in_answer(llm_only, payload)
    assert SYNTHESIS_BLOCK_MARKER in enforced
    if "Aircraft Options" in enforced:
        assert enforced.index(SYNTHESIS_BLOCK_MARKER) < enforced.index("Aircraft Options")


def test_ranked_path_single_authority_block():
    mission = MissionState(passenger_count=8, routes=["NYC -> London"])
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Executive transatlantic band.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
        recommend_aircraft=True,
    )
    du = attach_mission_understanding({}, pkt)
    body = format_broker_advisory_response(
        mission,
        [_rec("Gulfstream G280")],
        query="recommend",
        data_used=du,
    )
    assert body.count(SYNTHESIS_BLOCK_MARKER) == 1
    assert du.get("narrative_authority_built") == 1


def test_dedupe_strips_duplicate_markers():
    dup = (
        f"{SYNTHESIS_BLOCK_MARKER}\n\nMission Fit:\n\n* Route: A\n\n"
        f"{SYNTHESIS_BLOCK_MARKER}\n\nMission Fit:\n\n* Route: B\n\nAircraft Options:"
    )
    out = dedupe_advisory_body(dup)
    assert out.count(SYNTHESIS_BLOCK_MARKER) == 1

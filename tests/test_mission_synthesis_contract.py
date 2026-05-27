"""Regression — immutable synthesis survives ranked and LLM merge paths."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.broker_advisory_layer import format_broker_advisory_response
from services.mission.mission_synthesis_contract import (
    SYNTHESIS_BLOCK_MARKER,
    build_immutable_synthesis_block,
    enforce_immutable_synthesis_in_answer,
    synthesis_present_in_answer,
)
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    attach_mission_understanding,
    format_understanding_first_advisory,
)
from services.mission.mission_ranking_projection import (
    build_ranking_mission_snapshot,
    is_segmented_mission,
)


def _ceo_continuation_packet() -> MissionUnderstandingPacket:
    pkt = MissionUnderstandingPacket(
        operational_synthesis=(
            "US hubs with transatlantic executive travel; CEO continuation to Middle East "
            "nonstop is the peak planning leg — not the frequent domestic shuttle pattern."
        ),
        fallback_operational_band=[
            "Transatlantic super-mid / heavy-cabin executive band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
        recommend_aircraft=True,
        overall_confidence=0.85,
    )
    return pkt


def test_immutable_synthesis_block_contains_marker_and_bands():
    mission = MissionState(
        passenger_count=8,
        routes=["Dallas -> New York", "New York -> London", "London -> Dubai"],
    )
    pkt = _ceo_continuation_packet()
    block = build_immutable_synthesis_block(mission, pkt, query="CEO continues to Dubai")
    assert SYNTHESIS_BLOCK_MARKER in block
    assert "Middle East ULR continuation" in block
    assert "peak planning" in block.lower() or "Operational read" in block


def test_ranked_broker_response_preserves_synthesis_before_options():
    mission = MissionState(passenger_count=8, routes=["NYC -> London", "London -> Dubai"])
    pkt = _ceo_continuation_packet()
    du = attach_mission_understanding({}, pkt)
    rec = AircraftRecommendation(
        model="Gulfstream G650ER",
        category="ultra-long",
        total_score=0.82,
        confidence=0.75,
        rank=1,
        fit="Strong fit",
        avoid=False,
    )
    body = format_broker_advisory_response(
        mission,
        [rec],
        query="structure for CEO Dubai continuation",
        data_used=du,
    )
    assert SYNTHESIS_BLOCK_MARKER in body
    assert body.index(SYNTHESIS_BLOCK_MARKER) < body.index("Aircraft Options")
    assert "Middle East ULR continuation" in body


def test_llm_merge_reinjects_synthesis_when_missing():
    mission = MissionState(passenger_count=8, routes=["NYC -> London"])
    pkt = _ceo_continuation_packet()
    generic_llm = (
        "Aircraft Options:\n\n* Gulfstream G650ER — good fit for your mission.\n\n"
        "Verdict:\n\n* VIABLE WITH COMPROMISES: Gulfstream G650ER"
    )
    enforced = enforce_immutable_synthesis_in_answer(
        generic_llm,
        mission,
        pkt,
        query="CEO Dubai",
    )
    assert synthesis_present_in_answer(enforced, pkt)
    assert SYNTHESIS_BLOCK_MARKER in enforced
    if "Aircraft Options" in enforced:
        assert enforced.index(SYNTHESIS_BLOCK_MARKER) < enforced.index("Aircraft Options")


def test_segmented_mission_suppresses_global_mountain_for_ranking():
    mission = MissionState(
        passenger_count=10,
        routes=["TEB -> London", "TEB -> Aspen"],
        mountain_airport_requirement=True,
    )
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
    )
    assert is_segmented_mission(pkt)
    rank_m, _, trace = build_ranking_mission_snapshot(mission, pkt, None)
    assert trace.segment_isolated
    assert rank_m.mountain_airport_requirement is False
    assert "mountain_airport_requirement" in trace.suppressed_global_flags


def test_understanding_first_includes_synthesis_marker():
    mission = MissionState(passenger_count=8, routes=["Dallas -> London"])
    pkt = _ceo_continuation_packet()
    pkt.recommend_aircraft = False
    text = format_understanding_first_advisory(mission, pkt, query="continuation")
    assert SYNTHESIS_BLOCK_MARKER in text
    assert "Operational segments" in text or "Operational read" in text

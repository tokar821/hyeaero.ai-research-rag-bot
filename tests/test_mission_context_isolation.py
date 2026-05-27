"""Mission context isolation — no prior corridor bleed on pivot."""

from __future__ import annotations

from services.mission.mission_context_reconciliation import assess_mission_continuity
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import build_mission_understanding
from services.mission.mission_operational_graph import (
    MissionOperationalGraph,
    apply_broker_memory_to_packet,
    merge_graphs,
)
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.consultant.mission_state import MissionState
from services.session.broker_memory import BrokerMemory, update_broker_memory_from_understanding


def test_miami_then_nyc_tokyo_london_pivot():
    """Prior Caribbean mission must not appear in NYC/Tokyo/London synthesis."""
    mem = BrokerMemory()
    miami_profile = extract_mission(
        "Miami Caribbean 8 passengers runway flexibility over luxury"
    )
    miami_mission = MissionState(
        passenger_count=8, routes=miami_profile.route_labels()
    )
    miami_pkt = build_mission_understanding(
        "Miami Caribbean 8 passengers runway flexibility",
        miami_profile,
        miami_mission,
    )
    mem = update_broker_memory_from_understanding(mem, miami_pkt)
    assert any("caribbean" in b.lower() for b in mem.operational_bands)

    tokyo_profile = extract_mission(
        "NYC to Tokyo and London, 8 passengers, nonstop westbound winter"
    )
    tokyo_mission = MissionState(
        passenger_count=8, routes=tokyo_profile.route_labels()
    )
    cont = assess_mission_continuity(
        "NYC to Tokyo and London, 8 passengers",
        tokyo_profile,
        broker_memory=mem.to_dict(),
        prior_graph=MissionOperationalGraph(
            corridor_type="caribbean_regional",
            operational_bands=["Caribbean executive regional jet band"],
        ),
    )
    assert cont.mission_pivot is True

    pkt = MissionUnderstandingPacket()
    pkt = apply_broker_memory_to_packet(
        pkt, mem.to_dict(), apply_structural=cont.apply_structural_memory
    )
    assert not any("caribbean" in b.lower() for b in (pkt.fallback_operational_band or []))

    tokyo_pkt = build_mission_understanding(
        "NYC to Tokyo and London, 8 passengers, nonstop westbound winter",
        tokyo_profile,
        tokyo_mission,
        broker_memory=mem.to_dict(),
    )
    synth = (tokyo_pkt.operational_synthesis or "").lower()
    bands = " ".join(tokyo_pkt.fallback_operational_band or []).lower()
    assert tokyo_pkt.corridor_type != "caribbean_regional"
    assert "caribbean executive" not in bands
    assert "miami" not in synth


def test_merge_graphs_skips_prior_on_pivot():
    prior = MissionOperationalGraph(
        corridor_type="caribbean_regional",
        operational_bands=["Caribbean executive regional jet band"],
        inferred_flags={"island_ops": True},
    )
    current = MissionOperationalGraph(
        corridor_type="transatlantic_ulr",
        operational_bands=["Transatlantic ultra-long-range executive band"],
    )
    merged = merge_graphs(prior, current, allow_prior_merge=False)
    assert merged.corridor_type == "transatlantic_ulr"
    assert not any("caribbean" in b.lower() for b in merged.operational_bands)


def test_geography_pivot_detection():
    profile = extract_mission("New York to London, 4 passengers twice monthly")
    mem = {
        "operational_bands": ["Caribbean executive regional jet band"],
        "recurring_routes": ["miami -> caribbean"],
    }
    cont = assess_mission_continuity(
        "New York to London, 4 passengers",
        profile,
        broker_memory=mem,
    )
    assert cont.mission_pivot is True
    assert cont.apply_structural_memory is False

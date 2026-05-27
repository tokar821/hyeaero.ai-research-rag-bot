"""Mission operational graph — stabilization and fleet defer helpers."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_operational_graph import (
    MissionOperationalGraph,
    apply_broker_memory_to_packet,
    graph_from_packet,
    merge_graphs,
    should_defer_ranking_to_fleet,
    stabilize_mission_understanding,
)
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile


def test_protected_incompatible_bands_sticky_across_merge():
    prior = MissionOperationalGraph(
        incompatible_bands=True,
        inferred_flags={"incompatible_mission_bands": True},
    )
    current = MissionOperationalGraph(incompatible_bands=False, inferred_flags={})
    merged = merge_graphs(prior, current)
    assert merged.incompatible_bands is True
    assert merged.inferred_flags.get("incompatible_mission_bands") is True


def test_broker_memory_rehydrates_packet():
    pkt = MissionUnderstandingPacket()
    mem = {
        "incompatible_bands": True,
        "fleet_strategy_required": True,
        "operational_bands": ["transatlantic_ulr", "mountain_field"],
    }
    out = apply_broker_memory_to_packet(pkt, mem)
    assert out.inferred_constraints.get("incompatible_mission_bands") is True
    assert "transatlantic_ulr" in (out.fallback_operational_band or [])


def test_stabilize_preserves_prior_graph_when_continuity_valid():
    profile = MissionProfile(passengers=10)
    mission = MissionState(passenger_count=10, routes=["NYC -> London"])
    pkt = MissionUnderstandingPacket(
        corridor_type="multi_hub",
        inferred_constraints={"incompatible_mission_bands": True},
        fallback_operational_band=["Transatlantic ultra-long-range executive band"],
    )
    prior = MissionOperationalGraph(
        operational_bands=["Transatlantic ultra-long-range executive band", "mountain_field"],
        incompatible_bands=True,
        corridor_type="transatlantic_ulr",
    )
    from services.mission.mission_context_reconciliation import MissionContinuityAssessment

    cont = MissionContinuityAssessment(
        continuity_confidence=0.9,
        apply_structural_memory=True,
        apply_posture_memory=True,
    )
    stable, graph = stabilize_mission_understanding(
        pkt,
        query="NYC and London monthly",
        profile=profile,
        mission=mission,
        prior_graph=prior,
        continuity=cont,
    )
    assert graph.incompatible_bands is True
    assert stable.inferred_constraints.get("incompatible_mission_bands") is True
    assert any("transatlantic" in b.lower() for b in (stable.fallback_operational_band or graph.operational_bands))


def test_should_defer_ranking_when_incompatible():
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"incompatible_mission_bands": True},
    )
    assert should_defer_ranking_to_fleet(pkt, {}) is True


def test_graph_from_packet_fleet_flag():
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"incompatible_mission_bands": True},
    )
    g = graph_from_packet(pkt)
    assert g.fleet_strategy_required is True
    assert g.incompatible_bands is True

"""Geographic graph authority — explicit lock and priority rules."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.explicit_route_lock import extract_explicit_routes
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    attach_mission_understanding,
    build_mission_understanding,
)
from services.mission.pre_ranking_representation import apply_pre_ranking_representation


def _run(q: str):
    du = {}
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    pkt = build_mission_understanding(q, profile, mission, broker_memory=None)
    attach_mission_understanding(du, pkt)
    profile, mission, pkt = apply_pre_ranking_representation(
        q, profile, mission, pkt, data_used=du
    )
    graph = du.get("geographic_graph_authority") or {}
    return list(profile.route_labels() or []), graph, du


def test_explicit_la_tokyo_preserved():
    q = (
        "We operate Los Angeles to Tokyo, Singapore to Dubai, and also run Caribbean island "
        "operations out of Miami. How should routing be structured?"
    )
    routes, graph, _ = _run(q)
    blob = " ".join(routes).lower()
    assert "los angeles -> tokyo" in blob
    assert "singapore -> dubai" in blob
    assert "miami -> caribbean" in blob
    assert "miami -> tokyo" not in blob
    assert "tokyo -> caribbean" not in blob
    assert any("los angeles -> tokyo" == x.lower() for x in (graph.get("explicit_route_labels") or []))


def test_eu_executive_not_inverted_under_industrial():
    q = (
        "We run drilling operations in Texas and Nigeria, but executives regularly fly between "
        "Paris, Geneva, and Houston. What does the route graph look like?"
    )
    routes, graph, _ = _run(q)
    blob = " ".join(routes).lower()
    assert "houston -> paris" in blob or "houston -> geneva" in blob
    assert "geneva -> houston" not in blob
    assert "paris" in blob and "geneva" in blob
    assert graph.get("eu_exec_layer")


def test_permian_and_eu_overlay():
    q = (
        "We move equipment from Permian Basin sites in Texas to Northern Africa and also send "
        "executives to Paris and Zurich. What is the correct routing structure?"
    )
    routes, graph, _ = _run(q)
    blob = " ".join(routes).lower()
    assert "permian" in blob or "houston ->" in blob
    assert "paris" in blob and "zurich" in blob
    assert "paris -> permian" not in blob


def test_ny_base_hub_inversion():
    q = (
        "Our base is New York, but we also operate Dubai to London and Singapore to Dubai routes. "
        "How should the system structure this?"
    )
    routes, _, _ = _run(q)
    blob = " ".join(routes).lower()
    assert "new york" in blob
    assert "singapore" in blob and "dubai" in blob and "london" in blob
    assert "dubai -> new york" not in blob

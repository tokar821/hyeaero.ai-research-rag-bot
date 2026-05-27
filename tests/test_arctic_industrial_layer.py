"""Arctic / Northern Canada logistics layer tests."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    attach_mission_understanding,
    build_mission_understanding,
)
from services.mission.pre_ranking_representation import apply_pre_ranking_representation


def _run(q: str):
    data_used = {}
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    pkt = build_mission_understanding(q, profile, mission, broker_memory=None)
    attach_mission_understanding(data_used, pkt)
    profile, mission, pkt = apply_pre_ranking_representation(
        q, profile, mission, pkt, data_used=data_used
    )
    routes = list(profile.route_labels() or mission.routes or [])
    geo = data_used.get("geographic_route_intelligence") or {}
    return routes, geo, data_used


def _blob(routes):
    return " ".join(routes).lower()


def test_q1_arctic_gravel_northern_canada():
    q = (
        "We operate between Houston, Calgary oil fields, and London. Some sites are gravel strips "
        "in Northern Canada. Can a single aircraft realistically handle this network?"
    )
    routes, geo, _ = _run(q)
    blob = _blob(routes)
    assert "houston -> london" in blob or "houston" in blob and "london" in blob
    assert "arctic_industrial_layer" in (geo.get("regions_activated") or [])
    assert "remote gravel" in blob or "yellowknife -> remote" in blob
    assert "northern alberta" in blob or "calgary -> northern" in blob
    assert "houston -> northern alberta" not in blob


def test_q6_yellowknife_calgary_houston():
    q = (
        "We fly between Calgary, Yellowknife, and Houston, but also do regular New York to Chicago "
        "shuttle flights. How should these be connected operationally?"
    )
    routes, geo, _ = _run(q)
    blob = _blob(routes)
    assert "yellowknife" in blob
    assert "calgary" in blob
    assert "houston" in blob
    assert "new york" in blob and "chicago" in blob


def test_nunavut_field_ops_from_yellowknife():
    q = (
        "Our Nunavut field operations require regular logistics from Yellowknife into remote "
        "gravel strips across Northern Canada."
    )
    routes, geo, _ = _run(q)
    blob = _blob(routes)
    assert "arctic_industrial_layer" in (geo.get("regions_activated") or [])
    assert "nunavut field" in blob
    assert "remote gravel" in blob
    assert "yellowknife" in blob
    assert "yellowknife -> nunavut" in blob or "yellowknife -> remote" in blob

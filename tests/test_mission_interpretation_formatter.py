"""Semantic interpretation formatter regression tests."""

from __future__ import annotations

import re

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_authority_kernel import build_mission_authority_kernel
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import build_mission_understanding
from services.mission.mission_interpretation_formatter import (
    format_mission_interpretation,
    is_interpretation_only_query,
)


_AIRCRAFT_LEAK_RE = re.compile(
    r"\b(?:gulfstream|global\s+\d+|falcon\s+\d+|citation|learjet|embraer|phenom|hawker)\b",
    re.I,
)


def test_is_interpretation_only_query_true_for_structure_prompts():
    assert is_interpretation_only_query("What is the mission structure and dominant domains?")
    assert is_interpretation_only_query("How should this be understood semantically?")


def test_is_interpretation_only_query_false_for_recommendations():
    assert not is_interpretation_only_query("Which aircraft should we buy?")
    assert not is_interpretation_only_query("Recommend a jet for this mission.")


def test_formatter_outputs_required_sections_and_verdict_no_aircraft_no_routes():
    q = (
        "We operate Nunavut gravel strips and Houston oil fields, and fly executives to London. "
        "Winter dispatch failures keep happening. What is the mission structure?"
    )
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    du: dict = {}
    pkt = build_mission_understanding(q, profile, mission, data_used=du)
    kernel = build_mission_authority_kernel(
        mission, pkt, recommendations=[], query=q, data_used=du
    )
    out = format_mission_interpretation(mission, pkt, kernel, query=q, data_used=du)

    for header in (
        "Operational Structure",
        "Primary Utilization",
        "Secondary / Continuation Traffic",
        "Operational Domains",
        "Structural Conflicts",
        "Interpretation Verdict",
    ):
        assert header in out

    assert "->" not in out  # no route dumps
    assert not _AIRCRAFT_LEAK_RE.search(out)

    # must include an authoritative verdict line
    assert re.search(r"Interpretation Verdict\s*\n- .+", out) is not None


def test_formatter_avoids_banned_generic_phrases_and_jargon():
    q = "Mission structure classification please; no aircraft yet."
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    du: dict = {}
    pkt = build_mission_understanding(q, profile, mission, data_used=du)
    kernel = build_mission_authority_kernel(
        mission, pkt, recommendations=[], query=q, data_used=du
    )
    out = format_mission_interpretation(mission, pkt, kernel, query=q, data_used=du).lower()
    for banned in (
        "to optimize your operations",
        "here's a breakdown",
        "you should consider",
        "hub-and-spoke",
        "point-to-point",
    ):
        assert banned not in out


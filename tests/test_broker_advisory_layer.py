"""Broker advisory layer — tone, fit verdicts, LLM context boundaries."""

from __future__ import annotations

import re

from services.consultant.broker_advisory_layer import (
    BROKER_FORBIDDEN_PHRASES,
    build_broker_advisory_context,
    build_broker_llm_context_block,
    format_broker_advisory_response,
    sanitize_broker_prose,
)
from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_formatter import format_consultant_response


def test_broker_response_ends_with_fit_verdicts():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_broker_advisory_response(mission, recs)
    assert "Mission Fit:" in text
    assert "PRIMARY RECOMMENDATION:" in text


def test_broker_forbidden_phrases_absent():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="8 pax LA to Miami nonstop recommend",
    )
    lower = text.lower()
    for phrase in (
        "mission profile",
        "mission score",
        "confidence score",
        "worth considering",
        "if priorities shift",
        "balanced capability",
        "mission summary",
    ):
        assert phrase not in lower


def test_max_three_aircraft_in_broker_output():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=6)
    text = format_broker_advisory_response(mission, recs)
    good_line = next(
        ln for ln in text.splitlines() if "PRIMARY RECOMMENDATION:" in ln
    )
    models = [m.strip() for m in good_line.split(":", 1)[1].split(",")]
    cond_lines = [
        ln for ln in text.splitlines() if ln.startswith("VIABLE WITH COMPROMISES:")
    ]
    cond_count = 0
    if cond_lines:
        cond_count = len(cond_lines[0].split(":", 1)[1].split(","))
    assert len(models) + cond_count <= 3


def test_llm_context_has_no_scores():
    mission = build_mission_from_current_turn("6 executives SFO to Tokyo nonstop westbound winter")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    ctx = build_broker_advisory_context(mission, recs)
    block = ctx.to_llm_block()
    assert "VERIFIED MISSION FACTS" in block
    assert "model=" in block
    assert "Mission Fit:" not in block
    assert "total_score" not in block.lower()
    assert not re.search(r"\bconfidence\s*[:=]\s*\d", block, re.I)
    assert not re.search(r"\bmission score\s*[:=]", block, re.I)
    assert not re.search(r"rank\s*[:=]\s*\d", block, re.I)


def test_llm_context_block_from_pipeline_style():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    block = build_broker_llm_context_block(mission, recs)
    assert "VERIFIED MISSION FACTS" in block
    assert "feasible_aircraft_max" in block
    assert "Mission Fit:" not in block
    assert "Aircraft Options:" not in block


def test_transpacific_territory_opening():
    mission = build_mission_from_current_turn(
        "6 executives San Francisco to Tokyo nonstop westbound winter"
    )
    recs = [r for r in rank_aircraft_recommendations(mission, max_results=3) if not r.avoid]
    assert recs
    text = format_broker_advisory_response(mission, recs)
    assert "Mission Fit:" in text
    assert "Verdict:" in text


def test_sanitize_broker_strips_operationally():
    dirty = "Challenger 350 is operationally balanced for your mission profile."
    clean = sanitize_broker_prose(dirty)
    assert "operationally" not in clean.lower()
    assert "mission profile" not in clean.lower()

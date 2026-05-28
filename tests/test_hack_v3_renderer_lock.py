"""HACK v3 — renderer integrity lock tests."""

from __future__ import annotations

import pytest

from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.broker_response_renderer import format_broker_recommendation_response
from services.consultant.mission_state import MissionState
from services.recommendation.hack_v3_renderer_lock import (
    RenderIntegrityError,
    build_formatted_rows,
    freeze_ranked_list,
    render_hack_v3_locked_response,
    render_locked_table,
    verify_render_integrity,
)


def _contract():
    return [
        {
            "aircraft_name": "Citation Latitude",
            "composite_score": 0.6934,
            "eligibility_status": "ELIGIBLE",
            "verdict": "CONDITIONAL FIT",
        },
        {
            "aircraft_name": "Pilatus PC-24",
            "composite_score": 0.8100,
            "eligibility_status": "ELIGIBLE",
            "verdict": "GOOD FIT",
        },
    ]


def test_render_locked_table_verbatim_verdicts():
    frozen, _ = freeze_ranked_list(_contract())
    rows = build_formatted_rows(frozen)
    text = render_locked_table(rows)
    verify_render_integrity(frozen, rows, text)
    assert "## Ranked Aircraft List" in text
    assert "CONDITIONAL FIT" in text
    assert "GOOD FIT" in text
    assert "Why:" not in text
    assert "Mission Interpretation" not in text


def test_render_hack_v3_locked_response_sets_freeze_frame():
    du = {"hack_v2_ranking": _contract()}
    text = render_hack_v3_locked_response(du)
    assert du.get("freeze_frame") is True
    assert du.get("hack_v3_renderer_locked") is True
    assert "Pilatus PC-24" in text
    assert "| 1 | Citation Latitude |" in text
    assert "| 2 | Pilatus PC-24 |" in text


def test_broker_renderer_uses_hack_v3_when_contract_present():
    mission = MissionState(routes=["A -> B"], passenger_count=6)
    recs = [
        AircraftRecommendation(
            model="Citation Latitude",
            category="super-midsize",
            total_score=0.6934,
            confidence=0.5,
            rank=1,
            fit_verdict="CONDITIONAL FIT",
        ),
        AircraftRecommendation(
            model="Pilatus PC-24",
            category="light",
            total_score=0.81,
            confidence=0.5,
            rank=2,
            fit_verdict="GOOD FIT",
        ),
    ]
    du = {"hack_v2_ranking": _contract()}
    out = format_broker_recommendation_response(mission, recs, data_used=du)
    assert "## Ranked Aircraft List" in out
    assert "Mission Interpretation" not in out
    assert "Why:" not in out


def test_render_integrity_rejects_verdict_drift():
    du = {"hack_v2_ranking": _contract()}
    recs = [
        AircraftRecommendation(
            model="Pilatus PC-24",
            category="light",
            total_score=0.81,
            confidence=0.5,
            rank=1,
            fit_verdict="NOT A FIT",
        ),
    ]
    with pytest.raises(RenderIntegrityError):
        render_hack_v3_locked_response(du, recommendations=recs)


def test_render_integrity_rejects_order_drift():
    du = {"hack_v2_ranking": _contract()}
    recs = [
        AircraftRecommendation(
            model="Citation Latitude",
            category="super-midsize",
            total_score=0.69,
            confidence=0.5,
            rank=1,
            fit_verdict="CONDITIONAL FIT",
        ),
        AircraftRecommendation(
            model="Pilatus PC-24",
            category="light",
            total_score=0.81,
            confidence=0.5,
            rank=2,
            fit_verdict="GOOD FIT",
        ),
    ]
    with pytest.raises(RenderIntegrityError):
        render_hack_v3_locked_response(du, recommendations=recs)

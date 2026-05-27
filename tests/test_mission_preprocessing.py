"""Mission pre-processing layer — UNKNOWN semantics and no fabricated routes."""

from __future__ import annotations

import json

import pytest

from services.preprocessing import (
    UNKNOWN,
    attach_mission_preprocessing,
    preprocess_mission_from_query,
    preprocess_mission_json,
)
from services.preprocessing.schema import PreprocessedMission


def test_la_miami_extracts_route_fields():
    pre = preprocess_mission_from_query("8 pax LA to Miami nonstop recommend")
    assert pre.passengers == 8
    assert pre.origin == "Los Angeles"
    assert pre.destination == "Miami"
    assert pre.nonstop_required is True
    assert pre.route_evidence in ("validated_route", "explicit_from_to")


def test_no_route_without_places():
    pre = preprocess_mission_from_query("recommend a business jet for acquisition")
    assert pre.origin == UNKNOWN
    assert pre.destination == UNKNOWN
    assert pre.route_evidence == "none"


def test_transatlantic_keyword_does_not_invent_cities():
    pre = preprocess_mission_from_query("12 pax transatlantic nonstop large cabin")
    assert pre.transatlantic is True
    assert pre.origin == UNKNOWN
    assert pre.destination == UNKNOWN


def test_budget_and_ownership():
    pre = preprocess_mission_from_query(
        "250 hours a year fractional vs full ownership — around $10M budget"
    )
    assert pre.annual_hours == 250
    assert pre.ownership_interest == "fractional"
    assert pre.budget == 10_000_000.0


def test_unknown_defaults_empty_query():
    pre = preprocess_mission_from_query("")
    assert pre.passengers == UNKNOWN
    assert pre.origin == UNKNOWN
    assert pre.destination == UNKNOWN


def test_json_round_trip_public_fields_only():
    raw = preprocess_mission_json("6 passengers Dallas to Aspen")
    parsed = json.loads(raw)
    assert "route_evidence" not in parsed
    assert parsed["passengers"] == 6
    assert parsed["origin"] == "Dallas"
    assert parsed["destination"] == "Aspen"


def test_attach_mission_preprocessing_merges_data_used():
    du: dict = {}
    attach_mission_preprocessing(du, "8 pax LA to Miami")
    assert "mission_preprocessing" in du
    assert "mission_preprocessing_json" in du
    assert du["mission_preprocessing"]["destination"] == "Miami"


def test_multi_destination_marks_unknown_single_destination():
    q = (
        "I usually travel with executives from San Francisco to Tokyo and London. "
        "Nonstop westbound in winter."
    )
    pre = preprocess_mission_from_query(q)
    assert pre.passengers == 6
    assert pre.origin == "San Francisco"
    assert pre.destination == UNKNOWN
    assert pre.westbound is True
    assert pre.winter_operation is True


def test_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        PreprocessedMission(passengers=4, recommended_aircraft="G650")  # type: ignore[call-arg]

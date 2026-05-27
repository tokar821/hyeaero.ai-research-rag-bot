"""Mission Extraction Layer — unit tests."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from services.mission_extraction import (
    MissionExtractionResult,
    extract_mission_requirements,
    extract_mission_requirements_json,
    safe_validate_extraction,
    validate_extraction_json,
    validate_extraction_payload,
)

_AIRCRAFT_RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend|you should buy|best choice is|shortlist|top pick)\b",
    re.I,
)
_CATALOG_MODEL_RE = re.compile(
    r"\b(?:Challenger\s+\d+|Gulfstream|Citation|Falcon|Global\s+\d|Learjet|Praetor)\b",
    re.I,
)


def test_sf_tokyo_london_executives_example():
    q = (
        "I usually travel with executives from San Francisco to Tokyo and London. "
        "I want nonstop capability westbound in winter."
    )
    result = extract_mission_requirements(q)
    assert result.passengers == 6
    assert result.origin == "San Francisco"
    assert result.destination == ["Tokyo", "London"]
    assert result.nonstop_required is True
    assert result.westbound_sensitive is True
    assert result.winter_ops is True
    assert result.transpacific is True
    assert result.transatlantic is True
    assert result.international_ops is True
    assert result.inferred_aircraft_category == "ultra_long_range"


def test_json_output_is_valid_json_only():
    q = "6 pax Dallas to Aspen hot and high"
    raw = extract_mission_requirements_json(q)
    assert raw == raw.strip()
    assert not raw.startswith("```")
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert parsed["passengers"] == 6
    assert parsed["origin"] == "Dallas"
    assert parsed["destination"] == ["Aspen"]
    assert parsed["hot_high_ops"] is True
    assert parsed["mountain_airports"] is True


def test_miami_caribbean_short_runway():
    q = "8 passengers Miami to Caribbean, short runway focus"
    result = extract_mission_requirements(q)
    assert result.passengers == 8
    assert result.origin == "Miami"
    assert result.destination == ["Caribbean"]
    assert result.short_runway_ops is True
    assert result.caribbean is True
    assert result.inferred_aircraft_category == "midsize"


def test_fractional_ownership_hours():
    q = "We fly 250 hours a year — fractional vs full ownership for a Challenger 350?"
    result = extract_mission_requirements(q)
    assert result.mission_type == "ownership"
    assert result.ownership_interest == "fractional"
    assert result.annual_hours == 250
    assert result.origin is None


def test_empty_message_all_null():
    result = extract_mission_requirements("")
    dumped = result.model_dump()
    assert dumped["passengers"] is None
    assert dumped["origin"] is None
    assert dumped["destination"] is None


def test_no_aircraft_recommendation_language_in_json():
    q = "12 passengers New York to London nonstop — what large-cabin options are realistic?"
    raw = extract_mission_requirements_json(q)
    assert not _AIRCRAFT_RECOMMENDATION_RE.search(raw)
    assert not _CATALOG_MODEL_RE.search(raw)


def test_validate_rejects_extra_fields():
    with pytest.raises(ValidationError):
        validate_extraction_payload({"passengers": 4, "recommended_aircraft": "G650"})


def test_validate_rejects_invalid_passengers():
    with pytest.raises(ValidationError):
        validate_extraction_payload({"passengers": 0})


def test_validate_json_round_trip():
    q = "G650 vs Global 7500 for 12 pax transatlantic cabin"
    raw = extract_mission_requirements_json(q)
    validated = validate_extraction_json(raw)
    assert validated.mission_type == "comparison"
    assert validated.passengers == 12
    assert validated.transatlantic is True


def test_safe_validate_returns_error_on_bad_data():
    ok, err = safe_validate_extraction({"passengers": "many"})
    assert ok is None
    assert err is not None


def test_operating_cost_priority_high():
    q = "8 passengers LA to Miami nonstop — prioritize operating economics"
    result = extract_mission_requirements(q)
    assert result.operating_cost_priority == "high"


def test_budget_extracted():
    q = "Around $10M budget for a midsize jet"
    result = extract_mission_requirements(q)
    assert result.budget == 10_000_000.0


def test_comparison_mission_type():
    q = "Falcon 2000 vs Challenger 350 for 9 passengers"
    result = extract_mission_requirements(q)
    assert result.mission_type == "comparison"
    assert result.passengers == 9

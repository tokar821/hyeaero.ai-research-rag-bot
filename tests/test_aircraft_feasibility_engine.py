"""Deterministic aircraft feasibility engine — hard rejection tests."""

from __future__ import annotations

import pytest

from services.aircraft_feasibility import (
    AircraftFeasibilityVerdict,
    evaluate_aircraft_feasibility,
    filter_feasible_aircraft,
    mission_context_from_json,
)


def _mission(**kwargs):
    base = {
        "passengers": 6,
        "origin": None,
        "destination": None,
        "nonstop_required": True,
        "westbound_sensitive": False,
        "winter_ops": False,
        "baggage_priority": None,
        "runway_priority": None,
        "short_runway_ops": False,
        "mountain_airports": False,
        "hot_high_ops": False,
        "international_ops": True,
        "transatlantic": False,
        "transpacific": False,
    }
    base.update(kwargs)
    return base


def test_output_schema_shape():
    mission = _mission(
        origin="Miami",
        destination=["Caribbean"],
        nonstop_required=False,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Citation Latitude")
    d = verdict.to_dict()
    assert set(d.keys()) >= {
        "feasible",
        "rejectionReasons",
        "payloadPenalty",
        "runwayPenalty",
        "winterPenalty",
        "reservesSatisfied",
    }
    assert isinstance(d["feasible"], bool)
    assert isinstance(d["rejectionReasons"], list)
    assert d["reservesSatisfied"] is True


def test_challenger_350_lax_london_nonstop_hard_reject():
    """Super-mids cannot do US West Coast → London nonstop year-round."""
    mission = _mission(
        origin="Los Angeles",
        destination=["London"],
        nonstop_required=True,
        transatlantic=True,
        westbound_sensitive=True,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Challenger 350")
    assert not verdict.feasible
    assert verdict.rejection_reasons
    assert any(
        "transatlantic" in r.lower() or "practical available" in r.lower()
        for r in verdict.rejection_reasons
    )
    assert verdict.stage_distance_nm >= 5400


def test_praetor_600_ny_tokyo_westbound_winter_hard_reject():
    mission = _mission(
        origin="New York",
        destination=["Tokyo"],
        nonstop_required=True,
        westbound_sensitive=True,
        winter_ops=True,
        transpacific=True,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Praetor 600")
    assert not verdict.feasible
    assert any(
        "transpacific" in r.lower() or "ultra-long" in r.lower() or "practical available" in r.lower()
        for r in verdict.rejection_reasons
    )
    assert verdict.winter_penalty > 0


def test_longitude_honolulu_sydney_hard_reject():
    mission = _mission(
        origin="Honolulu",
        destination=["Sydney"],
        nonstop_required=True,
        transpacific=True,
        westbound_sensitive=True,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Challenger Longitude")
    assert not verdict.feasible
    assert verdict.stage_distance_nm >= 4300


def test_global_7500_sf_tokyo_westbound_winter_feasible():
    mission = _mission(
        origin="San Francisco",
        destination=["Tokyo"],
        nonstop_required=True,
        westbound_sensitive=True,
        winter_ops=True,
        transpacific=True,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Global 7500")
    assert verdict.feasible
    assert verdict.rejection_reasons == []
    assert verdict.reserves_satisfied


def test_g650er_ny_tokyo_westbound_winter_hard_reject():
    """Even ULR G650ER is hard-rejected on NY–Tokyo westbound winter under conservative margin."""
    mission = _mission(
        origin="New York",
        destination=["Tokyo"],
        nonstop_required=True,
        westbound_sensitive=True,
        winter_ops=True,
        transpacific=True,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Gulfstream G650ER")
    assert not verdict.feasible
    assert verdict.winter_penalty > 0


def test_nbaa_reserves_always_applied():
    ctx = mission_context_from_json(
        _mission(origin="Boston", destination=["Paris"], nonstop_required=True, transatlantic=True)
    )
    from services.aircraft_feasibility.range_margin import compute_mission_range_requirement

    req = compute_mission_range_requirement(ctx)
    assert req.nbaa_reserve_nm >= 200.0
    assert req.reserves_satisfied


def test_payload_penalty_reduces_available_range():
    mission = _mission(
        origin="Los Angeles",
        destination=["Miami"],
        passengers=10,
        baggage_priority="high",
        nonstop_required=True,
    )
    light = evaluate_aircraft_feasibility(mission, "Challenger 350")
    assert light.payload_penalty > 0


def test_mountain_hot_high_rejects_cj2():
    mission = _mission(
        origin="Dallas",
        destination=["Aspen"],
        mountain_airports=True,
        hot_high_ops=True,
        nonstop_required=False,
    )
    verdict = evaluate_aircraft_feasibility(mission, "Citation CJ2")
    assert not verdict.feasible
    assert any("mountain" in r.lower() or "hot" in r.lower() for r in verdict.rejection_reasons)


def test_short_runway_rejects_global_7500():
    mission = _mission(
        origin="Miami",
        destination=["Caribbean"],
        short_runway_ops=True,
        runway_priority="high",
        nonstop_required=False,
    )
    results = filter_feasible_aircraft(
        mission,
        ["Pilatus PC-24", "Citation Latitude", "Global 7500"],
    )
    assert results["Global 7500"].feasible is False
    assert results["Citation Latitude"].feasible is True


def test_filter_never_includes_impossible_in_feasible_list():
    mission = _mission(
        origin="Los Angeles",
        destination=["London"],
        nonstop_required=True,
        transatlantic=True,
    )
    results = filter_feasible_aircraft(
        mission,
        ["Challenger 350", "Praetor 600", "Gulfstream G650", "Global 7500"],
    )
    assert not results["Challenger 350"].feasible
    assert not results["Praetor 600"].feasible
    assert results["Gulfstream G650"].feasible or results["Global 7500"].feasible


def test_unknown_aircraft_rejected():
    mission = _mission(origin="New York", destination=["Boston"])
    verdict = evaluate_aircraft_feasibility(mission, "Not A Real Jet")
    assert not verdict.feasible
    assert any("unknown" in r.lower() for r in verdict.rejection_reasons)


def test_sf_tokyo_executives_mission_json_example():
    """End-to-end from mission extraction-style JSON."""
    mission = {
        "passengers": 6,
        "origin": "San Francisco",
        "destination": ["Tokyo"],
        "nonstop_required": True,
        "westbound_sensitive": True,
        "winter_ops": True,
        "transpacific": True,
        "international_ops": True,
    }
    assert not evaluate_aircraft_feasibility(mission, "Challenger 350").feasible
    assert not evaluate_aircraft_feasibility(mission, "Challenger Longitude").feasible
    assert evaluate_aircraft_feasibility(mission, "Global 7500").feasible

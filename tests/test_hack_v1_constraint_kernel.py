"""HACK v1 — hard aviation constraint kernel tests."""

from __future__ import annotations

import pytest

from services.consultant.mission_state import MissionState
from services.mission.adapters import mission_state_to_profile
from services.mission.models import MissionProfile, Route
from services.recommendation.hack_v1_constraint_kernel import (
    HACK_V1_EMPTY_MESSAGE,
    apply_hack_v1_gate,
    run_hack_v1_constraint_kernel,
)
from services.recommendation.tier_downgrade_recovery import tier_downgrade_recovery


def _profile(routes: list[str], *, pax: int = 8, seasonal: str = "") -> MissionProfile:
    return MissionProfile(
        routes=[Route.from_label(r) for r in routes if Route.from_label(r)],
        passengers=pax,
        seasonal_note=seasonal,
        nonstop_required=True,
        nbaa_reserve_required=True,
    )


def test_light_jets_hard_rejected_transatlantic():
    profile = _profile(["San Francisco -> London"], pax=8)
    result = run_hack_v1_constraint_kernel(
        profile,
        ["Citation CJ2", "Citation CJ4", "Learjet 75", "Gulfstream G650ER"],
    )
    assert "Citation CJ2" not in result.feasible_aircraft_list
    assert "Citation CJ4" not in result.feasible_aircraft_list
    assert "Learjet 75" not in result.feasible_aircraft_list
    assert "Gulfstream G650ER" in result.feasible_aircraft_list
    rules = {r.rule_id for r in result.rejection_log}
    assert rules & {"light_jet_long_stage", "class_band_violation", "transatlantic_light_jet_pax"}


def test_westbound_winter_transatlantic_blocks_midsize():
    profile = _profile(["New York -> London"], pax=6, seasonal="winter westbound")
    profile.westbound_sensitive = True
    result = run_hack_v1_constraint_kernel(
        profile,
        ["Challenger 350", "Gulfstream G280", "Gulfstream G650ER"],
    )
    assert "Gulfstream G650ER" in result.feasible_aircraft_list
    assert "Challenger 350" not in result.feasible_aircraft_list
    assert "Gulfstream G280" not in result.feasible_aircraft_list


def test_tier_recovery_respects_hack_v1_empty():
    mission = MissionState(routes=["San Francisco -> London"], passenger_count=8)
    du: dict = {"hack_v1_constraint_empty": True, "hack_v1_permanent_exclusions": ["Citation CJ2"]}
    recs, tier = tier_downgrade_recovery(
        mission,
        "recommend aircraft",
        prior_recommendations=[],
        data_used=du,
    )
    assert recs == []
    assert tier == "hack_v1_empty"


def test_empty_constraint_message_constant():
    assert "NO PHYSICALLY VIABLE" in HACK_V1_EMPTY_MESSAGE


def test_apply_gate_empty_intersection_marks_constraint_empty():
    profile = _profile(["Yellowknife -> Remote Gravel Strips"], pax=6)
    filtered, result = apply_hack_v1_gate(
        profile,
        [],
        all_candidates=["Global 7500"],
        query="Northern Canada Arctic gravel strips",
    )
    assert filtered == []
    assert result.constraint_empty is True


def test_apply_gate_uses_kernel_survivors_when_graph_feasible_empty():
    profile = _profile(["Yellowknife -> Remote Gravel Strips"], pax=6)
    filtered, result = apply_hack_v1_gate(
        profile,
        [],
        all_candidates=["Pilatus PC-24", "Global 7500"],
        query="Northern Canada Arctic gravel strips",
    )
    assert "Pilatus PC-24" in filtered
    assert result.constraint_empty is False


def test_gravel_arctic_blocks_heavy_jets():
    profile = _profile(["Yellowknife -> Remote Gravel Strips"], pax=6)
    result = run_hack_v1_constraint_kernel(
        profile,
        ["Global 7500", "Pilatus PC-24", "Citation CJ2"],
        query="Northern Canada Arctic gravel strips",
    )
    assert "Global 7500" not in result.feasible_aircraft_list
    assert "Pilatus PC-24" in result.feasible_aircraft_list

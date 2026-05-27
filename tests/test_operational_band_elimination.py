"""Operational-band elimination must not resurrect eliminated models."""

from __future__ import annotations

from services.elimination.operational_band import (
    determine_operational_band,
    filter_models_to_operational_band,
)
from services.mission.models import MissionProfile


def test_mountain_zero_nm_does_not_restore_eliminated_midsize():
    # Non-mountain + 0 nm: all super-mids outside allowed bands — must not restore eliminated aircraft.
    profile = MissionProfile(nonstop_required=True)
    feasible = ["Citation Latitude", "Challenger 350", "Gulfstream G280", "Praetor 600"]
    cats = {
        "citation latitude": "midsize",
        "challenger 350": "super-midsize",
        "gulfstream g280": "super-midsize",
        "praetor 600": "super-midsize",
    }
    result = determine_operational_band(
        profile,
        feasible,
        distance_nm=0.0,
        model_categories=cats,
    )
    assert "Citation Latitude" in result.survivors
    assert "Challenger 350" in result.eliminated or "Challenger 350" in result.downgraded
    filtered = filter_models_to_operational_band(feasible, result, include_downgraded=False)
    assert filtered == ["Citation Latitude"]


def test_filter_empty_survivors_returns_empty():
    from services.elimination.operational_band import BandEliminationResult

    result = BandEliminationResult(target_band=None, survivors=[], eliminated=["Citation Latitude"])
    assert filter_models_to_operational_band(["Citation Latitude", "Challenger 350"], result) == []

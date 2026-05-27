"""Soft elimination downgrades — brokers explain compromises instead of disappearing aircraft."""

from services.elimination.conditional_downgrade import (
    CompromiseLabel,
    elimination_severity,
    feasibility_for_soft_elimination,
)
from services.elimination.operational_band import (
    BandEliminationResult,
    determine_operational_band,
    filter_models_to_operational_band,
)
from services.mission.models import MissionProfile


def test_in_band_comparison_is_soft():
    reason = "Not comparable in-band; mission band is ultra_long_range"
    assert elimination_severity(reason, distance_nm=3100, elimination_kind="band") == "soft"


def test_light_jet_on_long_corridor_is_hard():
    reason = (
        "Outside operational band for 3200 nm corridor "
        "(light_jet not credible vs ultra_long_range)"
    )
    assert elimination_severity(reason, distance_nm=3200, elimination_kind="band") == "hard"


def test_soft_feasibility_stays_feasible():
    fr = feasibility_for_soft_elimination(
        "Gulfstream G280",
        "Not comparable in-band; mission band is large_cabin",
        label=CompromiseLabel.VIABLE_WITH_COMPROMISES,
    )
    assert fr.feasible
    assert fr.operational_risk_level == "compromise"


def test_filter_includes_downgraded():
    result = BandEliminationResult(
        target_band=None,
        survivors=["G650"],
        downgraded=["Challenger 350"],
        compromise_labels={"Challenger 350": "VIABLE WITH COMPROMISES"},
    )
    out = filter_models_to_operational_band(
        ["G650", "Challenger 350", "Citation Latitude"],
        result,
    )
    assert "G650" in out
    assert "Challenger 350" in out
    assert "Citation Latitude" not in out


def test_transcon_soft_downgrades_super_mid_outside_ulr_band():
    profile = MissionProfile(nonstop_required=True)
    feasible = ["Gulfstream G650", "Gulfstream G280"]
    cats = {
        "gulfstream g650": "ultra-long-range",
        "gulfstream g280": "super-midsize",
    }
    result = determine_operational_band(
        profile,
        feasible,
        distance_nm=3100,
        model_categories=cats,
    )
    assert "Gulfstream G650" in result.survivors
    assert "Gulfstream G280" in result.downgraded
    assert result.compromise_labels.get("Gulfstream G280")
    filtered = filter_models_to_operational_band(feasible, result)
    assert "Gulfstream G650" in filtered
    assert "Gulfstream G280" in filtered

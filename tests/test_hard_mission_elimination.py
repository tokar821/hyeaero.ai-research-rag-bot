"""Hard elimination before ranking / narrative — ULR westbound transpacific."""

from services.mission.feasibility_engine import filter_feasible_aircraft
from services.mission.models import MissionProfile, Route
from services.pipeline.run_pipeline import run_advisory_pipeline
from services.recommendation.hard_mission_elimination import (
    RULE_ULR_WESTBOUND_PACIFIC,
    apply_hard_mission_elimination,
    detect_hard_elimination_context,
    hard_elimination_reason,
)


def _ny_tokyo_profile() -> MissionProfile:
    return MissionProfile(
        passengers=8,
        routes=[Route(origin="New York", destination="Tokyo")],
        nonstop_required=True,
        westbound_sensitive=True,
        seasonal_note="winter_headwinds",
        nbaa_reserve_required=True,
    )


def test_detects_ulr_westbound_pacific_gate():
    profile = _ny_tokyo_profile()
    ctx = detect_hard_elimination_context(profile)
    assert ctx is not None
    assert ctx.rule_id == RULE_ULR_WESTBOUND_PACIFIC
    assert ctx.required_route_nm >= 5500


def test_auto_rejects_super_mids_and_named_models():
    profile = _ny_tokyo_profile()
    ctx = detect_hard_elimination_context(profile)
    assert ctx is not None
    for model in (
        "Challenger 350",
        "Gulfstream G280",
        "Praetor 600",
        "Citation Latitude",
        "Challenger Longitude",
    ):
        assert hard_elimination_reason(model, ctx)


def test_allows_ulr_allowlist_only():
    profile = _ny_tokyo_profile()
    ctx = detect_hard_elimination_context(profile)
    assert ctx is not None
    for model in ("Global 7500", "Gulfstream G650", "Gulfstream G650ER", "Falcon 8X", "Global 6500"):
        assert hard_elimination_reason(model, ctx) is None


def test_apply_hard_elimination_filters_feasible_list():
    profile = _ny_tokyo_profile()
    candidates = [
        "Challenger 350",
        "Gulfstream G280",
        "Praetor 600",
        "Global 7500",
        "Falcon 8X",
        "Gulfstream G650",
    ]
    survivors, eliminated, log, ctx = apply_hard_mission_elimination(profile, candidates)
    assert ctx is not None
    assert "Challenger 350" in eliminated
    assert "Gulfstream G280" in eliminated
    assert "Global 7500" in survivors
    assert "Falcon 8X" in survivors
    assert any(e.get("hard_rule_id") == RULE_ULR_WESTBOUND_PACIFIC for e in log)


def test_pipeline_excludes_super_mids_on_sfo_tokyo_winter():
    profile = MissionProfile(
        passengers=6,
        routes=[Route(origin="San Francisco", destination="Tokyo")],
        nonstop_required=True,
        westbound_sensitive=True,
        seasonal_note="winter_headwinds",
        nbaa_reserve_required=True,
    )
    results = filter_feasible_aircraft(
        profile,
        [
            "Challenger 350",
            "Gulfstream G280",
            "Praetor 600",
            "Challenger Longitude",
            "Global 7500",
            "Falcon 8X",
        ],
    )
    assert not results["Challenger 350"].feasible
    assert not results["Praetor 600"].feasible
    assert results["Global 7500"].feasible


def test_advisory_pipeline_shortlist_ulr_only():
    result = run_advisory_pipeline(
        "New York to Tokyo nonstop westbound winter recommend",
        mission_profile=_ny_tokyo_profile(),
        max_results=6,
    )
    models = [r.model for r in result.recommendations]
    assert models
    for banned in ("Challenger 350", "Gulfstream G280", "Praetor 600", "Citation Latitude"):
        assert banned not in models
    assert all(
        m in ("Global 7500", "Falcon 8X", "Gulfstream G650", "Gulfstream G650ER", "Global 6500")
        or "Global" in m
        or "Falcon 8X" in m
        or "G650" in m
        for m in models
    )

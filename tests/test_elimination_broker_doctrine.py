"""Elimination-first broker doctrine — bands, verdicts, language, image policy."""

from __future__ import annotations

from services.broker.broker_language import apply_broker_language_rules, broker_refusal_message
from services.broker.broker_verdicts import BrokerVerdict, map_legacy_verdict, verdict_from_operational_signals
from services.elimination.operational_band import (
    OperationalBand,
    determine_operational_band,
    model_operational_band,
    models_comparable_in_band,
)
from services.mission.mission_profile_inference import infer_mission_profile
from services.mission.models import MissionProfile
from services.orchestration.image_trust_policy import (
    explicit_cabin_interior_requested,
    should_activate_image_trust,
)
from services.session.broker_memory import BrokerMemory, update_broker_memory_from_turn


def test_legacy_verdict_maps_to_broker_vocabulary():
    assert map_legacy_verdict("BEST FIT") == BrokerVerdict.PRIMARY_RECOMMENDATION
    assert map_legacy_verdict("CONDITIONAL FIT") == BrokerVerdict.VIABLE_WITH_COMPROMISES
    assert map_legacy_verdict("NOT A FIT") == BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE


def test_verdict_from_operational_signals_not_score_only():
    assert (
        verdict_from_operational_signals(
            hard_feasible=True, margin_nm=50, penalty_total=0.1
        )
        == BrokerVerdict.MISSION_RISKY
    )
    assert (
        verdict_from_operational_signals(
            hard_feasible=True, margin_nm=500, penalty_total=0.05
        )
        == BrokerVerdict.PRIMARY_RECOMMENDATION
    )


def test_teb_lon_eliminates_g280_from_ulr_band():
    profile = MissionProfile(nonstop_required=True)
    feasible = ["g280", "g650", "global 7500"]
    result = determine_operational_band(profile, feasible, distance_nm=3100)
    assert result.target_band == OperationalBand.ULTRA_LONG_RANGE
    assert "g650" in result.survivors
    assert "global 7500" in result.survivors
    assert "g280" in result.downgraded or "g280" in result.eliminated


def test_cross_band_comparison_invalid():
    ok, reason = models_comparable_in_band("g280", "g650")
    assert not ok
    assert "cross-band" in reason.lower()


def test_g650_vs_global_7500_same_band():
    ok, _ = models_comparable_in_band("g650", "global 7500")
    assert ok
    assert model_operational_band("g650") == model_operational_band("global 7500")


def test_mission_profile_inference_executive():
    inferred = infer_mission_profile("Board roadshow, 8 executives, nonstop TEB to London")
    assert inferred.utilization_style == "executive_shuttle"
    assert inferred.nonstop_preference is True


def test_broker_memory_persists_nonstop():
    mem = BrokerMemory()
    mem = update_broker_memory_from_turn(
        mem,
        route="teterboro -> london",
        inferred_profile={"nonstop_preference": True, "utilization_style": "executive_shuttle"},
    )
    assert mem.nonstop_preference is True
    assert "teterboro -> london" in mem.recurring_routes


def test_anti_brochure_language():
    text, violations = apply_broker_language_rules("This is an ideal aircraft with excellent range.")
    assert violations
    assert "ideal aircraft" not in text.lower()


def test_broker_refusal_not_robotic():
    msg = broker_refusal_message()
    assert "reliable data for this" not in msg.lower()


def test_image_trust_off_on_advisory_turn():
    assert not should_activate_image_trust("Recommend aircraft for TEB to London, 8 pax, February")
    assert should_activate_image_trust("Show me a photo of N650GX exterior")
    assert not explicit_cabin_interior_requested("Recommend G650 for this mission")
    assert explicit_cabin_interior_requested("Show me the cabin layout on the G650")

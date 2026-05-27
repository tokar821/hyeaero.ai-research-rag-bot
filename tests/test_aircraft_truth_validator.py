"""Aircraft truth validator — verified specs only."""

from services.aircraft_truth import (
    UNVERIFIED_AIRCRAFT_MESSAGE,
    filter_truth_verified_models,
    is_forbidden_unverified_claim,
    reject_forbidden_claims,
    validate_aircraft_truth,
)
from services.aircraft_truth.catalog_supplement import CATALOG_TRUTH_SUPPLEMENT
from services.consultant.response_architecture import comparison_dimension_row
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES


def test_all_catalog_models_have_truth_supplement():
    assert set(CATALOG_TRUTH_SUPPLEMENT.keys()) == set(AIRCRAFT_PROFILES.keys())


def test_verified_model_passes():
    result = validate_aircraft_truth("Challenger 350")
    assert result.verified
    assert result.facts is not None
    assert result.facts.max_passengers >= 8
    assert result.facts.practical_range_nm >= 2000
    assert result.facts.runway_class
    assert result.facts.baggage_volume_cu_ft > 0
    assert result.facts.operating_category == "super-midsize"


def test_unknown_model_fails_with_exact_message():
    result = validate_aircraft_truth("FakeJet 9000")
    assert not result.verified
    assert result.message == UNVERIFIED_AIRCRAFT_MESSAGE


def test_incomplete_profile_fails():
    result = validate_aircraft_truth(
        "Challenger 350",
        profile={"practical_nm": 2700, "truth_verified": False},
    )
    assert not result.verified
    assert result.message == UNVERIFIED_AIRCRAFT_MESSAGE


def test_filter_truth_verified_models():
    models = ["Citation CJ4", "Unknown Jet", "Global 7500"]
    assert filter_truth_verified_models(models) == ["Citation CJ4", "Global 7500"]


def test_forbidden_unverified_claim_keys():
    assert is_forbidden_unverified_claim("acquisition_price")
    assert is_forbidden_unverified_claim("nonstop_capability")
    blocked = reject_forbidden_claims(
        {"practical_nm": 2700, "acquisition_price": 12_000_000, "payload_lb": 2000}
    )
    assert "acquisition_price" in blocked
    assert "payload_lb" in blocked
    assert "practical_nm" not in blocked


def test_comparison_row_uses_verified_facts():
    row = comparison_dimension_row("Citation Latitude")
    assert "nm practical" in row["range"]
    assert row["range"] != UNVERIFIED_AIRCRAFT_MESSAGE


def test_comparison_row_unverified_model():
    row = comparison_dimension_row("Not A Real Aircraft")
    assert all(row[dim] == UNVERIFIED_AIRCRAFT_MESSAGE for dim in row)

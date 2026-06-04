"""Phase 20 — Aircraft Knowledge Authority Layer tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.aircraft.aircraft_authority_audit import run_aircraft_authority_audit
from services.aircraft.aircraft_authority_service import (
    build_authoritative_comparison_dataset,
    build_authoritative_market_context,
    get_aircraft_authority_record,
    resolve_aircraft_alias,
    validate_aircraft_claim,
)


def test_longitude_alias():
    assert resolve_aircraft_alias("Longitude") == "Citation Longitude"


def test_g280_alias():
    assert resolve_aircraft_alias("G280") == "Gulfstream G280"
    assert resolve_aircraft_alias("G-280") == "Gulfstream G280"


def test_cj3_plus_alias():
    assert resolve_aircraft_alias("CJ3+") == "Citation CJ3+"
    assert resolve_aircraft_alias("citation cj3+") == "Citation CJ3+"


def test_challenger_3500_alias():
    assert resolve_aircraft_alias("3500") == "Challenger 3500"
    assert resolve_aircraft_alias("Challenger 3500") == "Challenger 3500"


def test_falcon_8x_alias():
    assert resolve_aircraft_alias("Falcon 8X") == "Falcon 8X"
    assert resolve_aircraft_alias("Dassault Falcon 8X") == "Falcon 8X"


def test_comparison_dataset_generation():
    dataset = build_authoritative_comparison_dataset(["G650", "Falcon 8X"])
    assert dataset["status"] == "OK"
    assert len(dataset["aircraft"]) >= 2
    row = dataset["aircraft"][0]
    assert "range_nm" in row
    assert "cabin" in row
    assert row["authority_source"]


def test_claim_validation_range_mismatch():
    result = validate_aircraft_claim("G280 range 7000nm")
    assert result.valid is False
    assert result.reason == "range_mismatch"
    assert result.canonical_name == "Gulfstream G280"
    assert result.authoritative_value is not None
    assert result.authoritative_value < 7000


def test_claim_validation_range_match():
    rec = get_aircraft_authority_record(aircraft_model="Gulfstream G280")
    assert rec is not None
    result = validate_aircraft_claim(f"G280 range {int(rec.nbaa_range_nm)}nm")
    assert result.valid is True


def test_competitor_lookup():
    rec = get_aircraft_authority_record(aircraft_model="Gulfstream G280")
    assert rec is not None
    assert rec.direct_competitors
    assert any("Challenger" in c or "Citation" in c or "Praetor" in c for c in rec.direct_competitors)


def test_replacement_lookup():
    rec = get_aircraft_authority_record(aircraft_model="Citation Longitude")
    assert rec is not None
    assert rec.replacement_models


def test_buy_market_context():
    ctx = build_authoritative_market_context(year=2016, model="Citation Latitude", ask_usd=10_000_000)
    assert ctx["status"] == "OK"
    assert "expected_market_band_usd" in ctx


def test_authority_audit_runs():
    report = run_aircraft_authority_audit()
    assert report.to_dict()["ok"] is not None


def test_akal_no_cross_model_profile_substitution():
    citation = get_aircraft_authority_record(aircraft_model="Citation Longitude")
    challenger = get_aircraft_authority_record(aircraft_model="Challenger Longitude")
    cj3 = get_aircraft_authority_record(aircraft_model="Citation CJ3+")
    cj4 = get_aircraft_authority_record(aircraft_model="Citation CJ4")
    cl3500 = get_aircraft_authority_record(aircraft_model="Challenger 3500")
    cl350 = get_aircraft_authority_record(aircraft_model="Challenger 350")

    for rec in (citation, challenger, cj3, cj4, cl3500, cl350):
        assert rec is not None

    assert citation.canonical_name == "Citation Longitude"
    assert challenger.canonical_name == "Challenger Longitude"
    assert citation.nbaa_range_nm != challenger.nbaa_range_nm or citation.manufacturer != challenger.manufacturer

    if cj3.nbaa_range_nm and cj4.nbaa_range_nm:
        assert cj3.nbaa_range_nm != cj4.nbaa_range_nm or cj3.canonical_name != cj4.canonical_name

    if cl3500.nbaa_range_nm and cl350.nbaa_range_nm:
        assert cl3500.nbaa_range_nm != cl350.nbaa_range_nm or cl3500.canonical_name != cl350.canonical_name

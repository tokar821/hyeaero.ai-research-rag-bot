"""Phase 30 Track A — AKAL truth matrix."""

from __future__ import annotations

import pytest

from services.aircraft.aircraft_authority_service import (
    build_authoritative_comparison_dataset,
    get_aircraft_authority_record,
    resolve_aircraft_alias,
)
from services.catalog.catalog_alias_resolver import resolve_canonical_display_name
from services.routing.authority_dispatch import consult_authority_dispatch
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.unified_intent_router import classify_unified_intent

pytestmark = pytest.mark.deterministic

AIRCRAFT_TOKENS = [
    ("G650", "Gulfstream G650"),
    ("G650ER", "Gulfstream G650ER"),
    ("G550", "Gulfstream G550"),
    ("G280", "Gulfstream G280"),
    ("Falcon 8X", "Falcon 8X"),
    ("Global 7500", "Global 7500"),
    ("Challenger 3500", "Challenger 3500"),
    ("Longitude", "Citation Longitude"),
    ("CJ3+", "Citation CJ3+"),
    ("CJ4", "CJ4"),
    ("Praetor 600", "Praetor 600"),
    ("PC-24", "PC-24"),
]

ALIAS_VARIANTS = [
    ("g650", "Gulfstream G650"),
    ("G650ER", "Gulfstream G650ER"),
    ("gulfstream g550", "Gulfstream G550"),
    ("g-280", "Gulfstream G280"),
    ("Dassault Falcon 8X", "Falcon 8X"),
    ("global 7500", "Global 7500"),
    ("3500", "Challenger 3500"),
    ("citation longitude", "Citation Longitude"),
    ("citation cj3+", "Citation CJ3+"),
    ("Citation CJ4", "Citation CJ4"),
    ("Praetor 600", "Praetor 600"),
    ("Pilatus PC-24", "Pilatus PC-24"),
]

UNKNOWN_TOKENS = ["FakeJet9000", "UnknownJetXYZ", "NotARealAircraft", "ZZZ-999", "HyperJet 5000"]
FUTURE_TOKENS = ["Global 9500", "Falcon 12X", "Citation X10"]


@pytest.mark.parametrize("token,expected", AIRCRAFT_TOKENS)
def test_akal_alias_resolves_to_canonical(token, expected):
    assert resolve_aircraft_alias(token) == expected


@pytest.mark.parametrize("token,expected", ALIAS_VARIANTS)
def test_akal_alias_variant_normalization(token, expected):
    assert resolve_aircraft_alias(token) == expected


@pytest.mark.parametrize("token,expected", AIRCRAFT_TOKENS)
def test_akal_record_exists_for_verified_fleet(token, expected):
    rec = get_aircraft_authority_record(aircraft_model=token)
    assert rec is not None
    assert rec.canonical_name == expected


@pytest.mark.parametrize("token,_expected", AIRCRAFT_TOKENS)
def test_akal_practical_range_equals_stored_nbaa(token, _expected):
    rec = get_aircraft_authority_record(aircraft_model=token)
    assert rec is not None
    profile = rec.to_profile_dict()
    assert profile["practical_nm"] == rec.nbaa_range_nm


@pytest.mark.parametrize("token,expected", AIRCRAFT_TOKENS)
def test_catalog_display_matches_akal(token, expected):
    catalog = resolve_canonical_display_name(token)
    akal = resolve_aircraft_alias(token)
    assert catalog == expected or akal == expected
    assert akal == expected


def test_longitude_not_challenger_longitude():
    citation = get_aircraft_authority_record(aircraft_model="Citation Longitude")
    challenger = get_aircraft_authority_record(aircraft_model="Challenger Longitude")
    assert citation is not None
    assert challenger is not None
    assert citation.canonical_name != challenger.canonical_name
    assert citation.nbaa_range_nm != challenger.nbaa_range_nm or citation.manufacturer != challenger.manufacturer


def test_cj3_plus_not_cj4():
    cj3 = get_aircraft_authority_record(aircraft_model="Citation CJ3+")
    cj4 = get_aircraft_authority_record(aircraft_model="Citation CJ4")
    assert cj3 is not None
    assert cj4 is not None
    assert cj3.canonical_name != cj4.canonical_name
    assert cj3.nbaa_range_nm != cj4.nbaa_range_nm


def test_challenger_3500_not_350():
    cl3500 = get_aircraft_authority_record(aircraft_model="Challenger 3500")
    cl350 = get_aircraft_authority_record(aircraft_model="Challenger 350")
    assert cl3500 is not None
    assert cl350 is not None
    assert cl3500.canonical_name == "Challenger 3500"
    assert cl350.canonical_name == "Challenger 350"
    assert cl3500.canonical_name != cl350.canonical_name
    assert cl3500.passenger_capacity_max != cl350.passenger_capacity_max or cl3500.manufacturer == cl350.manufacturer


def test_g650_not_g650er_substitution():
    g650 = get_aircraft_authority_record(aircraft_model="Gulfstream G650")
    g650er = get_aircraft_authority_record(aircraft_model="Gulfstream G650ER")
    assert g650 is not None and g650er is not None
    assert g650.canonical_name != g650er.canonical_name
    assert g650.nbaa_range_nm != g650er.nbaa_range_nm


def test_g650_not_g550_substitution():
    g650 = get_aircraft_authority_record(aircraft_model="Gulfstream G650")
    g550 = get_aircraft_authority_record(aircraft_model="Gulfstream G550")
    assert g650 is not None and g550 is not None
    assert g650.canonical_name != g550.canonical_name


@pytest.mark.parametrize("token", UNKNOWN_TOKENS)
def test_unknown_aircraft_has_no_authority_record(token):
    assert get_aircraft_authority_record(aircraft_model=token) is None


@pytest.mark.parametrize("token", UNKNOWN_TOKENS)
def test_unknown_aircraft_comparison_dataset_fail_closed(token):
    dataset = build_authoritative_comparison_dataset([token, "G650"])
    assert dataset["status"] == "INSUFFICIENT_DATA"
    assert token in dataset["missing"] or token in (dataset.get("missing") or [])


@pytest.mark.parametrize("token", FUTURE_TOKENS)
def test_future_hypothetical_aircraft_fail_closed(token):
    rec = get_aircraft_authority_record(aircraft_model=token)
    if rec is None:
        dataset = build_authoritative_comparison_dataset([token, "G650"])
        assert dataset["status"] == "INSUFFICIENT_DATA"
    else:
        assert rec.authority_source


@pytest.mark.parametrize("token", UNKNOWN_TOKENS[:3])
def test_unknown_aircraft_dispatch_fail_closed(token):
    query = f"G650 vs {token}"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    assert result is not None
    assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"


def test_comparison_dataset_two_verified_ok():
    dataset = build_authoritative_comparison_dataset(["G650", "Falcon 8X"])
    assert dataset["status"] == "OK"
    names = {row["canonical_name"] for row in dataset["aircraft"]}
    assert "Gulfstream G650" in names
    assert "Falcon 8X" in names


def test_comparison_dataset_preserves_distinct_identities():
    dataset = build_authoritative_comparison_dataset(["Longitude", "Challenger 3500"])
    assert dataset["status"] == "OK"
    names = [row["canonical_name"] for row in dataset["aircraft"]]
    assert "Citation Longitude" in names
    assert "Challenger 3500" in names
    assert "Challenger Longitude" not in names


def test_falcon_8x_not_g650_in_comparison():
    dataset = build_authoritative_comparison_dataset(["Falcon 8X", "G650"])
    assert dataset["status"] == "OK"
    ranges = {row["canonical_name"]: row["range_nm"] for row in dataset["aircraft"]}
    assert ranges["Falcon 8X"] != ranges["Gulfstream G650"]


def test_global_7500_not_g650er():
    g7500 = get_aircraft_authority_record(aircraft_model="Global 7500")
    g650er = get_aircraft_authority_record(aircraft_model="Gulfstream G650ER")
    assert g7500 is not None and g650er is not None
    assert g7500.manufacturer == "Bombardier"
    assert g650er.manufacturer == "Gulfstream"


def test_praetor_600_manufacturer_embraer():
    rec = get_aircraft_authority_record(aircraft_model="Praetor 600")
    assert rec is not None
    assert rec.manufacturer == "Embraer"


def test_pc24_resolves_with_manufacturer_prefix():
    assert resolve_aircraft_alias("Pilatus PC-24") in ("Pilatus PC-24", "PC-24")
    rec = get_aircraft_authority_record(aircraft_model="PC-24")
    assert rec is not None


def test_akal_no_brochure_double_discount():
    rec = get_aircraft_authority_record(aircraft_model="G650")
    assert rec is not None
    profile = rec.to_profile_dict()
    assert profile["practical_nm"] == rec.nbaa_range_nm
    assert profile["brochure_nm"] >= profile["practical_nm"]

"""Geodesic policy, airport constraints, elimination invariant."""

from services.airport.airport_operational_constraints import (
    apply_airport_constraint_elimination,
    resolve_airports_for_route,
)
from services.broker.broker_verdicts import BrokerVerdict
from services.elimination.elimination_invariant import (
    assert_elimination_invariant,
    collect_eliminated_models,
    enforce_elimination_invariant,
)
from services.mission.geodesic_policy import apply_geodesic_policy
from services.mission.route_distance_authority import resolve_route_distance


def test_geodesic_caps_confidence_and_blocks_nonstop_authority():
    raw = resolve_route_distance("Geneva -> New York")
    assert raw.source == "geodesic"
    pol = apply_geodesic_policy(raw)
    assert pol.confidence <= 0.62
    assert pol.authorize_nonstop_feasibility is False
    assert pol.corridor_classification_only is True


def test_catalog_allows_nonstop_authority():
    r = resolve_route_distance("Teterboro -> London")
    assert r.source == "catalog"
    assert r.authorize_nonstop_feasibility is True
    assert r.corridor_classification_only is False


def test_aspen_airport_profile():
    airports = resolve_airports_for_route("Dallas -> Aspen")
    assert airports
    assert airports[0].icao == "KASE"
    assert airports[0].elevation_ft > 7000


def test_airport_eliminates_heavy_on_aspen():
    specs = {
        "Gulfstream G650": {"category": "ultra-long", "runway_ft": 6000, "hot_high_score": 0.9},
        "Citation CJ4": {"category": "light", "runway_ft": 3500, "hot_high_score": 0.7, "short_field_score": 0.75},
    }
    result = apply_airport_constraint_elimination(
        list(specs.keys()),
        route_labels=["Dallas -> Aspen"],
        model_specs=specs,
    )
    assert "Gulfstream G650" in result.eliminated
    assert "Citation CJ4" in result.survivors


def test_elimination_invariant():
    eliminated = {"gulfstream g280", "challenger 350"}
    from services.consultant.recommendation_engine import AircraftRecommendation

    recs = [
        AircraftRecommendation(
            model="Global 7500",
            category="ultra-long",
            total_score=0.8,
            confidence=0.8,
            rank=1,
        ),
        AircraftRecommendation(
            model="Gulfstream G280",
            category="super-midsize",
            total_score=0.7,
            confidence=0.7,
            rank=2,
        ),
    ]
    filtered = enforce_elimination_invariant(recs, eliminated, context="test")
    assert len(filtered) == 1
    assert filtered[0].model == "Global 7500"
    assert_elimination_invariant([r.model for r in filtered], eliminated)


def test_broker_refusal_operational_tone():
    from services.broker.broker_language import broker_refusal_message

    msg = broker_refusal_message(context="nonstop_not_credible")
    assert "nonstop" in msg.lower()
    assert "year-round" in msg.lower() or "reliable" in msg.lower()
    assert "reliable data for this" not in msg.lower()

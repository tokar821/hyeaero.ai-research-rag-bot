from rag.intent import (
    AviationIntent,
    classify_aviation_intent_json,
    classify_consultant_intent,
)
from rag.intent.aviation_classifier import classify_aviation_intent_detailed


def test_json_shape():
    out = classify_aviation_intent_json("Who owns N999XX?", None)
    assert set(out.keys()) == {"intent", "confidence"}
    assert out["intent"] == "registration_lookup"
    assert 0.0 <= out["confidence"] <= 1.0


def test_registration_tail_and_ownership():
    r = classify_aviation_intent_detailed("Tell me about tail N123AB", None)
    assert r.intent == AviationIntent.REGISTRATION_LOOKUP


def test_registration_us_n_number_formats():
    for tail in ("N123AB", "N1234", "N98765"):
        r = classify_aviation_intent_detailed(f"What is {tail}?", None)
        assert r.intent == AviationIntent.REGISTRATION_LOOKUP


def test_serial_lookup_phrase():
    r = classify_aviation_intent_detailed("MSN 525A-0349 history", None)
    assert r.intent == AviationIntent.SERIAL_LOOKUP


def test_mission_before_specs_range():
    r = classify_aviation_intent_detailed("Can a G650 fly nonstop New York to Tokyo?", None)
    assert r.intent == AviationIntent.MISSION_FEASIBILITY


def test_specs_range_without_mission():
    r = classify_aviation_intent_detailed("What is the max range of the Falcon 8X?", None)
    assert r.intent == AviationIntent.AIRCRAFT_SPECS


def test_comparison():
    r = classify_aviation_intent_detailed("Challenger 350 vs Citation Latitude for charter", None)
    assert r.intent == AviationIntent.AIRCRAFT_COMPARISON


def test_market_price_vs_for_sale():
    r1 = classify_aviation_intent_detailed("Typical asking price for a used PC-12", None)
    assert r1.intent == AviationIntent.MARKET_PRICE
    r2 = classify_aviation_intent_detailed("Is N45GX still for sale on Controller?", None)
    # Tail / registration pattern is evaluated before marketplace phrasing.
    assert r2.intent == AviationIntent.REGISTRATION_LOOKUP


def test_operator_lookup():
    r = classify_aviation_intent_detailed("Who operates the Global 7500 registered N100XF?", None)
    # Registration mark also present → registration wins (legal identity first)
    assert r.intent in (AviationIntent.REGISTRATION_LOOKUP, AviationIntent.OPERATOR_LOOKUP)


def test_consultant_classification_includes_aviation():
    ic = classify_consultant_intent("Compare Gulfstream G280 and Phenom 300E range", None)
    assert ic.aviation_intent == AviationIntent.AIRCRAFT_COMPARISON
    d = ic.asdict()
    assert d.get("aviation_intent") == "aircraft_comparison"
    assert d.get("intent") == "aircraft_comparison"

"""Strict Google Image query decision engine (deterministic JSON)."""

from __future__ import annotations

from services.image_query_decision_engine import (
    generate_ultra_precise_google_image_queries_json,
    query_violates_banned_terms,
)


def test_banned_terms_detect_jet():
    assert query_violates_banned_terms("N807JS jet cabin") is True
    assert query_violates_banned_terms("N807JS cabin interior") is False


def test_tail_cabin_queries_json():
    out = generate_ultra_precise_google_image_queries_json(
        "show me N807JS cabin please",
        required_tail=None,
        required_marketing_type=None,
        phly_rows=None,
        strict_tail_mode=False,
    )
    assert set(out.keys()) == {"queries"}
    qs = out["queries"]
    assert 3 <= len(qs) <= 5
    for q in qs:
        assert len(q.split()) <= 6
        assert "N807JS" in q
        low = q.lower()
        assert any(f in low.split() for f in ("cabin", "interior", "cockpit", "exterior"))
        assert "jet" not in low.split()
        assert "plane" not in low.split()
        assert "aircraft" not in low.split()


def test_tail_discovery_no_facet_four_queries():
    out = generate_ultra_precise_google_image_queries_json("N628TS")
    qs = out["queries"]
    assert len(qs) == 4
    facets = {q.split()[-1].lower() for q in qs}
    assert facets == {"exterior", "cabin", "interior", "cockpit"}


def test_model_cockpit_queries():
    out = generate_ultra_precise_google_image_queries_json("Gulfstream G650 cockpit")
    qs = out["queries"]
    assert qs
    assert all("cockpit" in q.lower() for q in qs)
    assert all("g650" in q.lower() or "gulfstream" in q.lower() for q in qs)


def test_gulfstream_cabin_prepends_compact_g_interior():
    """Manual SearchAPI checks: ``G650 interior`` is high-recall; keep it first."""
    out = generate_ultra_precise_google_image_queries_json(
        "Gulfstream G650 cabin interior photos"
    )
    qs = out["queries"]
    assert qs[0] == "G650 interior"
    assert "G650 cabin" in qs[:3]


def test_interior_keyword_pins_compact_query_before_marketing_variants():
    """Even with a long question, the first SearchAPI ``q`` should mirror ``G650 interior``."""
    out = generate_ultra_precise_google_image_queries_json(
        "Gulfstream G650 interior and fuel burn"
    )
    assert out["queries"][0] == "G650 interior"


def test_discovery_without_interior_keyword_leaves_exterior_first():
    out = generate_ultra_precise_google_image_queries_json("Tell me about the Gulfstream G650")
    qs = out["queries"]
    assert qs[0] == "Gulfstream G650 exterior"


def test_invalid_returns_empty():
    out = generate_ultra_precise_google_image_queries_json("Falcon 9000 cabin photos")
    assert out["queries"] == []


def test_ultra_long_range_cabin_browse_queries():
    out = generate_ultra_precise_google_image_queries_json(
        "What is the best private jet cabin for transcontinental comfort?"
    )
    qs = out["queries"]
    assert len(qs) == 5
    joined = " ".join(qs).lower()
    assert "challenger" in joined and "300" in joined
    assert "latitude" in joined
    assert "falcon" in joined and "2000" in joined
    assert "challenger" in joined and "650" in joined
    assert "global" in joined and "6000" in joined
    assert "high resolution" in joined


def test_luxury_premium_hotel_feel_triggers_large_cabin_browse():
    for phrase in ("luxury", "premium", "hotel feel"):
        out = generate_ultra_precise_google_image_queries_json(phrase)
        joined = " ".join(out["queries"]).lower()
        assert "challenger" in joined or "falcon" in joined or "global" in joined
        assert "cj2" not in joined
        assert "learjet" not in joined


def test_multi_facet_tail_one_query_per_facet():
    out = generate_ultra_precise_google_image_queries_json(
        "I'd like to see N628TS exterior, cabin, cockpit"
    )
    qs = out["queries"]
    assert qs == ["N628TS exterior", "N628TS cabin", "N628TS cockpit"]
    assert all(len(q.split()) <= 6 for q in qs)

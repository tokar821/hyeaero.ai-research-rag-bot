import re

from rag.intent.aviation_image_query_generator import generate_aviation_image_queries


def _no_forbidden(qs: list) -> bool:
    pat = re.compile(r"\b(house|hotel|home|interior design|luxury room)\b", re.I)
    return all(not pat.search(q) for q in qs)


def _all_anchored(qs: list) -> bool:
    anchor = re.compile(
        r"\b(aircraft|private jet|business jet|gulfstream|bombardier|dassault|embraer|"
        r"cessna|challenger|falcon|citation|global|legacy|g\d{2,4})\b",
        re.I,
    )
    return all(anchor.search(q) for q in qs)


def test_g650_cheaper_peers_in_queries():
    intent = {
        "intent_type": "aircraft_lookup",
        "aircraft": None,
        "category": "ultra long range",
        "visual_focus": "interior",
        "constraints": {"budget": None, "style": None, "comparison_target": "Gulfstream G650"},
    }
    out = generate_aviation_image_queries(intent, min_queries=3, max_queries=5)
    qs = out["queries"]
    assert 3 <= len(qs) <= 5
    blob = " ".join(qs).lower()
    assert "falcon 7x" in blob or "7x" in blob
    assert "global 5000" in blob or "5000" in blob
    assert "challenger 650" in blob or "650" in blob
    assert _no_forbidden(qs) and _all_anchored(qs)


def test_under_15m_budget_models():
    intent = {
        "intent_type": "interior_visual",
        "aircraft": None,
        "category": None,
        "visual_focus": "interior",
        "constraints": {"budget": 12_000_000, "style": None, "comparison_target": None},
    }
    out = generate_aviation_image_queries(intent, min_queries=3, max_queries=5)
    qs = out["queries"]
    blob = " ".join(qs).lower()
    assert "challenger 300" in blob or "falcon 2000" in blob or "legacy 450" in blob or "gulfstream iv" in blob
    assert _no_forbidden(qs) and _all_anchored(qs)


def test_explicit_aircraft_cockpit():
    intent = {
        "intent_type": "cockpit",
        "aircraft": "Citation Longitude",
        "category": "midsize",
        "visual_focus": "cockpit",
        "constraints": {"budget": None, "style": None, "comparison_target": None},
    }
    out = generate_aviation_image_queries(intent, min_queries=3, max_queries=5)
    qs = out["queries"]
    assert any("cockpit" in q.lower() or "flight deck" in q.lower() for q in qs)
    assert all("citation longitude" in q.lower() or "longitude" in q.lower() for q in qs)
    assert _no_forbidden(qs) and _all_anchored(qs)


def test_style_suffix_safe():
    intent = {
        "intent_type": "interior_visual",
        "aircraft": "Global 7500",
        "category": "ultra long range",
        "visual_focus": "interior",
        "constraints": {"budget": None, "style": "modern minimal", "comparison_target": None},
    }
    out = generate_aviation_image_queries(intent, min_queries=3, max_queries=5)
    qs = out["queries"]
    assert any("modern" in q.lower() for q in qs)
    assert _no_forbidden(qs)

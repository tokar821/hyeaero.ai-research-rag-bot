"""Tests for Query Isolation Engine (NEW vs CONTEXTUAL)."""

from rag.query_isolation_engine import isolate_query_mode


def test_new_when_explicit_new_aircraft_tail():
    h = [{"role": "user", "content": "What about N807JS?"}, {"role": "assistant", "content": "N807JS is a Citation Excel."}]
    out = isolate_query_mode("N888YG", h)
    assert out["mode"] == "NEW"
    assert out["resolved_entity"] == "N888YG"


def test_contextual_show_me_that_resolves_last_entity():
    h = [
        {"role": "user", "content": "Have you N807JS?"},
        {"role": "assistant", "content": "N807JS is a Citation Excel."},
    ]
    out = isolate_query_mode("show me that", h)
    assert out["mode"] == "CONTEXTUAL"
    assert out["resolved_entity"] == "N807JS"


def test_contextual_compare_it_resolves_last_entity():
    h = [
        {"role": "user", "content": "Tell me about Gulfstream G650"},
        {"role": "assistant", "content": "Gulfstream G650 overview."},
    ]
    out = isolate_query_mode("compare it", h)
    assert out["mode"] == "CONTEXTUAL"
    assert "G650" in (out["resolved_entity"] or "")


def test_new_for_new_concept_without_entity():
    h = [
        {"role": "user", "content": "Tell me about N807JS"},
        {"role": "assistant", "content": "N807JS is a Citation Excel."},
    ]
    out = isolate_query_mode("best cabin under $12M", h)
    assert out["mode"] == "NEW"
    assert out["resolved_entity"] is None


def test_deictic_without_thread_entity_falls_back_to_new():
    out = isolate_query_mode("show me that", None)
    assert out["mode"] == "NEW"
    assert out["resolved_entity"] is None


from __future__ import annotations

from rag.consultant_luxury_escalation import (
    apply_luxury_escalation_score_adjustments,
    interpret_luxury_escalation,
    luxury_rerank_anchor,
)


def test_something_nicer_triggers_general_escalation():
    p = interpret_luxury_escalation("show me something nicer", None)
    assert p.active
    assert p.escalate_general
    assert "global 7500" in " ".join(p.boost_terms).lower()
    anchor = luxury_rerank_anchor("show me something nicer", p)
    assert "retrieval bias" in anchor.lower()
    assert "large cabin" in anchor.lower()


def test_hotel_and_private_airline():
    p = interpret_luxury_escalation("like a hotel suite with warm lighting", None)
    assert p.hotel_vibe
    assert "divan" in [t.lower() for t in p.boost_terms]

    p2 = interpret_luxury_escalation("private airline feel wide cabin", None)
    assert p2.private_airline
    assert any("galley" in t for t in p2.boost_terms)


def test_modern_prioritizes_newer_types():
    p = interpret_luxury_escalation("more modern cabin please", None)
    assert p.modern_interior
    assert any("g500" in t for t in p.boost_terms)


def test_score_adjustment_boosts_global_over_hawker():
    p = interpret_luxury_escalation("something nicer", None)
    rows = [
        {"chunk_text": "Hawker 800XP listing vintage cabin", "score": 0.55},
        {"chunk_text": "Bombardier Global 7500 wide cabin interior", "score": 0.54},
    ]
    out = apply_luxury_escalation_score_adjustments(rows, p)
    scores = [float(r["score"]) for r in out]
    assert scores[1] > scores[0]

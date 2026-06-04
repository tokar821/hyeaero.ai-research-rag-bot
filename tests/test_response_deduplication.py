"""Phase 46 — response deduplication tests."""

from __future__ import annotations

import re

from services.truth_compression.decision_deduplicator import deduplicate_decisions_in_answer
from services.truth_compression.response_simplifier import simplify_response
from services.truth_compression.truth_synthesizer import BrokerTruthState


def test_deduplicate_removes_where_i_would_look():
    truth = BrokerTruthState(
        recommendation={"primary_recommendation": "Citation Longitude", "confidence": "HIGH"},
    )
    raw = (
        "At $12M, I would focus on G280 and Latitude equally.\n\n"
        "Where I would look:\n"
        "• G280\n"
        "• Latitude\n\n"
        "My primary recommendation would be the Citation Longitude - fits cap."
    )
    out = deduplicate_decisions_in_answer(raw, truth)
    assert "where i would look" not in out.lower()


def test_simplify_keeps_one_supporting_block():
    truth = BrokerTruthState(
        recommendation={"primary_recommendation": "G280", "confidence": "HIGH"},
    )
    raw = (
        "My primary recommendation would be the Gulfstream G280 - entry band.\n\n"
        "Supporting market context:\n"
        "• Line one\n\n"
        "Supporting market context:\n"
        "• Line two"
    )
    out = simplify_response(raw, truth)
    assert out.count("Supporting market context") == 1


def test_template_headers_removed():
    truth = BrokerTruthState(recommendation={"primary_recommendation": "G280"})
    raw = (
        "Overview\n"
        "Some text\n\n"
        "Recommendation\n"
        "More text\n\n"
        "My primary recommendation would be the Gulfstream G280 - yes."
    )
    out = simplify_response(raw, truth, pathways=["REDUNDANT_TEMPLATE_HEADERS"])
    assert not re.search(r"(?im)^overview\s*$", out)
    assert "primary recommendation" in out.lower()

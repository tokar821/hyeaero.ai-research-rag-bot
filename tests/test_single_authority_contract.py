"""Phase 46 — single authority contract enforcement tests."""

from __future__ import annotations

import re

from services.truth_compression.truth_authority_registry import (
    TruthDomain,
    layer_may_decide,
    owner_for_domain,
    violating_layer_for_phrase,
)
from services.truth_compression.truth_compression_layer import compress_ui_contract_sections
from services.truth_compression.truth_synthesizer import BrokerTruthState


def test_domain_owners():
    assert owner_for_domain(TruthDomain.INTENT) == "intent_collapse"
    assert owner_for_domain(TruthDomain.RECOMMENDATION) == "executive_broker"
    assert layer_may_decide("executive_broker", TruthDomain.RECOMMENDATION)
    assert not layer_may_decide("broker_reasoning", TruthDomain.RECOMMENDATION)


def test_forbidden_primary_phrase_for_reasoning():
    assert violating_layer_for_phrase(
        "My primary recommendation would be the G280.",
        speaking_layer="broker_decision",
    ) == "executive_broker"
    assert violating_layer_for_phrase(
        "My primary recommendation would be the G280.",
        speaking_layer="executive_broker",
    ) is None


def test_ui_contract_single_recommendation_section():
    truth = BrokerTruthState(
        recommendation={"primary_recommendation": "Citation Longitude", "confidence": "HIGH"},
    )
    contract = {
        "sections": [
            {"type": "overview", "content": "Overview text"},
            {"type": "recommendation", "content": "Consider G280 and Latitude"},
            {"type": "recommendation", "content": "Citation Longitude is best"},
            {"type": "analysis", "content": "Analysis text"},
        ],
        "render_hints": {},
    }
    out = compress_ui_contract_sections(
        contract,
        truth,
        pathways=["REDUNDANT_TEMPLATE_HEADERS"],
    )
    recs = [s for s in out["sections"] if s.get("type") == "recommendation"]
    assert len(recs) == 1
    assert "Longitude" in recs[0]["content"]
    assert out["render_hints"].get("single_authority") == "executive_broker"


def test_longitude_vs_g280_acceptance_shape():
    """Simulated committee output compresses to one executive voice."""
    from services.truth_compression.truth_compression_layer import apply_truth_compression

    du = {
        "executive_recommendation": {
            "primary_recommendation": "Citation Longitude",
            "confidence": "HIGH",
            "rationale": "Best mission fit at $12M.",
        },
        "broker_decision": {
            "direct_answer": "At $12M, I would focus on Gulfstream G280 and Citation Longitude.",
            "alternatives": [
                {"model": "Gulfstream G280"},
                {"model": "Citation Longitude"},
            ],
        },
    }
    stacked = (
        "At $12M, I would focus on Gulfstream G280 and Citation Longitude - both credible.\n\n"
        "My primary recommendation would be the Citation Longitude - best mission fit at $12M.\n\n"
        "If that does not clear diligence, I would consider:\n"
        "• Gulfstream G280 - if OEM preference shifts."
    )
    out = apply_truth_compression(stacked, query="$12M Longitude or G280?", data_used=du)
    assert out.count("My primary recommendation") == 1
    assert "Citation Longitude" in out
    assert not re.search(r"(?is)i would focus on .+ and .+ and", out)

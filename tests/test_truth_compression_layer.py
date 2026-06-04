"""Phase 46 — truth compression layer tests."""

from __future__ import annotations

import re

from services.truth_compression.redundancy_detector import detect_redundant_pathways
from services.truth_compression.truth_compression_layer import apply_truth_compression
from services.truth_compression.truth_synthesizer import synthesize_truth_state


def _stacked_answer() -> str:
    return (
        "At $12M, I would focus on Gulfstream G280, Citation Longitude, and Praetor 600.\n\n"
        "Where I would look:\n"
        "• Gulfstream G280 — entry band\n"
        "• Citation Latitude — super-mid\n\n"
        "My primary recommendation would be the Citation Longitude - best mission fit at this cap.\n\n"
        "If that does not clear diligence, I would consider:\n"
        "• Gulfstream G280 - if Gulfstream branding matters more.\n\n"
        "My primary recommendation would be the Gulfstream G280 - duplicate voice.\n\n"
        "Supporting market context:\n"
        "• Median near $22M\n\n"
        "Supporting market context:\n"
        "• Liquidity moderate"
    )


def test_apply_truth_compression_stamps_state():
    du: dict = {
        "canonical_intent_frame": {"primary_intent": "BUY", "confidence": 0.9},
        "executive_recommendation": {
            "primary_recommendation": "Citation Longitude",
            "confidence": "HIGH",
        },
    }
    out = apply_truth_compression(_stacked_answer(), data_used=du)
    assert du.get("truth_compression_applied") == 1
    assert du.get("broker_truth_state")
    assert out.count("My primary recommendation") == 1


def test_executive_leads_response():
    du = {
        "executive_recommendation": {
            "primary_recommendation": "Citation Longitude",
            "confidence": "HIGH",
        },
    }
    out = apply_truth_compression(_stacked_answer(), data_used=du)
    assert out.lower().startswith("my primary recommendation")
    assert "where i would look" not in out.lower()
    assert out.lower().count("supporting market context") <= 1


def test_redundancy_pathways_detected():
    truth = synthesize_truth_state(
        {
            "executive_recommendation": {"primary_recommendation": "Citation Longitude"},
            "broker_decision": {"direct_answer": "At $12M, I would focus on Gulfstream G280"},
        }
    )
    paths = detect_redundant_pathways(_stacked_answer(), truth)
    assert "REDUNDANT_EQUAL_WEIGHT_OPTIONS" in paths or "REDUNDANT_DECISION_MIRROR" in paths


def test_synthesize_truth_state():
    truth = synthesize_truth_state(
        {
            "canonical_intent_frame": {"primary_intent": "COMPARE", "confidence": 0.85, "ambiguity_flags": []},
            "broker_decision": {"decision_intent": "BUDGET_MATCH"},
            "executive_recommendation": {"primary_recommendation": "Gulfstream G280", "confidence": "MODERATE"},
            "adversarial": {"budget_feasibility": "FEASIBLE"},
        }
    )
    assert truth.intent["primary_intent"] == "COMPARE"
    assert truth.primary_model == "Gulfstream G280"
    assert truth.confidence > 0

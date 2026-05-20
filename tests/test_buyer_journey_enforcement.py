"""Buyer journey canonical copy (cabin shopping thread)."""

from __future__ import annotations

from rag.buyer_journey_enforcement import (
    bigger_modern_copy,
    cabin_feel_over_speed_copy,
    comparison_cockpit_copy,
    enforce_buyer_journey_answer,
    g700_vs_global_7500_copy,
    less_corporate_copy,
    modern_cabin_under_10m_copy,
)


def test_modern_cabin_under_10m():
    out = enforce_buyer_journey_answer(
        "The Gulfstream G650 has 7,000 nm range and seats 19.",
        query="Show me modern cabin under $10M.",
        data_used={"consultant_shopping_pivot": 1, "aircraft_images": [{}]},
    )
    assert "challenger 350" in out.lower()
    assert "praetor 500" in out.lower()
    assert "nm" not in out.lower()


def test_less_corporate():
    out = enforce_buyer_journey_answer(
        "Got it! For a modern cabin under $10M, consider Phenom 300.",
        query="Something less corporate.",
        data_used={
            "consultant_refinement_type": "style_shift",
            "aircraft_images": [{}],
        },
    )
    assert "praetor" in out.lower()
    assert "falcon" in out.lower()
    assert "phenom" not in out.lower()
    assert "what do you mean" not in out.lower()


def test_bigger():
    out = enforce_buyer_journey_answer(
        "How many passengers are you flying?",
        query="Bigger.",
        data_used={"consultant_refinement_type": "size_upgrade", "aircraft_images": [{}]},
    )
    lo = out.lower()
    assert "global 6000" in lo or "global" in lo
    assert "falcon 8x" in lo
    assert "g500" in lo
    assert "passengers" not in lo


def test_g700_vs_global():
    out = enforce_buyer_journey_answer(
        "G700 range is 7500 nm. Global 7500 cruises at Mach 0.90 with baggage volume...",
        query="Compare G700 vs Global 7500.",
        data_used={},
    )
    lo = out.lower()
    assert "presence" in lo or "dramatic" in lo
    assert "comfort" in lo or "refined" in lo
    assert "nm" not in lo
    assert "mach" not in lo


def test_cockpit_comparison():
    out = enforce_buyer_journey_answer(
        "Which aircraft cockpit would you like to see?",
        query="Show cockpit too.",
        data_used={
            "consultant_refinement_type": "view_change",
            "consultant_conversation_state": {
                "conversation_memory": {"comparison_target": "G700 vs Global 7500"},
            },
        },
    )
    lo = out.lower()
    assert "g700" in lo
    assert "global" in lo
    assert "which aircraft" not in lo


def test_cabin_feel_over_speed():
    out = enforce_buyer_journey_answer(
        "Mach 0.90 and runway performance favor the G700.",
        query="I care more about cabin feel than speed.",
        data_used={},
    )
    lo = out.lower()
    assert "mach" not in lo
    assert "atmosphere" in lo or "feel" in lo or "lounge" in lo


def test_canonical_snippets_exist():
    assert "Challenger 350" in modern_cabin_under_10m_copy()
    assert "Praetor" in less_corporate_copy()
    assert "Global 6000" in bigger_modern_copy()
    assert "presence" in g700_vs_global_7500_copy().lower()
    assert "futuristic" in comparison_cockpit_copy().lower()
    assert "atmosphere" in cabin_feel_over_speed_copy().lower()

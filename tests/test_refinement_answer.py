"""Refinement follow-up answer polish."""

from __future__ import annotations

from rag.refinement_answer import enforce_size_upgrade_answer, enforce_style_shift_answer
from rag.response_safety import enforce_consultant_quality


def test_style_shift_less_corporate():
    raw = "The Challenger 350 is a solid choice with a price range around $10 million."
    out = enforce_style_shift_answer(
        raw,
        query="Something less corporate.",
        data_used={
            "consultant_refinement_type": "style_shift",
            "consultant_gallery_marketing_anchor": "Challenger 350",
        },
    )
    assert "less corporate" in out.lower() or "boardroom" in out.lower() or "residential" in out.lower()


def test_style_shift_rejects_shopping_reset():
    bad = (
        "Got it! For a modern cabin under $10M, consider the Embraer Phenom 300 or Citation CJ3+."
    )
    out = enforce_style_shift_answer(
        bad,
        query="Something less corporate.",
        data_used={
            "consultant_refinement_type": "style_shift",
            "consultant_gallery_marketing_anchor": "Challenger 350",
        },
    )
    assert "phenom" not in out.lower()
    assert "challenger 350" in out.lower()


def test_size_upgrade_rejects_ulr_under_10m():
    from rag.buyer_journey_enforcement import enforce_buyer_journey_answer

    bad = "Step up to a Gulfstream G650 for maximum cabin volume under $10M."
    out = enforce_buyer_journey_answer(
        bad,
        query="Bigger.",
        data_used={
            "consultant_refinement_type": "size_upgrade",
            "consultant_conversation_state": {
                "current_budget": "$10M",
                "conversation_memory": {"active_budget_usd": 10_000_000.0},
            },
            "consultant_gallery_marketing_anchor": "Challenger 350",
            "aircraft_images": [{}],
        },
    )
    lo = out.lower()
    assert "g650" not in lo
    assert "global" in lo or "falcon" in lo


def test_good_fit_stripped_globally():
    raw = "Both jets are excellent.\n\n✅ GOOD FIT"
    out = enforce_consultant_quality(
        raw,
        query="Compare G700 vs Global 7500.",
        data_used={"consultant_response_mode": "comparison_mode"},
    )
    assert "GOOD FIT" not in out

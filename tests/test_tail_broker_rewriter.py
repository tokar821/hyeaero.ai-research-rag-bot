"""Phase 47 — tail broker mode rewrite tests."""

from __future__ import annotations

from services.market_reality.tail_broker_rewriter import rewrite_tail_investigation


def test_tail_rewriter_no_speculation_and_requests_info():
    raw = (
        "On N719GF, I would not speculate beyond what we can verify on the record and the listing package.\n\n"
        "Before treating this tail as a buy, I need:\n"
        "• Year and total time\n"
        "• Engine program status\n"
        "• Maintenance / damage history\n"
        "• The listing link or broker package\n"
    )
    out = rewrite_tail_investigation(raw, registration="N719GF")
    low = out.lower()
    assert "cannot tell you whether it is worth buying" in low
    assert "send me the listing" in low
    assert "total time" in low
    assert "engine program" in low


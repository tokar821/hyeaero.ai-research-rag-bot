"""Listing ask_price sanity filter."""

from __future__ import annotations

from rag.consultant_market_lookup import filter_listings_sane_ask_prices


def test_challenger_350_implausible_ask_dropped():
    rows = [
        {
            "manufacturer": "Bombardier",
            "model": "Challenger 350",
            "ask_price": 850_000,
            "listing_url": "https://example.com/1",
        },
        {
            "manufacturer": "Bombardier",
            "model": "Challenger 350",
            "ask_price": 18_500_000,
            "listing_url": "https://example.com/2",
        },
    ]
    out = filter_listings_sane_ask_prices(rows)
    assert len(out) == 1
    assert float(out[0]["ask_price"]) == 18_500_000

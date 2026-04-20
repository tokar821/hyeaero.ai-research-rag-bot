"""Photo-focus Tavily query must follow the latest explicit model, not stale Phly rows."""

from rag.consultant_market_lookup import build_aircraft_photo_focus_tavily_query


def test_latest_g650_wins_over_phly_challenger():
    phly = [{"manufacturer": "Bombardier", "model": "Challenger 300", "registration_number": ""}]
    q = "I want something like a G650 but cheaper. Show interior."
    hist = [{"role": "user", "content": "Challenger 300 for sale"}]
    out = build_aircraft_photo_focus_tavily_query(q, phly, hist)
    assert out
    low = out.lower()
    assert "g650" in low or "gulfstream" in low
    assert "challenger" not in low

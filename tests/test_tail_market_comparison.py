"""Tail vs market comparison answers."""

from services.broker_execution.output_governance import _guard_tail_acquisition_and_mission_answer
from services.broker_execution.tail_market_comparison import (
    is_tail_market_comparison_query,
    render_tail_market_comparison_answer,
)


def _du_n807js():
    return {
        "tail_registration": "N807JS",
        "tail_depth_mode": "comparison",
        "phly_rows": [
            {
                "registration_number": "N807JS",
                "manufacturer": "Cessna",
                "model": "Citation Excel",
                "airframe_total_time": 13910,
                "year": 2003,
                "engine_program": "MSP Gold",
                "aircraft_status": "For Sale",
                "ask_price": 3395000,
            }
        ],
        "market_reality": {"band_mid_usd": 3_200_000, "model": "Citation Excel"},
    }


def test_detects_tail_vs_market_query():
    assert is_tail_market_comparison_query("N807JS vs the market", _du_n807js())
    assert is_tail_market_comparison_query(
        "How does N807JS stack up against typical Citation Excel comps?", _du_n807js()
    )
    assert not is_tail_market_comparison_query("G280 vs Citation Longitude", {})


def test_render_market_comparison_includes_ask_and_utilization():
    out = render_tail_market_comparison_answer("N807JS vs the market", _du_n807js())
    assert "N807JS" in out
    assert "market" in out.lower()
    assert "3.39" in out or "3,395" in out or "ask" in out.lower()
    assert "utilization" in out.lower() or "13,910" in out
    assert "msp gold" in out.lower()


def test_render_market_without_phly_rows_uses_du_after_ensure(monkeypatch):
    du = {"tail_registration": "N807JS", "phly_rows": []}

    def _fake_ensure(q, data_used):
        data_used["phly_rows"] = _du_n807js()["phly_rows"]
        return True

    monkeypatch.setattr(
        "services.broker_execution.tail_fact_loader.ensure_tail_facts_for_query",
        _fake_ensure,
    )
    out = render_tail_market_comparison_answer("N807JS vs the market", du)
    assert "msp gold" in out.lower()
    assert "13,910" in out or "13910" in out


def test_guard_replaces_llm_with_market_comparison():
    llm = "Gulfstream G650 is a great comparison for N807JS on the market."
    out = _guard_tail_acquisition_and_mission_answer(
        llm,
        query="N807JS vs the market",
        data_used=_du_n807js(),
    )
    assert "broker read" in out.lower() or "ask / status" in out.lower()
    assert "g650" not in out.lower() or "n807js" in out.lower()

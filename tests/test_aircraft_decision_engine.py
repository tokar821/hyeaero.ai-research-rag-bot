"""Aircraft Decision Engine — structured verdict + scores."""

from __future__ import annotations

from unittest.mock import patch

from services.aircraft_decision_engine import (
    extract_mission_profile,
    infer_jet_class,
    public_decision_payload,
    resolve_target_aircraft,
    run_aircraft_decision_engine,
)


def test_extract_mission_profile_parses_pax_budget_nm():
    m = extract_mission_profile(
        "Should I buy a Phenom 300 for 8 pax, longest leg 1900 nm, budget $9.5m, private use"
    )
    assert m["passengers"] == 8
    assert m["longest_leg_nm"] == 1900.0
    assert m["budget_millions_usd"] == 9.5
    assert m["usage"] == "private"


def test_infer_jet_class_phenom():
    assert infer_jet_class("Embraer Phenom 300") == "light"


def test_resolve_target_aircraft_finds_model():
    mm, mfr, mdl = resolve_target_aircraft("Compare Phenom 300 vs CJ4 for charter")
    assert "Phenom" in mm or "phenom" in mm.lower()


@patch("services.aircraft_decision_engine._optional_db", return_value=None)
def test_public_decision_payload_strips_internals(_db):
    raw = run_aircraft_decision_engine("random gibberish xyzabc123", db=None)
    pub = public_decision_payload(raw)
    assert not any(k.startswith("_") for k in pub)
    assert set(pub.keys()) == {
        "aircraft",
        "verdict",
        "fit_score",
        "deal_score",
        "risk_score",
        "insight",
        "recommendation",
        "alternatives",
    }


@patch("services.aircraft_decision_engine._optional_embedding_pinecone", return_value=(None, None))
@patch("services.aircraft_decision_engine.estimate_value_hybrid")
@patch("services.aircraft_decision_engine.run_comparison")
def test_run_engine_with_mocks(mock_mc, mock_est, _emb):
    mock_mc.return_value = {"rows": [{"ask_price": 8.5e6}], "error": None}
    mock_est.return_value = {
        "estimated_value_millions": 8.2,
        "confidence_pct": 72,
        "message": None,
        "error": None,
    }
    out = public_decision_payload(
        run_aircraft_decision_engine(
            "Phenom 300 charter 6 pax 1600 nm $8m — worth buying?",
            db=object(),
        )
    )
    assert out["verdict"] in ("BUY", "CONDITIONAL BUY", "PASS")
    assert 0 <= out["fit_score"] <= 100
    assert 0 <= out["deal_score"] <= 100
    assert 0 <= out["risk_score"] <= 100
    assert isinstance(out["alternatives"], list)

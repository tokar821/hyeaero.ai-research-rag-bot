"""HyeAero orchestrated intelligence bundle (structured JSON)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.hye_aero_intelligence_engine import run_hye_aero_aircraft_intelligence


def test_invalid_model_short_circuits():
    out = run_hye_aero_aircraft_intelligence(
        "Falcon 9000 cabin",
        db=MagicMock(),
        include_visual_intel=False,
        include_market=False,
        include_valuation=False,
        include_acquisition_scores=False,
    )
    assert out["engine_status"] == "INVALID_MODEL"
    assert out["validity"]["status"] == "invalid_model"
    assert out["visual_intelligence"]["images"] == []


def test_placeholder_tail_short_circuits():
    out = run_hye_aero_aircraft_intelligence(
        "N00000",
        db=MagicMock(),
        include_visual_intel=False,
    )
    assert out["engine_status"] == "INVALID_REGISTRATION"
    assert out["visual_intelligence"]["images"] == []


@patch("services.hye_aero_intelligence_engine.run_aircraft_image_intelligence")
@patch("services.hye_aero_intelligence_engine.run_aircraft_decision_engine")
@patch("services.hye_aero_intelligence_engine.estimate_value_hybrid")
@patch("services.hye_aero_intelligence_engine.run_comparison")
@patch("services.faa_master_lookup.fetch_faa_master_owner_rows")
@patch("services.hye_aero_intelligence_engine.resolve_aircraft_identity")
def test_ok_path_with_mocks(
    mock_resolve,
    mock_faa,
    mock_mc,
    mock_est,
    mock_dec,
    mock_vis,
):
    mock_resolve.return_value = ("Cessna Citation Excel", True, "")
    mock_faa.return_value = (
        [
            {
                "n_number": "N807JS",
                "serial_number": "12345",
                "year_mfr": 2005,
                "faa_reference_model": "Cessna 560XL",
                "status_code": "V",
                "registrant_name": "Example LLC",
                "city": "Miami",
                "state": "FL",
                "country": "US",
                "type_aircraft": "Fixed wing multi engine",
            }
        ],
        "n_number_only",
    )
    mock_mc.return_value = {"rows": [], "error": None}
    mock_est.return_value = {"estimated_value_millions": 3.2, "confidence_pct": 55, "error": None}
    mock_dec.return_value = {
        "aircraft": "Cessna Citation Excel",
        "verdict": "CONDITIONAL BUY",
        "fit_score": 55,
        "deal_score": 50,
        "risk_score": 52,
        "insight": "Test",
        "recommendation": "Test",
        "alternatives": ["Citation CJ3+"],
    }
    mock_vis.return_value = {"aircraft": "N807JS", "image_type": "cabin", "images": [], "insight": "ok"}

    db = MagicMock()
    out = run_hye_aero_aircraft_intelligence(
        "N807JS interior cabin",
        db=db,
        include_visual_intel=True,
        include_market=True,
        include_valuation=True,
        include_acquisition_scores=True,
    )
    assert out["engine_status"] == "OK"
    assert out["identity"]["primary_registration"] == "N807JS"
    assert out["faa_registry_snapshot"]["match_kind"] == "n_number_only"
    assert out["faa_registry_snapshot"]["record"]["n_number"] == "N807JS"
    assert "schema_version" in out and out["schema_version"] == "1.0"

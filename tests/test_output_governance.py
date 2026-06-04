"""Output governance — single writer / LLM-primary gating."""

from __future__ import annotations

import re

import pytest

from services.broker_execution.output_governance import (
    _guard_tail_acquisition_and_mission_answer,
    apply_governed_client_answer,
    enforce_final_render_contract,
    is_llm_primary_output,
    resolve_output_governance,
)
from services.broker_execution.response_mode_classifier import ResponseMode


def test_llm_primary_skips_template_layers():
    du = {"llm_executed": True, "consultant_llm_draft": 1}
    plan = resolve_output_governance("Who owns N807JS?", du)
    assert plan.llm_primary
    assert plan.response_mode == ResponseMode.FACT_ONLY
    assert not plan.executive
    assert not plan.data_first
    assert not plan.market_reality
    assert not plan.broker_decision


def test_deterministic_tail_skips_executive_not_data_first_layer():
    """Tail registry turns use fact renderer; executive and data_first template layers stay off."""
    du = {}
    plan = resolve_output_governance("Who owns N807JS?", du)
    assert not plan.llm_primary
    assert plan.response_mode == ResponseMode.FACT_ONLY
    assert not plan.executive
    assert not plan.data_first


def test_final_contract_strips_forbidden_and_registry_blocks():
    raw = (
        "[REGISTRY FACTS]\nOwner: Acme LLC\n\n"
        "If I were buying today I would focus on records.\n\n"
        "• Owner: Acme LLC\n"
        "execution_path: tail_dispatch\n"
    )
    du = {
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "Acme LLC"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "G550"},
        ],
        "tail_registration": "N807JS",
    }
    out = enforce_final_render_contract(raw, query="Who owns N807JS?", data_used=du)
    assert "if i were buying" not in out.lower()
    assert "execution_path" not in out.lower()
    assert "[registry" not in out.lower()
    assert "acme" in out.lower()


def test_governed_pipeline_preserves_llm_prose():
    du = {"llm_executed": True}
    prose = "Owner: Acme Aviation LLC."
    out = apply_governed_client_answer(prose, query="Who owns N807JS?", data_used=du)
    assert "Acme" in out
    assert du.get("output_governance_applied") == 1
    assert du.get("model_authority_skipped_llm_primary") == 1


def test_is_llm_primary_output():
    assert is_llm_primary_output({"llm_executed": True})
    assert is_llm_primary_output({"consultant_llm_draft": 1})
    assert not is_llm_primary_output({})


@pytest.mark.parametrize(
    "phrase",
    [
        "If I were buying today",
        "operational synthesis",
        "send me the listing package",
    ],
)
def test_hygiene_strips_forbidden_on_llm_path(phrase):
    du = {"llm_executed": True}
    raw = f"• Owner: Test Co\n\n{phrase}, I would verify records."
    out = apply_governed_client_answer(raw, query="Who owns N123AB?", data_used=du)
    assert phrase.lower() not in out.lower()


def test_biggest_risks_shorthand_uses_dossier():
    q = "N807JS -> biggest risks?"
    du = {
        "tail_depth_mode": "acquisition_risks",
        "tail_registration": "N807JS",
        "phly_rows": [
            {
                "registration_number": "N807JS",
                "manufacturer": "Cessna",
                "model": "Citation Excel",
                "airframe_total_time": 13910,
                "year": 2003,
                "engine_program": "MSP Gold",
                "maintenance_tracking_program": "Cescom/CAMP",
                "aircraft_status": "For Sale",
                "ask_price": 3395000,
            }
        ],
    }
    llm = (
        "MaintenanceTracking:TheaircraftistrackedusingCescom/CAMPEnsurethereareno gaps."
    )
    out = _guard_tail_acquisition_and_mission_answer(llm, query=q, data_used=du)
    assert "maintenance tracking" in out.lower()
    assert "maintenancetracking:" not in out.lower()
    assert "•" in out or "utilization" in out.lower()


def test_route_map_guard_replaces_n807js_drift():
    from services.broker_execution.output_governance import _guard_route_visualization_answer

    llm = (
        "The Cessna Citation Excel N807JS flies Boston to Denver. "
        "Aircraft Status: For Sale Asking Price: $3,395,000"
    )
    du: dict = {"active_tail": "N807JS", "llm_executed": True}
    out = _guard_route_visualization_answer(
        llm,
        query="show route map Boston to Denver",
        data_used=du,
    )
    assert "n807js" not in out.lower() or "range map" in out.lower()
    assert "for sale" not in out.lower()
    assert du.get("consultant_visualization_svg") or "boston" in out.lower()


def test_miami_paris_18m_llm_path_replaces_ulr_list():
    q = "Miami to Paris 10 pax $18M nonstop"
    llm = (
        "Here are a few aircraft that could fit: Gulfstream G500, Bombardier Global 6000."
    )
    du = {"llm_executed": True}
    out = apply_governed_client_answer(llm, query=q, data_used=du)
    assert "g500" not in out.lower() or "budget is too low" in out.lower()


def test_miami_paris_18m_replaces_ulr_aircraft_list():
    q = "Miami to Paris 10 pax $18M nonstop"
    llm = (
        "Here are a few aircraft: Gulfstream G500, Bombardier Global 6000, Dassault Falcon 7X."
    )
    out = _guard_tail_acquisition_and_mission_answer(llm, query=q, data_used={})
    assert "g500" not in out.lower() or "budget is too low" in out.lower()
    assert "nonstop" in out.lower() or "tech stop" in out.lower() or "charter" in out.lower()

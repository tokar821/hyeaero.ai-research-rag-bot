"""Broker-style ownership / registry lookup answers."""

from services.broker_execution.tail_answer_shaper import render_registry_broker_answer
from services.broker_execution.tail_depth_mode import classify_tail_depth_mode


def test_registry_lookup_classified_as_owner():
    depth, reg = classify_tail_depth_mode("where is N7509 registered?")
    assert depth.value == "owner"
    assert reg == "N7509"


def test_owner_broker_narrative_with_bold():
    du = {
        "tail_registration": "N807JS",
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "HRL VENTURES LLC"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Cessna 560XL Citation Excel"},
            {"kind": "year", "label": "Year of manufacture", "value": "2003"},
        ],
        "faa_master_row": {"city": "Chicago", "state": "IL"},
    }
    out = render_registry_broker_answer("Who is owner of N807JS?", du)
    assert "**N807JS**" in out
    assert "**HRL VENTURES LLC**" in out
    assert "Chicago, IL" in out
    assert "2003" in out
    assert "Key registration details" in out


def test_trust_registrant_registry_style_lead():
    du = {
        "tail_registration": "N7509",
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "BANK OF UTAH TRUSTEE"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Bombardier BD-700-2A12 Global 7500"},
        ],
        "faa_master_row": {"city": "Salt Lake City", "state": "UT"},
        "tavily_llm_synthesis": {
            "operating_company_name": "VistaJet / Vista America",
            "confidence": "high",
        },
        "phly_rows": [{"base_code": "KPBI"}],
    }
    out = render_registry_broker_answer("where N7509 registered?", du)
    assert "officially registered to **BANK OF UTAH TRUSTEE**" in out
    assert "**VistaJet / Vista America**" in out
    assert "Salt Lake City, UT" in out
    assert "KPBI" in out

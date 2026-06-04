"""Final render stripper — manual Q regression samples."""

from services.broker_execution.final_render_stripper import strip_report_scaffolds
from services.broker_execution.output_governance import enforce_final_render_contract


def test_strips_mission_fit_scaffold():
    raw = (
        "Mission Fit:\n* Route: New York -> Tokyo\n* Pax: 8\n\n"
        "Aircraft Options:\n* Global 7500 - Why it fits: cabin\n\n"
        "Verdict: VIABLE WITH COMPROMISES"
    )
    out = strip_report_scaffolds(raw)
    assert "Mission Fit:" not in out
    assert "Aircraft Options:" not in out


def test_strips_broker_injection_and_dedupes():
    raw = (
        "When comparing the Gulfstream G280 and the Citation Longitude, both jets are similar.\n\n"
        "When comparing the Gulfstream G280 and the Citation Longitude, both jets are similar.\n\n"
        "Key risk: Price is only half the story.\n"
        "What I would do: Get a spec sheet."
    )
    out = strip_report_scaffolds(raw)
    assert out.count("When comparing") == 1
    assert "Key risk:" not in out
    assert "What I would do:" not in out


def test_listing_broker_checklist_removed():
    raw = (
        "At $17.9M, a Challenger 350 can be plausible.\n\n"
        "Before treating it as a bargain, I would verify:\n year and total time\n\n"
        "Inventory: Limited inventory - fewer comps.\n"
    )
    out = strip_report_scaffolds(raw)
    assert "before treating" not in out.lower()
    assert "inventory:" not in out.lower()


def test_fact_only_compact_from_facts():
    raw = (
        "The aircraft N807JS is owned by HRL Ventures LLC. "
        "FAA-registered address: HRL VENTURES LLC 11 E ADAMS ST CHICAGO. "
        "Total airframe time: 13,910 hours. Engine program MSP Gold."
    )
    du = {
        "llm_executed": True,
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "HRL Ventures LLC"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Citation Excel"},
            {"kind": "registry_status", "label": "Status", "value": "For sale"},
        ],
    }
    out = enforce_final_render_contract(raw, query="Who owns N807JS?", data_used=du)
    assert "FAA-registered" not in out
    assert "HRL" in out
    assert len(out) < 400

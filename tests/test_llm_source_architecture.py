"""LLM context must receive facts only — not report templates at the source."""

from services.broker_execution.fact_pack_builder import build_fact_pack, render_fact_pack_for_llm_context
from services.consultant.llm_explanation_layer import build_narration_system_addendum


def test_fact_pack_includes_pipeline_facts_not_templates():
    du = {
        "pipeline_llm_facts": (
            "[VERIFIED MISSION FACTS — pre-validated shortlist]\n"
            "mission_summary: 8 pax LA to Miami\n"
            "- model='Citation Latitude' broker_verdict='PRIMARY RECOMMENDATION'"
        ),
    }
    pack = build_fact_pack("8 pax LA to Miami recommend", du)
    block = render_fact_pack_for_llm_context(pack)
    assert "Citation Latitude" in block
    assert "Mission Fit:" not in block
    assert "Aircraft Options:" not in block


def test_narration_addendum_is_prose_not_scaffold():
    text = build_narration_system_addendum(query_intent="aircraft_recommendation")
    assert "Forbidden" in text or "forbidden" in text.lower()
    assert "Mission Fit, Aircraft Options, Verdict" in text
    assert "Use fixed structure" not in text

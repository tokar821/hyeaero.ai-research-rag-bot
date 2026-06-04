"""LLM-primary path must not apply full deterministic quality overrides."""

from rag.response_safety import enforce_consultant_quality


def test_llm_primary_skips_hardcoded_mission_override():
    du = {"llm_executed": True, "consultant_llm_draft": 1}
    prose = "Owner: Acme Aviation LLC."
    out = enforce_consultant_quality(
        prose,
        query="Who owns N807JS?",
        data_used=du,
    )
    assert "Acme" in out
    assert "Challenger 350" not in out
    assert du.get("consultant_quality_llm_primary_hygiene") == 1


def test_non_llm_still_applies_mission_guardrail():
    du = {}
    out = enforce_consultant_quality(
        "Draft mentioning random jets.",
        query="8 passengers LA to Miami $10M nonstop",
        data_used=du,
    )
    assert "Challenger" in out or "Latitude" in out or "G280" in out

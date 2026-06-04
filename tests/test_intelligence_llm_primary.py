from services.consultant.intelligence_engine import run_consultant_intelligence_layer


def test_intelligence_skips_structured_formatter_when_llm_primary():
    du = {"llm_executed": True, "consultant_llm_draft": 1}
    raw = "Owner: Acme LLC. Gulfstream G550 on registry."
    result = run_consultant_intelligence_layer(
        answer=raw,
        query="Who owns N807JS?",
        data_used=du,
    )
    assert result.data_used_patch.get("consultant_intelligence_llm_primary_hygiene_only") == 1
    assert "Acme" in result.answer
    assert result.data_used_patch.get("consultant_structured_formatter") is None

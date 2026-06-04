from services.broker_execution.output_governance import refresh_cached_consultant_payload


def test_cache_refresh_strips_forbidden_on_llm_cached_answer():
    payload = {
        "answer": (
            "• Owner: Acme LLC\n\n"
            "If I were buying today I would focus on maintenance records."
        ),
        "data_used": {"llm_executed": True},
    }
    out = refresh_cached_consultant_payload(payload, query="Who owns N807JS?")
    assert "if i were buying" not in (out.get("answer") or "").lower()
    assert out["data_used"].get("cached_answer_governance_refresh") == 1

from rag.consultant_query_expand import expand_consultant_research_queries


def test_rules_expand_g650_price_no_llm():
    out = expand_consultant_research_queries(
        "What's the asking price for a Gulfstream G650?",
        openai_api_key="",
        chat_model="gpt-4o-mini",
    )
    assert "tavily_query" in out and "rag_queries" in out
    assert len(out["rag_queries"]) <= 2
    tq = out["tavily_query"].lower()
    assert "gulfstream" in tq or "g650" in tq
    assert "price" in tq or "asking" in tq or "sale" in tq
    rag = " ".join(out["rag_queries"]).lower()
    assert "g650" in rag or "gulfstream" in rag


def test_rules_expand_tail_operator():
    out = expand_consultant_research_queries(
        "Who operates N123AB?",
        openai_api_key="",
        chat_model="gpt-4o-mini",
    )
    tq = out["tavily_query"]
    assert "N123AB" in tq or '"N123AB"' in tq.replace(" ", "")
    low = tq.lower()
    assert "operator" in low or "owner" in low


def test_rules_expand_uses_history_for_tail(monkeypatch):
    monkeypatch.delenv("CONSULTANT_QUERY_EXPAND_LLM", raising=False)
    out = expand_consultant_research_queries(
        "What's the range?",
        openai_api_key="sk-test",
        chat_model="gpt-4o-mini",
        history_snippet="user: Tell me about N999XX Gulfstream G550\nassistant: ...",
    )
    assert len(out["rag_queries"]) <= 2
    combined = (out["tavily_query"] + " " + " ".join(out["rag_queries"])).upper()
    assert "N999XX" in combined or "N999" in combined
    assert "G550" in combined or "GULFSTREAM" in combined
    assert "range" in " ".join(out["rag_queries"]).lower() or "range" in out["tavily_query"].lower()


def test_llm_flag_skips_rules(monkeypatch):
    calls = {"n": 0}

    def fake_llm(*a, **k):
        calls["n"] += 1
        return {"tavily_query": "LLM_TQ", "rag_queries": ["a", "b"]}

    monkeypatch.setenv("CONSULTANT_QUERY_EXPAND_LLM", "1")
    monkeypatch.setattr(
        "rag.consultant_query_expand._expand_consultant_research_queries_llm",
        fake_llm,
    )
    out = expand_consultant_research_queries(
        "test query",
        openai_api_key="x",
        chat_model="m",
    )
    assert calls["n"] == 1
    assert out["tavily_query"] == "LLM_TQ"

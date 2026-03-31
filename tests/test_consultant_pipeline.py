from rag.consultant_pipeline import (
    build_consultant_llm_context,
    load_consultant_pipeline_config,
    summarize_consultant_entities,
)


def test_context_builder_order():
    ctx, parts = build_consultant_llm_context(
        phly_authority="AUTH_BLOCK",
        market_block="MARKET",
        tavily_block="WEB",
        rag_results=[
            {"full_context": "VEC1", "chunk_text": ""},
            {"full_context": "", "chunk_text": "VEC2"},
        ],
        max_context_chars=50_000,
    )
    assert "AUTH_BLOCK" in ctx
    assert ctx.index("AUTH_BLOCK") < ctx.index("MARKET") < ctx.index("WEB")
    assert "VEC1" in ctx and "VEC2" in ctx
    assert len(parts) == 5


def test_router_config_loads():
    c = load_consultant_pipeline_config("gpt-4o-mini")
    assert c.max_rag_variants >= 1
    assert c.rag_max_chunks >= 8


def test_entity_detection_shape():
    d = summarize_consultant_entities(
        "N123AB Gulfstream",
        None,
        {"faa_lookup_tokens": ["N123AB"]},
        [{"id": 1}],
    )
    ad = d.asdict()
    assert ad["phlydata_row_count"] == 1
    assert "N123AB" in ad["lookup_tokens"] or "Gulfstream" in " ".join(ad["lookup_tokens"])

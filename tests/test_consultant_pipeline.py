from rag.consultant_pipeline import (
    build_consultant_llm_context,
    load_consultant_pipeline_config,
    summarize_consultant_entities,
)
from rag.context import build_consultant_llm_context as build_from_package
from rag.intent.schemas import AviationIntent, ConsultantIntent, IntentClassification


def test_context_builder_order():
    ctx, parts, meta = build_consultant_llm_context(
        phly_authority="AUTH_BLOCK",
        market_block="MARKET",
        tavily_block="WEB",
        rag_results=[
            {"full_context": "VEC1", "chunk_text": ""},
            {"full_context": "", "chunk_text": "VEC2"},
        ],
        max_context_chars=50_000,
        intent_classification=IntentClassification(
            primary=ConsultantIntent.GENERAL_AVIATION,
            source="test",
            aviation_intent=AviationIntent.GENERAL_QUESTION,
        ),
    )
    assert "MARKET_DATA" in ctx
    assert "OPERATIONAL_DATA" in ctx
    # Unclassified Phly lines and Tavily/RAG land in OPERATIONAL_DATA; listing block in MARKET_DATA.
    assert ctx.index("OPERATIONAL_DATA") < ctx.index("MARKET_DATA")
    assert "AUTH_BLOCK" in ctx
    assert "VEC1" in ctx and "VEC2" in ctx
    assert len(parts) >= 2
    assert meta.get("legacy_flat") is False


def test_context_builder_legacy_flat_without_intent():
    ctx, parts, meta = build_consultant_llm_context(
        phly_authority="AUTH_BLOCK",
        market_block="MARKET",
        tavily_block="WEB",
        rag_results=[{"full_context": "VEC1", "chunk_text": ""}],
        max_context_chars=50_000,
        intent_classification=None,
    )
    assert "AUTH_BLOCK" in ctx
    assert ctx.index("AUTH_BLOCK") < ctx.index("MARKET") < ctx.index("WEB")
    assert "AIRCRAFT_SPECS" not in ctx
    assert meta.get("legacy_flat") is True
    assert len(parts) == 4


def test_intent_comparison_omits_registry_and_market_sections():
    phly = """PhlyData — header
- Aircraft 1:
  - Serial: 123
[FOR USER REPLY — U.S. legal registrant (FAA MASTER) — MANDATORY VERBATIM]
  Registrant name: Secret Owner LLC
  Mailing street: 1 Hidden Rd
registration_number: N123AB
serial_number: 123
"""
    ic = IntentClassification(
        primary=ConsultantIntent.TECHNICAL_SPEC,
        source="test",
        aviation_intent=AviationIntent.AIRCRAFT_COMPARISON,
    )
    ctx, _, meta = build_consultant_llm_context(
        phly_authority=phly,
        market_block="LISTING ASK $1M",
        tavily_block="The registered owner is someone.",
        rag_results=[],
        max_context_chars=50_000,
        intent_classification=ic,
    )
    assert "REGISTRY_DATA" not in ctx
    assert "MARKET_DATA" not in ctx
    assert "Secret Owner" not in ctx
    assert "LISTING ASK" not in ctx
    assert "registered owner" not in ctx.lower()
    assert "AIRCRAFT_SPECS" in ctx
    assert "AIRCRAFT_SPECS" in meta.get("sections_included", [])


def test_registration_lookup_includes_registry_not_market():
    phly = """x
[FOR USER REPLY — U.S. legal registrant (FAA MASTER) — MANDATORY VERBATIM]
  Registrant name: Public Owner
"""
    ic = IntentClassification(
        primary=ConsultantIntent.REGISTRATION_LOOKUP,
        source="test",
        aviation_intent=AviationIntent.REGISTRATION_LOOKUP,
    )
    ctx, _, meta = build_consultant_llm_context(
        phly_authority=phly,
        market_block="SHOULD_DROP",
        tavily_block="extra",
        rag_results=[],
        max_context_chars=50_000,
        intent_classification=ic,
    )
    assert "REGISTRY_DATA" in ctx
    assert "Public Owner" in ctx
    assert "MARKET_DATA" not in ctx
    assert "SHOULD_DROP" not in ctx


def test_context_builder_same_as_context_package():
    ctx1, _, m1 = build_consultant_llm_context(
        phly_authority="A",
        market_block="",
        tavily_block="",
        rag_results=[],
        max_context_chars=100,
        intent_classification=None,
    )
    ctx2, _, m2 = build_from_package(
        phly_authority="A",
        market_block="",
        tavily_block="",
        rag_results=[],
        max_context_chars=100,
        intent_classification=None,
    )
    assert ctx1 == ctx2
    assert m1 == m2


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

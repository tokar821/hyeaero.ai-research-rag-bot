"""Hybrid retrieval: query kind + vector budget from fine intent."""

from rag.consultant_fine_intent import (
    ConsultantFineIntent,
    ConsultantFineIntentResult,
)
from rag.hybrid_retrieval import (
    HybridRetrievalQueryKind,
    build_hybrid_phly_structured_context_block,
    classify_hybrid_retrieval,
    prepend_hybrid_structured_context,
)


def test_hybrid_tail_ownership_market_structured():
    fine = ConsultantFineIntentResult(
        ConsultantFineIntent.OWNERSHIP_LOOKUP, 0.9, {"tails": ["N123AB"]}
    )
    p = classify_hybrid_retrieval("Who owns N123AB?", fine, ["N123AB"])
    assert p.kind == HybridRetrievalQueryKind.OWNERSHIP_LOOKUP
    assert p.vector_primary is False
    assert p.max_vector_chunks(18, True) == 0


def test_hybrid_mission_vector_primary():
    fine = ConsultantFineIntentResult(ConsultantFineIntent.AVIATION_MISSION, 0.9, {})
    p = classify_hybrid_retrieval("Can a G650 fly NYC to London nonstop?", fine, [])
    assert p.kind == HybridRetrievalQueryKind.MISSION_QUESTION
    assert p.vector_primary is True
    assert p.max_vector_chunks(18, True) == 18


def test_hybrid_comparison_vector_primary():
    fine = ConsultantFineIntentResult(ConsultantFineIntent.AIRCRAFT_COMPARISON, 0.9, {})
    p = classify_hybrid_retrieval("G650 vs Global 7500", fine, [])
    assert p.kind == HybridRetrievalQueryKind.COMPARISON_QUESTION
    assert p.vector_primary is True


def test_hybrid_listing_query_structured():
    fine = ConsultantFineIntentResult(ConsultantFineIntent.MARKET_QUESTION, 0.9, {})
    p = classify_hybrid_retrieval("Any Controller listings for Citation Latitude?", fine, [])
    assert p.kind == HybridRetrievalQueryKind.AIRCRAFT_LISTING_QUERY
    assert p.vector_primary is False


def test_hybrid_structured_fallback_full_vector_when_sql_empty():
    fine = ConsultantFineIntentResult(
        ConsultantFineIntent.OWNERSHIP_LOOKUP, 0.9, {"tails": ["N123AB"]}
    )
    p = classify_hybrid_retrieval("Who owns N123AB?", fine, ["N123AB"])
    assert p.max_vector_chunks(12, False) == 12


def test_build_hybrid_block_contains_tail_and_type():
    lines = build_hybrid_phly_structured_context_block(
        [
            {
                "registration_number": "N628TS",
                "serial_number": "123",
                "manufacturer": "Cessna",
                "model": "Citation XLS",
                "aircraft_status": "Available",
                "ask_price": 5_500_000,
            }
        ]
    )
    assert "N628TS" in lines
    assert "Citation XLS" in lines
    assert "Structured aircraft record" in lines


def test_prepend_only_when_structured_plan():
    fine = ConsultantFineIntentResult(ConsultantFineIntent.AIRCRAFT_SPECS, 0.9, {})
    plan = classify_hybrid_retrieval("N999AB range?", fine, ["N999AB"])
    rows = [{"registration_number": "N999AB", "model": "Falcon 7X"}]
    out = prepend_hybrid_structured_context("DETAIL_BLOCK", rows, plan)
    assert out.startswith("[HYBRID —")
    assert "DETAIL_BLOCK" in out

    plan2 = classify_hybrid_retrieval("What is a Citation CJ4?", fine, [])
    out2 = prepend_hybrid_structured_context("DETAIL_BLOCK", rows, plan2)
    assert out2 == "DETAIL_BLOCK"

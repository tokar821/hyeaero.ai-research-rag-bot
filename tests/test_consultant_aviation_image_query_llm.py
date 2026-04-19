"""LLM-assisted aviation-scoped Google Image query strings (optional, env-gated)."""

from unittest.mock import patch

import pytest

from services.consultant_aviation_image_query_llm import (
    AviationImageQueryEngineResult,
    _llm_image_query_is_safe_and_aviation,
    should_run_image_query_llm,
)
from services.consultant_image_search_orchestrator import (
    build_precision_image_search_queries,
    classify_premium_aviation_intent,
)


def test_llm_query_validator_accepts_private_jet_cabin():
    assert _llm_image_query_is_safe_and_aviation(
        "Gulfstream G650 cabin interior real photo -house -home -airbnb -hotel -wood"
    )
    assert _llm_image_query_is_safe_and_aviation(
        "Citation Latitude cabin interior private jet -house -home -airbnb -hotel -wood"
    )
    assert _llm_image_query_is_safe_and_aviation("G650 cabin interior real photo -house -hotel")


def test_llm_query_validator_rejects_bare_best_cabin():
    assert not _llm_image_query_is_safe_and_aviation("best cabin")
    assert not _llm_image_query_is_safe_and_aviation("nice interior design")


def test_llm_query_validator_tail_without_facet_still_ok_with_aviation():
    assert _llm_image_query_is_safe_and_aviation("N807JS private jet exterior")


@pytest.mark.parametrize(
    "mode,qs_len,expect_run",
    [
        ("0", 0, False),
        ("empty", 0, True),
        ("empty", 2, False),
        ("always", 3, True),
    ],
)
def test_should_run_respects_mode(monkeypatch, mode, qs_len, expect_run):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", mode)
    qs = ["a b c"] * qs_len if qs_len else []
    got = should_run_image_query_llm(
        user_query="best cabin",
        intent={"tail_number": "", "aircraft": ""},
        deterministic_queries=qs,
        required_tail=None,
        required_marketing_type=None,
        mm_for_scoring=None,
    )
    assert got is expect_run


def test_smart_runs_for_vague_cabin_without_anchor(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "1")
    assert should_run_image_query_llm(
        user_query="best cabin",
        intent={"tail_number": "", "aircraft": ""},
        deterministic_queries=["x y z"],
        required_tail=None,
        required_marketing_type=None,
        mm_for_scoring=None,
    )


def test_smart_skips_when_user_names_model(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "1")
    assert not should_run_image_query_llm(
        user_query="G650 cabin photos",
        intent={"tail_number": "", "aircraft": ""},
        deterministic_queries=["Gulfstream G650 cabin"],
        required_tail=None,
        required_marketing_type=None,
        mm_for_scoring=None,
    )


def test_smart_runs_for_similar_but_cheaper_even_with_model(monkeypatch):
    """Deterministic short queries miss nuance; LLM should still run in smart mode."""
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "1")
    assert should_run_image_query_llm(
        user_query="Show cockpit of something similar to Global 7500 but cheaper",
        intent={"tail_number": "", "aircraft": "", "image_type": "cockpit"},
        deterministic_queries=["Global 7500 cockpit"],
        required_tail=None,
        required_marketing_type=None,
        mm_for_scoring=None,
    )


def test_smart_runs_for_budget_cabin_request(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "1")
    assert should_run_image_query_llm(
        user_query="Don't explain, just show me best cabin under $15M",
        intent={"tail_number": "", "aircraft": "", "image_type": "cabin"},
        deterministic_queries=["business jet cabin"],
        required_tail=None,
        required_marketing_type=None,
        mm_for_scoring=None,
    )


def test_complexity_does_not_run_without_image_signal(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "1")
    assert not should_run_image_query_llm(
        user_query="What is a cheaper alternative to the Global 7500 on the used market",
        intent={"tail_number": "", "aircraft": ""},
        deterministic_queries=["some query here"],
        required_tail=None,
        required_marketing_type=None,
        mm_for_scoring=None,
    )


def test_build_precision_merges_llm_queries_first(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "always")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    intent = classify_premium_aviation_intent(
        "best cabin",
        required_tail=None,
        required_marketing_type=None,
        phly_rows=[],
    )

    fake_llm = [
        "Gulfstream G650 cabin interior real photo -house -home -airbnb -hotel -wood",
        "Bombardier Challenger 350 cabin interior private jet -house -home -airbnb -hotel",
    ]

    with patch(
        "services.image_query_decision_engine.generate_ultra_precise_google_image_queries_json",
        return_value={"queries": []},
    ), patch(
        "services.consultant_aviation_image_query_llm.run_aviation_image_query_engine_llm",
        return_value=AviationImageQueryEngineResult(fake_llm, 0.92, "test"),
    ):
        out, meta = build_precision_image_search_queries(
            intent,
            user_query="best cabin",
            strict_tail_mode=False,
            required_tail=None,
            required_marketing_type=None,
            phly_rows=[],
            mm_for_scoring=None,
        )
    assert out[0] == fake_llm[0]
    assert out[1] == fake_llm[1]
    assert 2 <= len(out) <= 5
    assert meta.get("image_query_engine", {}).get("confidence") == 0.92


def test_build_precision_suppresses_llm_when_confidence_low(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_IMAGE_QUERY_LLM", "always")

    intent = classify_premium_aviation_intent(
        "best cabin",
        required_tail=None,
        required_marketing_type=None,
        phly_rows=[],
    )
    weak = [
        "Gulfstream G650 cabin interior real photo -house -home -airbnb -hotel -wood",
    ]
    with patch(
        "services.image_query_decision_engine.generate_ultra_precise_google_image_queries_json",
        return_value={"queries": []},
    ), patch(
        "services.consultant_aviation_image_query_llm.run_aviation_image_query_engine_llm",
        return_value=AviationImageQueryEngineResult(weak, 0.55, "weak map"),
    ):
        out, meta = build_precision_image_search_queries(
            intent,
            user_query="best cabin",
            strict_tail_mode=False,
            required_tail=None,
            required_marketing_type=None,
            phly_rows=[],
            mm_for_scoring=None,
        )
    assert out == []
    assert meta["image_query_engine"]["suppress_gallery"] is True

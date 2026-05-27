"""Query-intent classification runs before ranked aircraft recommendation."""

from services.consultant.pre_llm_recommendation import should_run_pre_llm_pipeline
from services.recommendation.query_recommendation_intent import (
    QueryRecommendationIntent,
    classify_query_recommendation_intent,
    requires_ranked_aircraft_pipeline,
)


def test_compare_falcon_vs_praetor_is_comparison_mode():
    q = "Compare Falcon 2000LXS vs Praetor 600"
    r = classify_query_recommendation_intent(q)
    assert r.intent == QueryRecommendationIntent.AIRCRAFT_COMPARISON
    assert requires_ranked_aircraft_pipeline(r.intent)
    assert should_run_pre_llm_pipeline("aircraft_specs", q, query_intent=r.intent.value)


def test_what_aircraft_should_i_buy_is_acquisition():
    q = "What aircraft should I buy?"
    r = classify_query_recommendation_intent(q)
    assert r.intent == QueryRecommendationIntent.ACQUISITION_RECOMMENDATION
    assert r.allows_acquisition_framing
    assert should_run_pre_llm_pipeline("aircraft_recommendation", q, query_intent=r.intent.value)


def test_avoid_question_is_critique_not_acquisition():
    q = "What aircraft would you avoid?"
    r = classify_query_recommendation_intent(q)
    assert r.intent == QueryRecommendationIntent.AIRCRAFT_CRITIQUE
    assert not requires_ranked_aircraft_pipeline(r.intent)
    assert not should_run_pre_llm_pipeline("aircraft_recommendation", q, query_intent=r.intent.value)


def test_ownership_economics_skips_ranked_pipeline():
    q = "What is the direct operating cost and ownership economics of a G650?"
    r = classify_query_recommendation_intent(q)
    assert r.intent == QueryRecommendationIntent.OWNERSHIP_ECONOMICS
    assert not r.requires_ranked_pipeline


def test_visualization_skips_ranked_pipeline():
    q = "Show me interior photos of the Global 7500 cabin"
    r = classify_query_recommendation_intent(q)
    assert r.intent == QueryRecommendationIntent.VISUALIZATION_REQUEST
    assert not r.requires_ranked_pipeline


def test_range_map_beats_comparison_intent():
    q = "Show range map SFO to Paris for Falcon 8X vs G650"
    r = classify_query_recommendation_intent(q)
    assert r.intent == QueryRecommendationIntent.VISUALIZATION_REQUEST

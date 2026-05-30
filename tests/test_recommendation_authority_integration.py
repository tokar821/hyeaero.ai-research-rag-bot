"""Integration tests — recommendation authority across RAG, comparison, fallbacks."""

from __future__ import annotations

from rag.aviation_engines.context import build_aviation_engines_block
from rag.consultant_fine_intent import ConsultantFineIntent, ConsultantFineIntentResult
from services.broker.graceful_degradation import (
    degraded_empty_shortlist_guidance,
    ensure_non_empty_answer,
    safe_broker_fallback_response,
)
from services.consultant.comparison_engine import build_structured_comparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_authority import (
    EMPTY_PIPELINE_AUTHORITY_MESSAGE,
    RecommendationAuthority,
    enforce_orchestration_recommendation_authority,
    is_ranked_recommendation_query,
    reconcile_answer_with_pipeline,
    requires_recommendation_aircraft_authority,
)
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    format_understanding_first_advisory,
)


def _rec(model: str, category: str = "super-midsize") -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category=category,
        total_score=0.8,
        confidence=0.7,
        rank=1,
        fit="Strong fit",
        avoid=False,
    )


def _fine_recommendation() -> ConsultantFineIntentResult:
    return ConsultantFineIntentResult(
        intent=ConsultantFineIntent.AIRCRAFT_RECOMMENDATION,
        confidence=0.9,
        entities={"passengers": 8},
    )


# Case A — Praetor pipeline, LLM adds Caravan
def test_case_a_caravan_blocked_and_logged():
    recs = [_rec("Embraer Praetor 600")]
    du: dict = {}
    enforce_orchestration_recommendation_authority(du, "recommend aircraft Miami Caribbean 8 pax")
    du["approved_shortlist"] = ["Embraer Praetor 600"]
    du["final_ranked_aircraft"] = ["Embraer Praetor 600"]
    mission = MissionState(routes=["Miami -> Barbados"], passenger_count=8)
    llm = "Praetor 600 fits. Cessna Caravan is also popular in the islands."
    final, regen = reconcile_answer_with_pipeline(
        llm,
        mission=mission,
        recommendations=recs,
        data_used=du,
        query="recommend aircraft Miami Caribbean 8 pax",
    )
    assert regen
    assert "Caravan" not in final
    log = du.get("unauthorized_aircraft_references") or []
    assert any(
        isinstance(e, dict) and "Caravan" in (e.get("aircraft") or "")
        for e in log
    )


# Case B — empty shortlist
def test_case_b_empty_shortlist_no_aircraft_names():
    mission = MissionState(routes=["SFO -> Tokyo"], passenger_count=8)
    du: dict = {}
    enforce_orchestration_recommendation_authority(du, "recommend nonstop SFO Tokyo 8 pax")
    du["approved_shortlist"] = []
    body = degraded_empty_shortlist_guidance(
        mission,
        None,
        "recommend nonstop SFO Tokyo 8 pax",
        data_used=du,
    )
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in body
    auth = RecommendationAuthority.from_pipeline([], data_used=du)
    assert not auth.detect_unauthorized(body)


# Case C — RAG catalog King Air blocked when pipeline empty
def test_case_c_rag_catalog_suppressed_without_metadata():
    query = "recommend 8 pax NYC to LA nonstop"
    assert is_ranked_recommendation_query(query, data_used=None)
    block = build_aviation_engines_block(_fine_recommendation(), query, data_used=None)
    assert "AUTHORITY MODE" in block
    assert "King Air 350" not in block
    assert "Recommended aircraft" not in block.lower() or "do not list" in block.lower()


# Case D — comparison Praetor vs Challenger allowed
def test_case_d_explicit_comparison_models_allowed():
    mission = MissionState(routes=["NYC -> London"], passenger_count=8)
    du: dict = {}
    enforce_orchestration_recommendation_authority(
        du,
        "Compare Embraer Praetor 600 vs Bombardier Challenger 350",
    )
    du["query_recommendation_intent"] = "aircraft_comparison"
    models = ["Embraer Praetor 600", "Bombardier Challenger 350"]
    comp = build_structured_comparison(
        models,
        mission,
        locked_models_only=True,
        data_used=du,
        query="Compare Embraer Praetor 600 vs Bombardier Challenger 350",
    )
    assert len(comp.models) >= 2
    assert "Praetor" in " ".join(comp.models)
    assert "Challenger" in " ".join(comp.models)


# Case E — comparison shortlist, LLM adds Latitude
def test_case_e_unauthorized_latitude_removed():
    recs = [_rec("Embraer Praetor 600")]
    du: dict = {}
    enforce_orchestration_recommendation_authority(du, "recommend Praetor 600")
    du["approved_shortlist"] = ["Embraer Praetor 600"]
    mission = MissionState(routes=["Miami -> London"], passenger_count=8)
    llm = "Praetor 600 is best. Citation Latitude is a cheaper alternative."
    final, regen = reconcile_answer_with_pipeline(
        llm,
        mission=mission,
        recommendations=recs,
        data_used=du,
        query="recommend Praetor 600",
    )
    assert regen
    assert "Latitude" not in final
    assert "Praetor" in final


# Case F — metadata missing, intent still blocks catalog
def test_case_f_authority_active_without_orchestration_metadata():
    query = "What aircraft do you recommend for 8 passengers NYC to Paris nonstop?"
    assert requires_recommendation_aircraft_authority(None, query=query)
    du: dict = {}
    assert enforce_orchestration_recommendation_authority(du, query)
    assert du.get("pipeline_authority_enforced")
    block = build_aviation_engines_block(_fine_recommendation(), query, data_used=du)
    assert "AUTHORITY MODE" in block
    assert "filter_by_mission" not in block


def test_safe_fallback_authority_empty():
    du: dict = {}
    enforce_orchestration_recommendation_authority(du, "recommend 8 pax transatlantic")
    text = safe_broker_fallback_response(
        "recommend 8 pax transatlantic",
        mission=MissionState(routes=["NYC -> London"], passenger_count=8),
        data_used=du,
    )
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in text
    auth = RecommendationAuthority.from_pipeline([], data_used=du)
    assert not auth.detect_unauthorized(text)


def test_understanding_first_no_class_band_under_authority():
    mission = MissionState(routes=["NYC -> London"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Transatlantic executive band.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
        recommend_aircraft=True,
    )
    du: dict = {}
    enforce_orchestration_recommendation_authority(du, "recommend NYC London 8 pax")
    body = format_understanding_first_advisory(
        mission,
        pkt,
        recommendations=[],
        query="recommend NYC London 8 pax",
        data_used=du,
    )
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in body
    assert "Aircraft Class Band" not in body
    assert "Aircraft Options" not in body
    auth = RecommendationAuthority.from_pipeline([], data_used=du)
    assert not auth.detect_unauthorized(body)


def test_ensure_non_empty_strips_unauthorized_from_nonempty():
    du: dict = {}
    enforce_orchestration_recommendation_authority(du, "recommend")
    du["approved_shortlist"] = ["Embraer Praetor 600"]
    recs = [_rec("Embraer Praetor 600")]
    out = ensure_non_empty_answer(
        "Praetor 600 plus King Air 350 for short hops.",
        query="recommend",
        mission=MissionState(routes=["Miami -> Nassau"], passenger_count=6),
        recommendations=recs,
        data_used=du,
    )
    assert "King Air" not in out

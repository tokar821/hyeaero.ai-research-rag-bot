"""

Pre-LLM deterministic recommendation — run pipeline before the model generates text.



Query recommendation intent is classified first; purchase-style shortlists are not

run for critique, ownership economics, visualization, or specs-only turns.

"""



from __future__ import annotations

import re

from typing import Any, Dict, List, Optional, Tuple



from services.consultant.consultant_llm_policy import consultant_llm_narration_enabled
from services.consultant.llm_explanation_layer import build_pipeline_llm_fact_block

from services.consultant.recommendation_engine import detect_models_from_text

from services.pipeline.run_pipeline import AdvisoryPipelineResult

from services.recommendation.fit_policy import recommendation_limit_from_query

from services.recommendation.query_recommendation_intent import (

    QueryRecommendationIntent,

    QueryRecommendationIntentResult,

    apply_query_intent_metadata,

    classify_query_recommendation_intent,

    requires_ranked_aircraft_pipeline,

)

from services.recommendation.recommendation_pipeline import (

    pipeline_result_to_storage,

    run_recommendation_pipeline,

)





def should_run_pre_llm_pipeline(

    fine_intent: str,

    query: str,

    *,

    query_intent: Optional[str] = None,

    history: Optional[List[Dict[str, str]]] = None,

) -> bool:

    """

    Gate the deterministic pipeline on query recommendation intent (primary)

    and comparison model count (secondary).

    """

    del fine_intent  # legacy param retained for callers; query intent is authoritative

    if query_intent:

        try:

            qi = QueryRecommendationIntent(query_intent)

        except ValueError:

            qi = classify_query_recommendation_intent(query, history=history).intent

    else:

        qi = classify_query_recommendation_intent(query, history=history).intent



    if not requires_ranked_aircraft_pipeline(qi):

        return False

    if qi == QueryRecommendationIntent.AIRCRAFT_COMPARISON:
        mentioned = detect_models_from_text(query)
        if len(mentioned) >= 2:
            return True
        return bool(re.search(r"\bvs\.?\b|\bversus\b", (query or "").lower(), re.I))

    return True





def resolve_query_recommendation_intent(

    query: str,

    *,

    history: Optional[List[Dict[str, str]]] = None,

    data_used: Optional[Dict[str, Any]] = None,

) -> QueryRecommendationIntentResult:

    """Reuse stored intent when present; otherwise classify."""

    du = data_used if isinstance(data_used, dict) else {}

    stored = str(du.get("query_recommendation_intent") or "").strip()

    if stored:

        try:

            intent = QueryRecommendationIntent(stored)

            return QueryRecommendationIntentResult(

                intent=intent,

                confidence=float(du.get("query_recommendation_intent_confidence") or 0.8),

                source=str(du.get("query_recommendation_intent_source") or "stored"),

                requires_ranked_pipeline=requires_ranked_aircraft_pipeline(intent),

                allows_acquisition_framing=bool(du.get("query_recommendation_allows_acquisition")),

            )

        except ValueError:

            pass

    return classify_query_recommendation_intent(query, history=history)





def run_pre_llm_recommendation(

    query: str,

    *,

    conversation_state: Optional[Dict[str, Any]] = None,

    data_used: Optional[Dict[str, Any]] = None,

    fine_intent: str = "",

    history: Optional[List[Dict[str, str]]] = None,

) -> Tuple[str, Dict[str, Any], Optional[AdvisoryPipelineResult]]:

    """

    Execute deterministic pipeline and build LLM authority context.



    Returns ``(authority_block, data_used_patch, pipeline_result)``.

    """

    del fine_intent

    du: Dict[str, Any] = {}

    from services.preprocessing import attach_mission_preprocessing

    attach_mission_preprocessing(du, query)
    if isinstance(data_used, dict):
        for key in (
            "mission_preprocessing",
            "mission_preprocessing_json",
            "mission_preprocessing_meta",
        ):
            if key in du:
                data_used[key] = du[key]

    qri = resolve_query_recommendation_intent(query, history=history, data_used=data_used)

    apply_query_intent_metadata(du, qri)



    if not qri.requires_ranked_pipeline:

        from services.recommendation.query_recommendation_intent import build_intent_authority_note

        if qri.intent == QueryRecommendationIntent.VISUALIZATION_REQUEST:
            from services.consultant.mission_state import MissionState
            from services.consultant.visualization_handler import (
                build_visualization_authority_block,
                run_visualization_turn,
            )

            viz = run_visualization_turn(
                query,
                mission=MissionState(),
                history=history,
                conversation_state=conversation_state,
                data_used=du,
            )
            return build_visualization_authority_block(viz), du, None

        return build_intent_authority_note(qri), du, None



    mentioned = detect_models_from_text(query or "")

    ql = (query or "").lower()

    explicit = None

    if qri.intent == QueryRecommendationIntent.AIRCRAFT_COMPARISON and len(mentioned) >= 2:

        explicit = mentioned

    elif any(tok in ql for tok in ("compare", " versus ", " vs ", "versus")) and len(mentioned) >= 2:

        explicit = mentioned



    result, trace = run_recommendation_pipeline(

        query,

        conversation_state=conversation_state,

        data_used=du,

        explicit_candidates=explicit,

        max_results=recommendation_limit_from_query(query),

        query_intent=qri.intent.value,

    )



    fact_block = build_pipeline_llm_fact_block(
        result,
        query=query,
        query_intent=qri.intent.value,
        data_used=du,
    )

    patch: Dict[str, Any] = {

        "recommendation_decision_source": trace.decision_source,

        "recommendation_pipeline": trace.to_dict(),

        "deterministic_recommendation_pipeline": pipeline_result_to_storage(result),

        "pre_llm_pipeline_authority": 1,

        "pipeline_llm_facts": fact_block,

    }

    patch.update(du)

    try:
        from services.consultant.model_authority_guard import (
            register_mission_ranking_candidates,
            register_recovery_authority,
        )

        ranked = [r.model for r in result.recommendations if not getattr(r, "avoid", False)]
        register_mission_ranking_candidates(patch, ranked)
        register_recovery_authority(patch, ranked)
    except Exception:
        pass

    # LLM-primary path: facts flow through fact pack — not phly/RAG prefix injection.
    if consultant_llm_narration_enabled():
        return "", patch, result

    return fact_block, patch, result


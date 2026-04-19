"""Consultant retrieval orchestration.

Pipeline: :mod:`rag.conversation_guard` → fine intent (:mod:`rag.consultant_fine_intent`) →
:mod:`rag.hybrid_retrieval` (structured-first vs vector-primary) →
**aviation tool router** (registry SQL, Pinecone metadata filters) →
:mod:`rag.aviation_engines` + mission hints → **buyer psychology** (:mod:`services.buyer_psychology_engine`) →
**SQL** (``phlydata_aircraft``, ``faa_master``, listings) → **deal killer** (:mod:`services.deal_killer_engine`) when listing/Phly signals exist →
optional **Pinecone** (suppressed when structured SQL already answered the turn) →
Tavily → image stack → context builder → LLM answer.

Router env/config: :mod:`rag.consultant_pipeline`. Prompt strings: :mod:`rag.query_service` (lazy).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from rag.semantic_reranker import effective_reranker_model_name_from_env

logger = logging.getLogger(__name__)


def _fmt_q_preview(q: str, n: int = 72) -> str:
    s = (q or "").strip().replace("\n", " ")
    if not s:
        return "(empty)"
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _counter_entity_mix(results: List[Any], limit: int = 8) -> str:
    from collections import Counter

    def _et(r: Any) -> str:
        if isinstance(r, dict):
            return str(r.get("entity_type") or "?")
        return "?"

    c = Counter(_et(r) for r in (results or []))
    return ",".join(f"{k}:{v}" for k, v in c.most_common(limit))


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def run_consultant_retrieval_bundle(
    svc: Any,
    query: str,
    top_k: int,
    max_context_chars: int,
    score_threshold: Optional[float],
    history: Optional[List[Dict[str, str]]],
    progress: Any = None,
) -> Tuple[str, Any]:
    from rag.consultant_progress_log import new_progress_logger

    progress = progress or new_progress_logger()
    if progress:
        progress.step(
            "request_start",
            path="consultant_retrieval",
            q_len=len(query or ""),
            q_preview=_fmt_q_preview(query),
        )

    # 1) Rules: greetings, small talk, identity, arithmetic — no tools.
    from rag.conversation_guard import ConversationMessageType, evaluate_conversation_guard

    _api_key = getattr(svc, "openai_api_key", "") or ""
    _chat_model = (getattr(svc, "chat_model", "") or "").strip() or (
        (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()
    )

    _cg = evaluate_conversation_guard(
        query,
        history,
        openai_api_key=_api_key,
        chat_model=_chat_model,
    )
    if _cg.message_type != ConversationMessageType.AVIATION_QUERY:
        if progress:
            progress.step(
                "path_conversation_guard",
                short_circuit=1,
                conversation_message_type=_cg.message_type.value,
            )
        _du: Dict[str, Any] = {
            "consultant_conversation_guard": 1,
            "conversation_message_type": _cg.message_type.value,
        }
        if _cg.message_type == ConversationMessageType.NON_AVIATION_GENERAL:
            _du["consultant_non_aviation_general"] = 1
        return "small_talk", {
            "answer": _cg.reply or "",
            "sources": [],
            "data_used": _du,
            "aircraft_images": [],
            "error": None,
        }

    # 2) Fine intent (LLM JSON classifier or heuristic) → conversational / general short-circuits.
    from rag.aviation_tail import find_strict_tail_candidates
    from rag.consultant_fine_intent import (
        ConsultantFineIntent,
        apply_fine_intent_heuristics,
        build_consultant_tool_router,
        classify_consultant_fine_intent_llm,
        fine_intent_confidence_threshold,
        heuristic_fine_intent,
        is_conversational_fine_intent,
        llm_fine_intent_disabled,
        map_fine_intent_to_legacy_classification,
        should_run_aviation_tools,
    )
    from rag.consultant_llm_intent import generate_general_chat_reply_llm

    _strict_tails = find_strict_tail_candidates(query, history)
    _fi_thr = fine_intent_confidence_threshold()
    if llm_fine_intent_disabled() or not (_api_key or "").strip():
        _fine = apply_fine_intent_heuristics(
            heuristic_fine_intent(query, _strict_tails),
            query,
            _strict_tails,
        )
        if progress:
            progress.step(
                "consultant_fine_intent",
                intent=_fine.intent.value,
                confidence=round(_fine.confidence, 4),
                threshold=_fi_thr,
                source="heuristic",
            )
    else:
        _t_fi = time.perf_counter()
        _fine = classify_consultant_fine_intent_llm(
            query,
            history,
            api_key=_api_key,
            model=_chat_model,
        )
        if progress:
            progress.step(
                "consultant_fine_intent",
                intent=_fine.intent.value,
                confidence=round(_fine.confidence, 4),
                threshold=_fi_thr,
                source="llm",
                elapsed_ms=int((time.perf_counter() - _t_fi) * 1000),
            )

    if is_conversational_fine_intent(_fine):
        if progress:
            progress.step(
                "path_fine_intent_conversational",
                short_circuit=1,
                intent=_fine.intent.value,
            )
        if (_api_key or "").strip():
            _greply = generate_general_chat_reply_llm(
                query,
                history,
                api_key=_api_key,
                model=_chat_model,
            )
        else:
            _greply = (
                "Hello — I'm HyeAero.AI, the aviation intelligence assistant for Hye Aero.\n"
                "I can help with aircraft missions, specifications, ownership research, and market insights."
                if _fine.intent == ConsultantFineIntent.GREETING
                else "Happy to help when you're ready — HyeAero.AI for Hye Aero, missions, specs, ownership, or market."
            )
        return "small_talk", {
            "answer": _greply,
            "sources": [],
            "data_used": {
                "consultant_fine_intent": _fine.intent.value,
                "consultant_fine_intent_confidence": _fine.confidence,
                "consultant_fine_intent_conversational": 1,
            },
            "aircraft_images": [],
            "error": None,
        }

    if not should_run_aviation_tools(_fine, _fi_thr):
        if progress:
            progress.step(
                "path_general_chat_llm",
                short_circuit=1,
                reason="low_confidence_or_non_tool_intent",
                intent=_fine.intent.value,
                confidence=_fine.confidence,
            )
        if (_api_key or "").strip():
            _greply = generate_general_chat_reply_llm(
                query,
                history,
                api_key=_api_key,
                model=_chat_model,
            )
        else:
            _greply = (
                "I focus on business aviation — missions, specs, comparisons, and market context. "
                "What would you like to explore?"
            )
        return "small_talk", {
            "answer": _greply,
            "sources": [],
            "data_used": {
                "consultant_fine_intent": _fine.intent.value,
                "consultant_fine_intent_confidence": _fine.confidence,
                "consultant_fine_intent_threshold": _fi_thr,
                "consultant_fine_intent_general_chat": 1,
            },
            "aircraft_images": [],
            "error": None,
        }

    _router = build_consultant_tool_router(_fine, query, _strict_tails)
    _mission_hint = (_router.mission_reasoning_hint or "").strip()
    buyer_psychology_result: Optional[Dict[str, Any]] = None
    _aviation_engines_ctx = (_router.aviation_engines_block or "").strip()
    _consultant_aviation_prefix = "\n\n".join(
        p for p in (_mission_hint, _aviation_engines_ctx) if p
    ).strip()

    _t_block = time.perf_counter()
    prof = svc._professional_search_answer(query)
    if prof:
        if progress:
            progress.step(
                "path_professional_sql",
                elapsed_ms=int((time.perf_counter() - _t_block) * 1000),
            )
        logger.info("Professional search triggered (deterministic SQL) for query=%r", query)
        return "professional", prof


    from rag.query_service import (
        CONSULTANT_SYSTEM_PROMPT,
        _consultant_tavily_first_when_faa_ingest_miss_prefix,
        _consultant_faa_no_phly_priority_prefix,
    )

    
    from rag.consultant_query_expand import (
        expand_consultant_research_queries,
        format_tavily_payload_for_consultant,
        merge_tavily_consultant_payloads,
    )
    from rag.phlydata_consultant_lookup import (
        build_owner_operator_focus_tavily_query,
        consultant_merge_lookup_tokens,
        enrich_tavily_query_for_consultant,
    )
    from rag.consultant_image_intent import resolve_hybrid_image_gallery_intent
    from rag.consultant_suspicious_model import consultant_suspicious_aircraft_model_note
    from rag.consultant_market_lookup import (
        build_aircraft_model_photo_fallback_tavily_query,
        build_aircraft_photo_focus_tavily_query,
        build_consultant_market_authority_block,
        build_purchase_listing_tavily_query,
        consultant_wants_internal_market_sql,
        enrich_rag_queries_for_purchase,
        filter_tavily_results_for_phly_identity,
        strip_market_meta_zeros,
        tavily_price_highlights_block,
        wants_consultant_aircraft_detail_context,
        wants_consultant_aircraft_images_in_answer,
        wants_consultant_purchase_market_context,
    )
    from rag.consultant_tavily_gate import (
        empty_consultant_tavily_payload,
        should_run_consultant_tavily_after_internal,
    )
    from services.tavily_owner_hint import clamp_tavily_query, fetch_tavily_hints_for_query
    from rag.consultant_pipeline import (
        build_consultant_llm_context,
        load_consultant_pipeline_config,
        summarize_consultant_entities,
    )
    from rag.context.intent_context_policy import (
        SECTION_AIRCRAFT_SPECS,
        SECTION_MARKET_DATA,
        SECTION_REGISTRY_DATA,
    )
    from rag.answer import consultant_answer_style_suffix
    from rag.consultant_response_depth import (
        ResponseDepthKind,
        classify_response_depth,
        response_depth_prompt_suffix,
    )
    from rag.intent import ConsultantIntent
    from rag.intent.schemas import AviationIntent
    from rag.ranking import apply_structured_first_rag_order
    from rag.hybrid_retrieval import (
        HybridRetrievalPlan,
        classify_hybrid_retrieval,
        prepend_hybrid_structured_context,
    )

    hs = svc._consultant_history_snippet(history)
    hs_opt = hs.strip() or None
    
    pipe_cfg = load_consultant_pipeline_config(svc.chat_model or "")
    low_latency = pipe_cfg.low_latency
    fast_retrieval = pipe_cfg.fast_retrieval
    skip_expand = pipe_cfg.skip_expand
    single_tavily_pass = pipe_cfg.single_tavily_pass
    strict_market_sql = pipe_cfg.strict_market_sql
    tavily_per_pass = pipe_cfg.tavily_per_pass
    max_rag_variants = pipe_cfg.max_rag_variants
    enrich_rag_max = pipe_cfg.enrich_rag_max
    rag_max_chunks = pipe_cfg.rag_max_chunks
    tavily_timeout = pipe_cfg.tavily_timeout
    intent_model = pipe_cfg.intent_model

    intent_classification = map_fine_intent_to_legacy_classification(_fine.intent)
    _registry_always = (os.getenv("CONSULTANT_REGISTRY_SQL_ALWAYS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    registry_sql = bool(_router.registry_sql) or _registry_always
    pinecone_filt = _router.pinecone_filter
    av_l = intent_classification.aviation_intent.value if intent_classification.aviation_intent else None
    if progress:
        progress.step(
            "intent_classified",
            fine_intent=_fine.intent.value,
            fine_confidence=_fine.confidence,
            aviation=av_l or "-",
            primary=intent_classification.primary.value,
            query_kind=intent_classification.query_kind.value,
            intent_source=intent_classification.source,
            confidence=intent_classification.confidence,
            notes=_fmt_q_preview(intent_classification.notes, 160),
            registry_sql=registry_sql,
            pinecone_filter=_fmt_q_preview(str(pinecone_filt), 200),
        )
    logger.info(
        "Consultant intent: aviation=%s primary=%s source=%s registry_sql=%s",
        av_l,
        intent_classification.primary.value,
        intent_classification.source,
        registry_sql,
    )

    def _run_image_gallery_intent() -> Tuple[bool, str, Optional[str]]:
        return resolve_hybrid_image_gallery_intent(
            query,
            history,
            api_key=svc.openai_api_key or "",
            model=intent_model or svc.chat_model,
        )

    consultant_image_llm_intent: Optional[str] = None
    if skip_expand:
        def _run_phly() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
            return svc._phlydata_authority_block(
                query, history, registry_sql_enabled=registry_sql
            )
    
        with ThreadPoolExecutor(max_workers=2) as pre_pool:
            f_phly = pre_pool.submit(_run_phly)
            f_int = pre_pool.submit(_run_image_gallery_intent)
            phly_authority, phly_meta, phly_rows = f_phly.result()
            phly_authority = (
                ((_consultant_aviation_prefix + "\n\n") if _consultant_aviation_prefix else "")
                + _consultant_tavily_first_when_faa_ingest_miss_prefix(phly_meta)
                + _consultant_faa_no_phly_priority_prefix(phly_meta)
                + (phly_authority or "")
            )
            user_wants_gallery, consultant_image_intent_src, consultant_image_llm_intent = (
                f_int.result()
            )
        qstrip = (query or "").strip()
        expanded = {
            "tavily_query": qstrip[:400] if qstrip else "",
            "rag_queries": [qstrip] if qstrip else [""],
        }
    else:
        def _run_phly() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
            return svc._phlydata_authority_block(
                query, history, registry_sql_enabled=registry_sql
            )
    
        def _run_expand() -> Dict[str, Any]:
            return expand_consultant_research_queries(
                query,
                svc.openai_api_key or "",
                svc.chat_model,
                history_snippet=hs_opt,
            )
    
        with ThreadPoolExecutor(max_workers=3) as pre_pool:
            f_phly = pre_pool.submit(_run_phly)
            f_exp = pre_pool.submit(_run_expand)
            f_int = pre_pool.submit(_run_image_gallery_intent)
            phly_authority, phly_meta, phly_rows = f_phly.result()
            phly_authority = (
                ((_consultant_aviation_prefix + "\n\n") if _consultant_aviation_prefix else "")
                + _consultant_tavily_first_when_faa_ingest_miss_prefix(phly_meta)
                + _consultant_faa_no_phly_priority_prefix(phly_meta)
                + (phly_authority or "")
            )
            expanded = f_exp.result()
            user_wants_gallery, consultant_image_intent_src, consultant_image_llm_intent = (
                f_int.result()
            )
    if progress:
        rqs_prev = expanded.get("rag_queries") if isinstance(expanded, dict) else None
        rqs_prev = list(rqs_prev) if isinstance(rqs_prev, list) else []
        tqv = ""
        if isinstance(expanded, dict):
            tqv = str(expanded.get("tavily_query") or "").strip()
        if not tqv and not skip_expand:
            tqv = (query or "").strip()[:400]
        progress.step(
            "prefetch_parallel_done",
            skip_query_expand=skip_expand,
            phly_authority_chars=len(phly_authority or ""),
            phly_rows=len(phly_rows or []),
            faa_lookup_tokens=len((phly_meta.get("faa_lookup_tokens") or []) if phly_meta else []),
            gallery_intent=user_wants_gallery,
            expanded_tavily_q=_fmt_q_preview(tqv, 260),
            rag_variant_count=len(rqs_prev),
            rag_queries_preview=_fmt_q_preview(" | ".join(rqs_prev[:5]), 320),
        )
        if rqs_prev:
            progress.detail("pinecone_rag_queries_full", "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(rqs_prev)))

    skip_reg_market = not wants_consultant_purchase_market_context(query, history) and (
        intent_classification.primary == ConsultantIntent.REGISTRATION_LOOKUP
        or intent_classification.aviation_intent
        in (
            AviationIntent.SERIAL_LOOKUP,
            AviationIntent.OPERATOR_LOOKUP,
        )
    )
    market_block, market_meta = build_consultant_market_authority_block(
        svc.db,
        query,
        history,
        phly_rows,
        strict_market_sql=strict_market_sql,
        skip_for_registration_intent=skip_reg_market,
    )

    hybrid_plan: HybridRetrievalPlan = classify_hybrid_retrieval(query, _fine, _strict_tails)
    sql_nonempty = bool((phly_authority or "").strip()) or bool((market_block or "").strip())
    phly_authority = prepend_hybrid_structured_context(
        phly_authority or "",
        phly_rows,
        hybrid_plan,
    )
    sql_nonempty = bool((phly_authority or "").strip()) or bool((market_block or "").strip())
    vector_chunk_budget = hybrid_plan.max_vector_chunks(rag_max_chunks, sql_nonempty)

    # --- Tool usage / retrieval discipline overrides (policy enforcement) ---
    # Tail lookup must not fall back to vector DB when the user provided a tail.
    # Aircraft lookup should keep retrieved docs minimal (cap at 3).
    try:
        from rag.aviation_tail import find_visual_gallery_tail_candidates
        from rag.consultant_fine_intent import ConsultantFineIntent

        has_tail_for_policy = bool(find_visual_gallery_tail_candidates(query or "", history))
        if has_tail_for_policy and _fine.intent in (ConsultantFineIntent.OWNERSHIP_LOOKUP,):
            vector_chunk_budget = 0
        if _fine.intent in (ConsultantFineIntent.AIRCRAFT_SPECS,) and not has_tail_for_policy:
            vector_chunk_budget = max(0, min(int(vector_chunk_budget or 0), 3))
    except Exception:
        pass

    if progress:
        progress.step(
            "hybrid_retrieval",
            kind=hybrid_plan.kind.value,
            vector_primary=1 if hybrid_plan.vector_primary else 0,
            vector_chunk_budget=vector_chunk_budget,
            sql_context_nonempty=1 if sql_nonempty else 0,
        )
        progress.step(
            "internal_market_sql",
            market_block_chars=len(market_block or ""),
            skip_reg_market_sql=skip_reg_market,
            strict_market_sql=strict_market_sql,
        )
    rag_qs = enrich_rag_queries_for_purchase(
        list(expanded.get("rag_queries") or [query]),
        query,
        history,
        phly_rows,
        max_total=enrich_rag_max,
        strict_market_sql=strict_market_sql,
    )
    consultant_lookup_tokens = consultant_merge_lookup_tokens(
        query, history, phly_meta.get("faa_lookup_tokens")
    )
    entity_detection = summarize_consultant_entities(
        query,
        history,
        phly_meta,
        phly_rows,
        intent_classification,
        registry_sql_enabled=registry_sql,
    )
    if progress:
        ed = entity_detection.asdict()
        ae = ed.get("aviation_entities")
        ae_s = "-"
        if isinstance(ae, dict):
            try:
                ae_s = json.dumps(ae, ensure_ascii=False)
            except (TypeError, ValueError):
                ae_s = str(ae)
        progress.step(
            "entity_detection_bundle",
            intent_primary=ed.get("intent_primary"),
            aviation_intent=ed.get("aviation_intent"),
            intent_source=ed.get("intent_source"),
            registry_sql_enabled=ed.get("registry_sql_enabled"),
            phlydata_row_count=ed.get("phlydata_row_count"),
            lookup_tokens_n=len(ed.get("lookup_tokens") or []),
            lookup_tokens_preview=_fmt_q_preview(" ".join(ed.get("lookup_tokens") or []), 220),
            tail_candidates_preview=_fmt_q_preview(" ".join(ed.get("tail_candidates") or []), 140),
            serial_model_preview=_fmt_q_preview(" ".join(ed.get("serial_or_model_tokens") or []), 160),
            aviation_entities_json=_fmt_q_preview(ae_s, 240),
        )
    try:
        from services.buyer_psychology_engine import (
            consultant_buyer_psychology_enabled,
            run_buyer_psychology_engine,
        )

        if consultant_buyer_psychology_enabled():
            ae0 = entity_detection.aviation_entities or {}
            mlist0 = list(ae0.get("aircraft_models") or []) if isinstance(ae0, dict) else []
            buyer_psychology_result = run_buyer_psychology_engine(
                latest_query=query,
                conversation_history=history,
                detected_aircraft_interest=mlist0,
                budget_hint=None,
                mission_hint=_mission_hint or None,
                phly_rows=phly_rows,
            )
    except Exception as _bpe:
        logger.debug("buyer_psychology_engine skipped: %s", _bpe)
    tq = enrich_tavily_query_for_consultant(
        query,
        expanded.get("tavily_query") or query,
        phly_rows,
        history_snippet=hs_opt,
        lookup_tokens=consultant_lookup_tokens,
    )
    if progress:
        progress.step(
            "consultant_queries_enriched",
            rag_variants_after_purchase_enrich=len(rag_qs),
            merged_lookup_tokens=len(consultant_lookup_tokens or []),
            merged_lookup_preview=_fmt_q_preview(" ".join(consultant_lookup_tokens or []), 220),
        )
        progress.detail(
            "pinecone_rag_queries_after_enrich",
            "\n".join(f"  [{i+1}] {q}" for i, q in enumerate(rag_qs)),
        )

    tdepth: Optional[str] = None
    if (os.getenv("CONSULTANT_TAVILY_ADVANCED") or "").strip().lower() in ("1", "true", "yes"):
        tdepth = "advanced"
    
    sq = build_owner_operator_focus_tavily_query(
        query,
        phly_rows,
        history_snippet=hs_opt,
        lookup_tokens=consultant_lookup_tokens,
    )
    sq_c = " ".join(sq.split()).lower() if sq else ""
    tq_c = " ".join(tq.split()).lower()
    run_secondary = bool(sq and sq_c != tq_c)
    pq = build_purchase_listing_tavily_query(
        query, history, phly_rows, strict_market_sql=strict_market_sql
    )
    pq_c = " ".join(pq.split()).lower() if pq else ""
    merge_purchase = bool(pq and pq_c and pq_c not in {tq_c, sq_c})
    if single_tavily_pass:
        run_secondary = False
        merge_purchase = False
    img_q = build_aircraft_photo_focus_tavily_query(query, phly_rows, history)
    skip_img_pass = _env_truthy("CONSULTANT_TAVILY_SKIP_IMAGE_PASS")
    from services.searchapi_aircraft_images import searchapi_aircraft_images_enabled, searchapi_image_engine

    _searchapi_images = searchapi_aircraft_images_enabled()
    requested_tail_for_images: Optional[str] = None
    strict_tail_image_request = False
    if user_wants_gallery:
        try:
            from rag.aviation_tail import find_visual_gallery_tail_candidates

            tails_in_query = find_visual_gallery_tail_candidates(query or "", history)
            if tails_in_query:
                # Option B: tail requests require tail-verified images only (no model fallback).
                requested_tail_for_images = tails_in_query[0]
                strict_tail_image_request = True
        except Exception:
            pass
    inferred_marketing_type_for_images: Optional[str] = None
    if user_wants_gallery and not strict_tail_image_request:
        try:
            from rag.consultant_query_expand import _detect_manufacturers, _detect_models
            from services.searchapi_aircraft_images import (
                compose_manufacturer_model_phrase,
                normalize_aircraft_name,
            )

            blob = (query or "").strip()
            mans = _detect_manufacturers(blob.lower())
            mdls = _detect_models(blob)
            mt_q = compose_manufacturer_model_phrase(
                mans[0] if mans else "",
                mdls[0] if mdls else "",
            ).strip()
            mt_q = normalize_aircraft_name(mt_q) if mt_q else ""
            if not mt_q and mdls:
                mt_q = normalize_aircraft_name(mdls[0]) if mdls[0] else ""

            mt_phly = ""
            for r in phly_rows or []:
                if not isinstance(r, dict):
                    continue
                man = (r.get("manufacturer") or "").strip()
                mdl = (r.get("model") or "").strip()
                mm = compose_manufacturer_model_phrase(man, mdl).strip()
                if len(mm) >= 3:
                    mt_phly = normalize_aircraft_name(mm)
                    break

            # Prefer an explicit model mention in the **current** user text over unrelated Phly context
            # (stale rows or weak joins) so gallery SearchAPI queries stay on-type.
            if mt_q and len(mt_q) >= 3:
                inferred_marketing_type_for_images = mt_q
            elif mt_phly and len(mt_phly) >= 3:
                inferred_marketing_type_for_images = mt_phly
            else:
                inferred_marketing_type_for_images = None
        except Exception:
            inferred_marketing_type_for_images = None

    if strict_tail_image_request and requested_tail_for_images:
        # Tail-only image search query (explicit tail in the user message).
        img_q = clamp_tavily_query(
            f"\"{requested_tail_for_images}\" aircraft \"{requested_tail_for_images}\" jet "
            f"\"{requested_tail_for_images}\" plane \"{requested_tail_for_images}\" tail number "
            f"site:jetphotos.com {requested_tail_for_images} site:planespotters.net {requested_tail_for_images} "
            f"site:airliners.net {requested_tail_for_images}"
        )
    
    _t_vec = time.perf_counter()
    if vector_chunk_budget <= 0:
        results = []
        if progress:
            progress.step(
                "vector_db_retrieve_skipped",
                reason="hybrid_structured_sql_only",
                hybrid_kind=hybrid_plan.kind.value,
                sql_nonempty=1 if sql_nonempty else 0,
            )
    else:
        results = svc._retrieve_multi(
            rag_qs,
            top_k=top_k,
            score_threshold=score_threshold,
            max_results_total=vector_chunk_budget,
            max_query_variants=max_rag_variants,
            skip_rerank=fast_retrieval or low_latency,
            rerank_anchor_query=query,
            pinecone_filter=pinecone_filt,
        )
        results = apply_structured_first_rag_order(results, intent_classification.primary)
        results = svc._filter_rag_results_for_phly_aircraft(results, phly_rows)
        if progress:
            progress.step(
                "vector_db_retrieve",
                ms=int((time.perf_counter() - _t_vec) * 1000),
                chunks=len(results or []),
                skip_rerank=bool(fast_retrieval or low_latency),
                top_k=top_k,
                score_threshold=score_threshold or "",
                pinecone_filter=_fmt_q_preview(str(pinecone_filt), 200),
                chunk_entity_mix=_counter_entity_mix(results),
                rerank_anchor_preview=_fmt_q_preview(query, 120),
                hybrid_vector_budget=vector_chunk_budget,
            )
    force_tavily_always = _env_truthy("CONSULTANT_TAVILY_ALWAYS")
    run_tavily, tavily_gate_reason = should_run_consultant_tavily_after_internal(
        vector_result_count=len(results),
        sql_context_nonempty=sql_nonempty,
        force_always=force_tavily_always,
    )
    run_image_pass = bool(
        run_tavily and img_q and not skip_img_pass and user_wants_gallery and not _searchapi_images
    )
    gallery_tavily_forced = False
    if not run_tavily:
        tavily_passes = 0
        run_secondary = False
        merge_purchase = False
        run_image_pass = False
        if progress:
            progress.step(
                "tavily_skipped",
                reason=tavily_gate_reason,
                sql_nonempty=sql_nonempty,
                vector_hits=len(results or []),
                would_primary_q=_fmt_q_preview(tq, 260),
            )
        logger.info(
            "Consultant: Tavily skipped (internal SQL + vector sufficient, reason=%s)",
            tavily_gate_reason,
        )
    else:
        tavily_passes = (
            1
            + (1 if run_secondary else 0)
            + (1 if merge_purchase else 0)
            + (1 if run_image_pass else 0)
        )
        if progress:
            progress.step(
                "tavily_scheduled",
                passes=tavily_passes,
                reason=tavily_gate_reason,
                secondary=run_secondary,
                purchase_query=merge_purchase,
                image_pass=run_image_pass,
                primary_q=_fmt_q_preview(tq, 300),
                owner_q=_fmt_q_preview(sq, 240) if run_secondary and sq else "",
                listing_purchase_q=_fmt_q_preview(pq, 240) if merge_purchase and pq else "",
                photo_q=_fmt_q_preview(img_q, 240) if run_image_pass and img_q else "",
            )
        logger.debug("Consultant: Tavily run (fallback, reason=%s)", tavily_gate_reason)
    purchase_ctx = consultant_wants_internal_market_sql(
        query, history, strict=strict_market_sql
    )
    tavily_max_items = 20 if (purchase_ctx or run_image_pass) else 14
    tavily_body_chars = 2200 if purchase_ctx else 1600

    # When SearchAPI handles Bing Images, do not ask Tavily for parallel image blobs (text snippets only).
    tavily_include_images = bool(user_wants_gallery and not _searchapi_images)

    def _fetch_pri() -> Dict[str, Any]:
        return fetch_tavily_hints_for_query(
            tq,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=tavily_include_images,
        )

    def _fetch_sec() -> Optional[Dict[str, Any]]:
        if not run_secondary or not sq:
            return None
        return fetch_tavily_hints_for_query(
            sq,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=tavily_include_images,
        )

    def _fetch_pur() -> Optional[Dict[str, Any]]:
        if not merge_purchase or not pq:
            return None
        return fetch_tavily_hints_for_query(
            pq,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=tavily_include_images,
        )

    def _fetch_img() -> Optional[Dict[str, Any]]:
        if not run_image_pass or not img_q:
            return None
        return fetch_tavily_hints_for_query(
            img_q,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=tavily_include_images,
        )
    
    primary: Dict[str, Any] = {}
    secondary: Optional[Dict[str, Any]] = None
    tertiary: Optional[Dict[str, Any]] = None
    quaternary: Optional[Dict[str, Any]] = None
    _t_tv = time.perf_counter()
    max_workers_tv = 0
    if run_tavily:
        max_workers_tv = (
            1
            + (1 if run_secondary else 0)
            + (1 if merge_purchase else 0)
            + (1 if run_image_pass else 0)
        )
        max_workers_tv = max(1, min(4, max_workers_tv))
        with ThreadPoolExecutor(max_workers=max_workers_tv) as pool:
            f_pri = pool.submit(_fetch_pri)
            f_sec = pool.submit(_fetch_sec) if run_secondary else None
            f_pur = pool.submit(_fetch_pur) if merge_purchase else None
            f_img = pool.submit(_fetch_img) if run_image_pass else None
            primary = f_pri.result()
            if f_sec is not None:
                secondary = f_sec.result()
            if f_pur is not None:
                tertiary = f_pur.result()
            if f_img is not None:
                quaternary = f_img.result()
    else:
        primary = empty_consultant_tavily_payload()
    if progress:
        progress.step(
            "tavily_http_done",
            ms=int((time.perf_counter() - _t_tv) * 1000),
            run=run_tavily,
            workers=max_workers_tv,
        )
    tavily_payload = primary
    if run_secondary and secondary is not None:
        tavily_payload = merge_tavily_consultant_payloads(
            tavily_payload, secondary, max_results=14
        )
    if merge_purchase and tertiary is not None:
        tavily_payload = merge_tavily_consultant_payloads(
            tavily_payload, tertiary, max_results=18
        )
    if run_image_pass and quaternary is not None:
        tavily_payload = merge_tavily_consultant_payloads(
            tavily_payload, quaternary, max_results=20
        )
    # User asked for a gallery but Tavily was skipped (SQL + vector "sufficient") — still fetch images.
    if (
        not run_tavily
        and user_wants_gallery
        and (img_q or "").strip()
        and not skip_img_pass
        and not _searchapi_images
    ):
        try:
            img_only = fetch_tavily_hints_for_query(
                img_q,
                result_limit=tavily_per_pass,
                search_depth=tdepth,
                request_timeout=tavily_timeout,
                include_images=tavily_include_images,
            )
            tavily_payload = merge_tavily_consultant_payloads(
                tavily_payload, img_only, max_results=20
            )
            gallery_tavily_forced = True
            if progress:
                progress.step(
                    "tavily_gallery_forced",
                    reason="user_wants_gallery_while_gate_skipped_web",
                )
        except Exception as _gfe:
            logger.warning("Consultant: forced gallery-only Tavily failed: %s", _gfe)
    tavily_payload = filter_tavily_results_for_phly_identity(tavily_payload, phly_rows)
    
    purchase_tavily_merged = bool(
        run_tavily and merge_purchase and tertiary is not None
    )
    
    tavily_block = format_tavily_payload_for_consultant(
        tavily_payload, max_items=tavily_max_items, max_body_chars=tavily_body_chars
    )
    if purchase_ctx:
        ph = tavily_price_highlights_block(tavily_payload)
        if ph:
            tavily_block = f"{tavily_block}\n\n{ph}"
    tavily_hits = len(tavily_payload.get("results") or [])
    tavily_raw_imgs = tavily_payload.get("images") if isinstance(tavily_payload.get("images"), list) else []
    tavily_image_n = len(tavily_raw_imgs)
    if progress:
        progress.step(
            "tavily_formatted_for_context",
            hits=tavily_hits,
            tavily_chars=len(tavily_block or ""),
            tavily_image_candidates=tavily_image_n,
        )

    from services.consultant_aircraft_images import build_consultant_aircraft_images
    
    lr_img = market_meta.get("consultant_listing_rows_for_images") or []
    if not isinstance(lr_img, list):
        lr_img = []
    listing_urls_for_img = [
        str(r.get("listing_url")).strip()
        for r in lr_img
        if isinstance(r, dict) and (r.get("listing_url") or "").strip()
    ]
    # Photo-focused Tavily query uses quoted tail/serial; CDN URLs often omit registration in path —
    # relax URL-level identity matching when the user explicitly asked for a gallery.
    trust_tail_tavily_imgs = bool(user_wants_gallery)
    aircraft_images: List[Dict[str, Any]] = []
    image_boost_used = 0
    model_image_fallback_used = 0
    searchapi_gallery_meta: Dict[str, Any] = {}
    if user_wants_gallery:
        aircraft_images = build_consultant_aircraft_images(
            tavily_payload,
            phly_rows,
            listing_urls=listing_urls_for_img or None,
            listing_rows=lr_img or None,
            trust_tail_biased_tavily_images=trust_tail_tavily_imgs,
            required_tail=requested_tail_for_images,
            strict_tail_page_match=strict_tail_image_request,
            required_marketing_type=inferred_marketing_type_for_images,
            strict_model_title_alt_match=bool(inferred_marketing_type_for_images) and not strict_tail_image_request,
            user_query=query,
            history=history,
            gallery_meta_out=searchapi_gallery_meta,
        )
    # Tail-specific shots are sparse on CDNs — second search by make/model for representative photos.
    if (
        user_wants_gallery
        and len(aircraft_images) == 0
        and (run_tavily or gallery_tavily_forced)
        and phly_rows
        and not strict_tail_image_request
        and not _searchapi_images
    ):
        mf_q = build_aircraft_model_photo_fallback_tavily_query(phly_rows)
        if mf_q and mf_q.strip() != (img_q or "").strip():
            try:
                mf_boost = fetch_tavily_hints_for_query(
                    mf_q,
                    result_limit=tavily_per_pass,
                    search_depth=tdepth,
                    request_timeout=tavily_timeout,
                    include_images=tavily_include_images,
                )
                merged_mf = merge_tavily_consultant_payloads(
                    tavily_payload,
                    mf_boost,
                    max_results=20,
                )
                merged_mf = filter_tavily_results_for_phly_identity(merged_mf, phly_rows)
                aircraft_images = build_consultant_aircraft_images(
                    merged_mf,
                    phly_rows,
                    listing_urls=listing_urls_for_img or None,
                    listing_rows=lr_img or None,
                    trust_tail_biased_tavily_images=True,
                    user_query=query,
                    history=history,
                    gallery_meta_out=searchapi_gallery_meta,
                )
                if aircraft_images:
                    model_image_fallback_used = 1
            except Exception as mf_e:
                logger.debug("Consultant model image fallback Tavily skipped: %s", mf_e)
    # One extra Tavily call when the merged payload still yielded no images but we have a photo-biased query.
    if (
        user_wants_gallery
        and len(aircraft_images) == 0
        and (run_tavily or gallery_tavily_forced)
        and img_q
        and not skip_img_pass
        and tavily_include_images
        and not strict_tail_image_request
        and not _searchapi_images
    ):
        try:
            img_boost = fetch_tavily_hints_for_query(
                img_q,
                result_limit=tavily_per_pass,
                search_depth=tdepth,
                request_timeout=tavily_timeout,
                include_images=tavily_include_images,
            )
            merged_boost = merge_tavily_consultant_payloads(
                tavily_payload,
                img_boost,
                max_results=20,
            )
            merged_boost = filter_tavily_results_for_phly_identity(merged_boost, phly_rows)
            aircraft_images = build_consultant_aircraft_images(
                merged_boost,
                phly_rows,
                listing_urls=listing_urls_for_img or None,
                listing_rows=lr_img or None,
                trust_tail_biased_tavily_images=True,
                user_query=query,
                history=history,
                gallery_meta_out=searchapi_gallery_meta,
            )
            if aircraft_images:
                image_boost_used = 1
        except Exception as img_boost_e:
            logger.debug("Consultant image-boost Tavily pass skipped: %s", img_boost_e)
    
    has_phly = bool((phly_authority or "").strip())
    has_market = bool((market_block or "").strip())
    has_rag = bool(results)
    # SearchAPI/Bing gallery images are not Tavily hits — still count as retrievable web media for gating.
    has_tavily = tavily_hits > 0 or tavily_image_n > 0 or bool(aircraft_images)
    if not (has_phly or has_market or has_rag or has_tavily):
        if progress:
            progress.step(
                "path_general_knowledge",
                reason="no_phly_no_market_no_vector_no_tavily",
            )
        logger.info(
            "Consultant: no PhlyData, no listing/sales block, no vector hits, no Tavily → general knowledge (len=%d)",
            len(query),
        )
        return "gk", None
    
    context, _context_parts, ctx_meta = build_consultant_llm_context(
        phly_authority=phly_authority,
        market_block=market_block,
        tavily_block=tavily_block,
        rag_results=results,
        max_context_chars=max_context_chars,
        intent_classification=intent_classification,
    )
    included = set(ctx_meta.get("sections_included") or [])
    if not context.strip():
        if progress:
            progress.step("path_general_knowledge", reason="context_empty_after_assembly")
        logger.info(
            "RAG answer: no PhlyData, Tavily text, or Pinecone context; general knowledge (len=%d)",
            len(query),
        )
        return "gk", None

    if progress:
        progress.step(
            "llm_context_ready",
            ctx_chars=len(context),
            sections=",".join(sorted(included)),
            tokens_est=ctx_meta.get("context_tokens_est"),
            has_phly_sql=has_phly,
            has_market_sql=has_market,
            rag_chunks=len(results or []),
            tavily_hits=tavily_hits,
            gallery_images=len(aircraft_images),
        )

    data_used: Dict[str, Any] = dict(phly_meta)
    for k, v in strip_market_meta_zeros(market_meta).items():
        data_used[k] = v
    if low_latency:
        data_used["consultant_low_latency"] = 1
    if fast_retrieval:
        data_used["consultant_fast_retrieval"] = 1
    if skip_expand:
        data_used["consultant_skip_query_expand"] = 1
    if single_tavily_pass:
        data_used["consultant_tavily_single_pass"] = 1
    if strict_market_sql:
        data_used["consultant_market_sql_strict"] = 1
    data_used["consultant_pipeline"] = "hybrid_sql_phly_vector_rag_tavily_context_llm_v3"
    data_used["hybrid_retrieval_kind"] = hybrid_plan.kind.value
    data_used["hybrid_vector_primary"] = 1 if hybrid_plan.vector_primary else 0
    data_used["hybrid_vector_chunk_budget"] = vector_chunk_budget
    data_used["consultant_query_router"] = pipe_cfg.query_router_snapshot()
    data_used["consultant_entity_detection"] = entity_detection.asdict()
    data_used["consultant_intent"] = intent_classification.asdict()
    data_used["consultant_fine_intent"] = _fine.intent.value
    data_used["consultant_fine_intent_confidence"] = _fine.confidence
    data_used["consultant_fine_pinecone_filter"] = _fmt_q_preview(str(pinecone_filt), 200)
    if _aviation_engines_ctx:
        data_used["consultant_aviation_engines_context"] = 1
        data_used["consultant_aviation_engines_chars"] = len(_aviation_engines_ctx)
    if progress:
        data_used["consultant_progress_id"] = progress.request_id
    data_used["consultant_context_sections"] = sorted(included)
    data_used["consultant_context_tokens_est"] = ctx_meta.get("context_tokens_est")
    data_used["consultant_context_char_budget"] = ctx_meta.get("context_char_budget")
    data_used["consultant_context_legacy_flat"] = 1 if ctx_meta.get("legacy_flat") else 0
    data_used["registry_sql_enabled"] = 1 if registry_sql else 0
    if skip_reg_market:
        data_used["consultant_market_sql_skipped_registration_intent"] = 1
    data_used["consultant_fast_mode"] = (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    data_used["tavily_results"] = tavily_hits
    data_used["tavily_web_query_passes"] = tavily_passes
    if run_image_pass:
        data_used["tavily_image_focus_pass"] = 1
    if image_boost_used:
        data_used["tavily_image_boost_pass"] = 1
    if model_image_fallback_used:
        data_used["tavily_model_image_fallback_pass"] = 1
        data_used["consultant_aircraft_images_representative"] = 1
    data_used["tavily_gate_reason"] = tavily_gate_reason
    if force_tavily_always:
        data_used["consultant_tavily_always"] = 1
    if not run_tavily:
        data_used["tavily_skipped"] = 1
    if gallery_tavily_forced:
        data_used["consultant_tavily_gallery_forced"] = 1
    if purchase_tavily_merged:
        data_used["tavily_purchase_focus"] = 1
    data_used["tavily_error"] = tavily_payload.get("error")
    data_used["rag_query_variants"] = len(rag_qs)
    if svc._rerank_enabled_globally() and not (fast_retrieval or low_latency):
        data_used["rag_semantic_rerank"] = 1
        data_used["rag_rerank_model"] = effective_reranker_model_name_from_env()
        if results and any(r.get("rerank_score") is not None for r in results):
            data_used["rag_rerank_applied"] = 1
    else:
        data_used["rag_semantic_rerank"] = 0
    for r in results:
        et = (r.get("entity_type") or "other").replace("_", " ")
        data_used[et] = data_used.get(et, 0) + 1
    
    data_used["aircraft_images"] = aircraft_images
    data_used["consultant_aircraft_image_count"] = len(aircraft_images)
    if searchapi_gallery_meta:
        for _gk, _gv in searchapi_gallery_meta.items():
            if _gv is not None:
                data_used[_gk] = _gv
    if (
        user_wants_gallery
        and strict_tail_image_request
        and _searchapi_images
        and not aircraft_images
    ):
        # Part 3: strict tail + SearchAPI produced no title/URL-confirmed matches for that exact mark.
        data_used["consultant_strict_tail_no_confirmed_searchapi_images"] = 1
    if _searchapi_images:
        data_used["consultant_searchapi_images_enabled"] = 1
        data_used["consultant_searchapi_image_engine"] = searchapi_image_engine()
    # Internal join helper for image lookup — not needed by clients; keeps responses smaller.
    data_used.pop("consultant_listing_rows_for_images", None)
    if wants_consultant_aircraft_images_in_answer(query):
        data_used["consultant_user_asked_photos"] = 1
    if user_wants_gallery:
        data_used["consultant_show_image_ui_context"] = 1
        try:
            from services.aviation_intelligence_protocol import build_aviation_intelligence_envelope

            data_used["aviation_intelligence_envelope"] = build_aviation_intelligence_envelope(
                user_query=query,
                user_wants_gallery=True,
                phly_rows=phly_rows,
                aircraft_images=aircraft_images,
            )
        except Exception as _env_e:
            logger.debug("aviation_intelligence_envelope skipped: %s", _env_e)
    try:
        from services.image_answer_alignment_engine import (
            build_image_answer_alignment_plan,
            consultant_image_answer_alignment_enabled,
            format_alignment_block_for_layered_context,
        )

        if consultant_image_answer_alignment_enabled() and user_wants_gallery and aircraft_images:
            _gm_align = {
                k: data_used[k]
                for k in (
                    "consultant_premium_intent",
                    "image_query_engine",
                    "image_rank_filter_engine",
                )
                if data_used.get(k) not in (None, "", False, {})
            }
            _plan = build_image_answer_alignment_plan(
                user_query=query,
                aircraft_images=aircraft_images,
                phly_rows=phly_rows,
                gallery_meta=_gm_align,
                marketing_type_hint=inferred_marketing_type_for_images,
            )
            data_used["image_answer_alignment"] = _plan
            _align_ctx = format_alignment_block_for_layered_context(_plan)
            if _align_ctx:
                context = f"{(context or '').rstrip()}\n\n---\n\n**[IMAGE ANSWER ALIGNMENT CONTEXT]**\n\n{_align_ctx}".strip()
    except Exception as _ial_e:
        logger.debug("image_answer_alignment skipped: %s", _ial_e)
    if buyer_psychology_result is not None:
        data_used["buyer_psychology"] = buyer_psychology_result
    try:
        from services.deal_killer_engine import (
            consultant_deal_killer_enabled,
            run_deal_killer_from_consultant_context,
        )

        if consultant_deal_killer_enabled():
            _dk = run_deal_killer_from_consultant_context(
                phly_rows=phly_rows,
                primary_listing=data_used.get("consultant_primary_listing_for_deal_review"),
                query=query,
                buyer_psychology=data_used.get("buyer_psychology"),
                db=svc.db,
            )
            if _dk:
                data_used["deal_killer"] = _dk
    except Exception as _dke:
        logger.debug("deal_killer_engine skipped: %s", _dke)
    try:
        from services.aircraft_decision_engine import (
            consultant_query_requests_aircraft_decision,
            public_decision_payload,
            run_aircraft_decision_engine,
        )

        if consultant_query_requests_aircraft_decision(query):
            _dec = run_aircraft_decision_engine(query, db=svc.db)
            data_used["aircraft_decision"] = public_decision_payload(_dec)
    except Exception as _dec_e:
        logger.debug("aircraft_decision skipped: %s", _dec_e)
    data_used["consultant_image_intent_source"] = consultant_image_intent_src
    if consultant_image_llm_intent is not None:
        data_used["consultant_image_llm_intent"] = consultant_image_llm_intent

    _response_depth_kind = classify_response_depth(query, history, _fine.intent)
    data_used["consultant_response_depth"] = _response_depth_kind.value

    system_prompt = CONSULTANT_SYSTEM_PROMPT + consultant_answer_style_suffix(
        intent_classification.primary,
        intent_classification.aviation_intent,
    )
    # User-facing safety: never expose internal datasets/pipelines; treat bracketed context tags as internal-only.
    system_prompt += (
        "\n\n**Client-facing safety (strict):** Never mention or imply internal systems, datasets, or infrastructure. "
        "Do not say or reference: phlydata, database, internal records, internal database, internal snapshot, RAG, Pinecone, "
        "vector search, SQL, faa_master table, Controller, Aircraft Exchange, scrapes, pipelines, or table names. "
        "If the context contains bracketed tags (e.g. [AUTHORITATIVE ...], [FOR USER REPLY ...]), treat them as "
        "internal labels and do not repeat them.\n"
        "Attribution: use *typical operational performance for this class* / *from a broker’s perspective* when answering from general knowledge. "
        "Use *per the aircraft record in this brief* / listing-style phrasing **only** when the context for this turn actually includes those facts—never imply registry or market sourcing otherwise.\n"
        "If this thread already had a short fallback reply, do not repeat the same stock wording—vary or shorten.\n"
        "Recommendations: short conversational opening first—no essay or spec dump unless the user asks. "
        "Avoid philosophical aviation one-liners. Do not list serials/regs as the main thrust unless detailed data was requested.\n"
    )

    system_prompt += response_depth_prompt_suffix(_response_depth_kind)

    if user_wants_gallery or tavily_image_n > 0 or aircraft_images:
        system_prompt += (
            "\n\n**Web search / images (never contradict the product):** If this turn used Tavily/web context, "
            "returned image candidates, or the app is showing a gallery, you must **not** say you cannot perform "
            "web searches, cannot access external sites, or cannot show images, or that only the user can look "
            "up photos elsewhere. Public web material was already retrieved for this reply; the UI may list "
            "**Aircraft images** with Open links. Acknowledge that gallery when relevant. If the user asked for "
            "Google-style photos, describe curated in-app image results when present. If a shown image might be "
            "mislabeled vs. the tail in question, say third-party sites sometimes confuse similar registrations — "
            "not that you are unable to help."
            "\n\n**Photos vs live tracking:** A request to **see** / **show** the aircraft means **photos or the in-app "
            "gallery**, not real-time flight position. Do **not** answer with only FlightAware or Flightradar24 "
            "**tracking or current location** as a substitute for pictures, and do **not** claim you cannot provide "
            "visuals because they would need to be \"real-time\"."
            "\n\n**Image display enforcement (hard):** If images are available this turn, you must not output any refusal "
            "like \"I can't show images\" / \"I don't have photos\" / \"I cannot provide images\". Treat images as shown "
            "in the app and write a normal consultant response around that fact."
            "\n\n**Engine vs consultant:** You do **not** decide whether images exist. If `aircraft_images` are present in "
            "this response, they were retrieved by the system engine and are available to the user. Do not contradict "
            "that by saying you \"don't have\" photos."
        )

    _susp_model = consultant_suspicious_aircraft_model_note(query)
    if _susp_model:
        system_prompt += (
            f"\n\n**Likely non-existent aircraft name:** {_susp_model} "
            "State this clearly in plain language. **Do not** invent specifications, listings, or "
            "verified photos for the fake model string; suggest the closest real variants in short bullets."
        )

    # --- Reasoning / response-mode router (does not affect retrieval) ---
    try:
        from rag.aviation_tail import find_visual_gallery_tail_candidates
        from rag.consultant_response_mode import (
            ConsultantResponseMode,
            classify_consultant_response_mode,
            response_mode_prompt_suffix,
        )
        from rag.consultant_validity import validate_aircraft_model

        has_tail = bool(find_visual_gallery_tail_candidates(query or "", history))
        # Use "visual" phrasing as a signal even when gallery intent routing was not triggered upstream.
        has_visual_intent = bool(re.search(r"\b(show\s+me|can\s+i\s+see|any\s+photos?|pictures?|what\s+does\s+it\s+look)\b", (query or ""), re.I))
        v = validate_aircraft_model(query or "")
        mode: ConsultantResponseMode = classify_consultant_response_mode(
            query=query or "",
            fine_intent=str(getattr(_fine.intent, "value", _fine.intent)),
            has_tail=has_tail,
            has_visual_intent=bool(user_wants_gallery or has_visual_intent),
            suspicious_model_note=_susp_model or (v.message if v else None),
        )
        data_used["consultant_response_mode"] = mode.value
        system_prompt += response_mode_prompt_suffix(mode)

        # Internal confidence tag (not user-facing UI): high when tail + authoritative record, else medium/low.
        conf = "low"
        if has_tail:
            if bool(phly_authority) or int((phly_meta or {}).get("faa_master_owner_rows") or 0):
                conf = "high"
            else:
                conf = "medium"
        else:
            if mode in (ConsultantResponseMode.COMPARISON, ConsultantResponseMode.MISSION_ADVISORY, ConsultantResponseMode.STRATEGIC_OWNERSHIP):
                conf = "medium"
        data_used["consultant_confidence"] = conf
    except Exception:
        pass

    # Format requirement for structured retrieval responses (tail/serial/ownership/market).
    if hybrid_plan.kind.value in (
        "tail_number_lookup",
        "serial_number_lookup",
        "ownership_lookup",
        "aircraft_price_lookup",
        "aircraft_listing_query",
    ):
        if _response_depth_kind == ResponseDepthKind.CONFIRMATION:
            system_prompt += (
                "\n\n**Structured format (this turn):** The user is confirming prior information — "
                "answer briefly. **Do not** emit the full **Aircraft Record** template unless they explicitly ask for a full recap."
            )
        elif _response_depth_kind == ResponseDepthKind.AIRCRAFT_LOOKUP:
            system_prompt += (
                "\n\n**Required output format (structured aircraft queries):** "
                "Use plain-text sections: **Aircraft Overview**, **Key Specs**, **Ownership** (and **Market** when listing/price is central). "
                "Omit empty sections. Tight bullets. When context includes mandatory verbatim pricing/status or registrant lines, honor those within the sections — "
                "describe them in plain language—do not bury them in a field dump.\n"
                "If a listing is referenced, describe it as \"current aircraft marketplace listings\" without naming vendors.\n"
            )
        else:
            system_prompt += (
                "\n\n**Required output format (structured aircraft queries):**\n"
                "Aircraft Record:\n"
                "Tail Number: <NXXXX or (unknown)>\n"
                "Aircraft Type: <make/model>\n"
                "Year: <year or (unknown)>\n"
                "Status: <for sale / not for sale / unknown>\n"
                "\n"
                "Registrant:\n"
                "Name: <name or (unknown)>\n"
                "Location: <city, state, country or (unknown)>\n"
                "\n"
                "If a listing is referenced, describe it as \"current aircraft marketplace listings\" without naming vendors.\n"
            )
    if phly_authority and not ctx_meta.get("legacy_flat"):
        system_prompt += (
            "\n\n**Context layout:** Retrieved facts are grouped into **AIRCRAFT_SPECS**, **OPERATIONAL_DATA**, "
            "**MARKET_DATA**, and **REGISTRY_DATA**. Only subsections present in the user message apply. "
            "**Mandatory verbatim** rules for Phly pricing/status or FAA registrant apply **only** when "
            "**MARKET_DATA** or **REGISTRY_DATA** (respectively) appears in context for this turn."
        )
    if phly_authority and not registry_sql:
        system_prompt += (
            "\n\n**Registry SQL:** For this turn, **faa_master** was not queried (non–registration_lookup intent). "
            "If the user needs U.S. legal registrant facts and they are absent from PhlyData lines, use **Tavily/vector** "
            "in context and label sources — do not imply an FAA block was omitted in error."
        )
    if phly_authority:
        if ctx_meta.get("legacy_flat"):
            system_prompt += (
                "\n\nThe context may begin with an AUTHORITATIVE **PhlyData (Hye Aero aircraft source) + FAA MASTER** block. "
                "That block is Hye Aero's **canonical internal record**: identity, internal snapshot fields (status, ask-as-exported, programs, etc.), and legal U.S. registrant when present — **all override** web or vector for those fields. "
                "Listing/market rows elsewhere are **not** PhlyData — use them as **supplemental** context after PhlyData; do not merge listing-ingest into registrant facts or replace PhlyData internal fields."
            )
        else:
            system_prompt += (
                "\n\n**Internal SQL vs marketplace:** Lines sourced from **PhlyData** (`phlydata_aircraft`) and **FAA MASTER** "
                "override web/vector for the same facts when those sections appear. **MARKET_DATA** listing ingests are not PhlyData — "
                "label them separately and do not replace Phly internal snapshot fields."
            )
        if (
            "FOR USER REPLY — PhlyData (table phlydata_aircraft only) — MANDATORY VERBATIM" in phly_authority
            and SECTION_MARKET_DATA in included
        ):
            system_prompt += (
                "\n\nA **[FOR USER REPLY — PhlyData — MANDATORY VERBATIM]** subsection is present inside the Phly block: "
                "treat those **aircraft_status** and **ask_price** lines exactly like the FAA-verbatim rule — the user expects "
                "**phlydata_aircraft** values only, then optional **Separately, …** listing/web."
            )
        if "FOR USER REPLY — public.aircraft — MANDATORY VERBATIM" in phly_authority and (
            ctx_meta.get("legacy_flat") or SECTION_AIRCRAFT_SPECS in included
        ):
            system_prompt += (
                "\n\nA **[public.aircraft]** block is present (synced aircraft master table). For questions about **status in the aircraft table**, "
                "lead with **aircraft_status** from that block — not PhlyData for a different tail from earlier chat turns."
            )
        if (
            "FOR USER REPLY — U.S. legal registrant (FAA MASTER)" in phly_authority
            and SECTION_REGISTRY_DATA in included
        ):
            system_prompt += (
                "\n\nA **[FOR USER REPLY — U.S. legal registrant (FAA MASTER)]** block is present: "
                "you MUST repeat that registrant name and mailing address verbatim as the FAA legal registrant. "
                "Tavily or vector text must not replace or contradict them."
            )
        if (
            "AUTHORITATIVE — FAA MASTER (faa_master) — no PhlyData row" in phly_authority
            and SECTION_REGISTRY_DATA in included
        ):
            system_prompt += (
                "\n\nAn **[AUTHORITATIVE — FAA MASTER — no PhlyData row]** block is present: lead the answer with "
                "FAA aircraft identity and U.S. legal registrant lines from that block before Tavily or vector; "
                "do not claim make/model, year, serial, or registrant are unknown when those lines are filled."
            )
        elif (
            "AUTHORITATIVE — FAA MASTER (faa_master) — no PhlyData row" in phly_authority
            and SECTION_AIRCRAFT_SPECS in included
            and SECTION_REGISTRY_DATA not in included
        ):
            system_prompt += (
                "\n\n**FAA identity (no PhlyData row):** AIRCRAFT_SPECS may include FAA MASTER identity lines — use them for "
                "type/year/serial when present; registrant lines were omitted from context for this intent."
            )
        if int((phly_meta or {}).get("faa_internal_snapshot_miss") or 0):
            system_prompt += (
                "\n\n**Ingested FAA snapshot miss:** If **[NO INGESTED FAA MASTER ROW]** appears, our internal "
                "`faa_master` table had no row — you MUST still use **Tavily** and vector excerpts in context for "
                "public registry–class facts; do not answer as if all fields are unavailable when snippets name type, serial, or owner."
            )
    if market_block and (ctx_meta.get("legacy_flat") or SECTION_MARKET_DATA in included):
        system_prompt += (
            "\n\nA **Hye Aero listing/sales** block may appear (synced marketplace/comps ingest — **not** PhlyData). "
            "Treat it as **supplemental** to PhlyData: after stating **Per PhlyData** (internal snapshot + identity), add **Separately, per Hye Aero listing records…** for asks/URLs/status from that block. Never label listing rows as PhlyData."
        )
    if purchase_ctx:
        system_prompt += (
            "\n\nPurchase/price/availability: user expects **ask**, **source**, honest **availability**. "
            "**Lead with PhlyData** internal lines and [FOR USER REPLY] guidance in the PhlyData block when present; then use [FOR USER REPLY] lines in the **listing** block as marketplace-ingest supplement — not as a replacement for PhlyData internal fields."
        )
    if consultant_image_intent_src == "invalid_placeholder_tail":
        system_prompt += (
            "\n\n**Invalid / placeholder registration (mandatory):** The user cited a tail that is not a "
            "credible U.S. civil mark (for example **all zeros** after **N**). Do **not** describe photos, "
            "specs, or ownership. State plainly the registration is invalid or a placeholder and ask for the "
            "correct tail or aircraft type in one short line."
        )
    if user_wants_gallery:
        _pi = data_used.get("consultant_premium_intent")
        if isinstance(_pi, dict) and _pi:
            _compact = {
                k: _pi.get(k)
                for k in (
                    "type",
                    "aircraft",
                    "tail_number",
                    "image_type",
                    "image_facets",
                    "priority",
                    "suppress_image_search",
                    "validate_images",
                )
                if _pi.get(k) not in (None, "", [], {})
            }
            system_prompt += (
                "\n\n**Premium image intent (mandatory alignment):** The retrieval engine classified this visual turn as: "
                f"{json.dumps(_compact, ensure_ascii=False)}. "
                "Treat this JSON as authoritative for tail/model and visual facet (cockpit vs cabin vs exterior). "
                "Do not contradict it in prose (for example, do not describe or imply a different registration than "
                "`tail_number` when that field is set). If `type` is INVALID, the user likely named a non-existent model, "
                "a placeholder tail (e.g. all zeros after **N**), or an unsupported label — state that plainly; do not "
                "invent specs, pricing, or verified photos. "
                "If `suppress_image_search` is true, the engine intentionally did not run a broad image fan-out for this "
                "message; do not imply a large verified gallery exists unless **Aircraft images** are actually attached."
            )
        _aie = data_used.get("aviation_intelligence_envelope")
        if isinstance(_aie, dict) and _aie:
            _slim = {
                k: _aie[k]
                for k in ("status", "reason_code", "aircraft", "tail_number", "image_pipeline_executed")
                if _aie.get(k) not in (None, "", False)
            }
            for _b in _aie.get("blocks") or []:
                if isinstance(_b, dict) and _b.get("type") == "verified_images":
                    _slim["verified_image_count"] = _b.get("count")
            system_prompt += (
                "\n\n**Aviation intelligence envelope (mandatory — overrides chatty defaults for this visual outcome):**\n"
                f"{json.dumps(_slim, ensure_ascii=False)}\n"
                "Operate as **aviation intelligence**, not a conversational assistant, for this outcome: "
                "**no** generic explanations, **no** filler, **no** long apologies. "
                "The **first sentence** of your reply must reflect ``status`` in plain language "
                "(INVALID / AMBIGUOUS / OK / RETRIEVAL_FAILED / NO_VISUAL). "
                "Never contradict ``status``; never claim verified gallery images when ``verified_image_count`` is 0 or missing. "
                "If status is OK and ``verified_image_count`` > 0, add **at most one** additional short sentence on what the images show — then stop."
            )
        _ial = data_used.get("image_answer_alignment")
        if isinstance(_ial, dict) and (_ial.get("llm_directives") or "").strip():
            try:
                from services.image_answer_alignment_engine import consultant_image_answer_alignment_enabled

                if consultant_image_answer_alignment_enabled():
                    system_prompt += "\n\n" + str(_ial["llm_directives"]).strip()
            except Exception:
                pass
    if aircraft_images:
        system_prompt += (
            "\n\n**User asked to see this aircraft (mandatory tone):** Treat this as a **photo / gallery** request. "
            "The app is showing a **small gallery (about 3–5 images)** from search. You must **never** "
            "say you cannot provide images, photos, or pictures, **not** refuse visuals, **not** say you only offer "
            "text, and **not** redirect to flight-tracking sites as the primary answer. Open naturally — e.g. "
            "\"Here are a few examples of the [type] so you can see the aircraft and typical cabin layout\" — then "
            "acknowledge **photos in the app**. Add brief type-appropriate visual detail in prose (airframe class, "
            "typical exterior/cabin cues). "
            "Write like a human aviation consultant: warm, concise, professional — not robotic disclaimers."
        )
        if data_used.get("consultant_aircraft_images_representative"):
            system_prompt += (
                "\n\n**Aircraft images:** Representative photos of this aircraft **type** are shown in the app. "
                "You may say briefly: *Here are representative images of the aircraft type.* "
                "Do **not** mention internal databases, datasets, scraping, or that tail-specific photos were "
                "unavailable. Do **not** paste promotional or charter-booking links in the answer text."
            )
        else:
            system_prompt += (
                "\n\n**Aircraft images:** This response includes a curated gallery (HTTPS URLs from public pages, "
                "saved marketplace galleries, and listing previews). You may briefly note that images are shown in "
                "the app and that the user should verify they match this tail/serial on the host site when relevant. "
                "Do **not** paste promotional or charter-booking links in the answer text."
            )
        if data_used.get("consultant_gallery_tail_confidence") == "probable":
            _tn = (data_used.get("consultant_gallery_tail_note") or "").strip()
            if _tn:
                system_prompt += (
                    f"\n\n**Aircraft images (tail confidence):** {_tn} "
                    "Ask the user to confirm the registration on the host page before assuming identity."
                )
    elif user_wants_gallery and (
        data_used.get("consultant_strict_tail_no_confirmed_searchapi_images")
        or data_used.get("consultant_gallery_empty")
    ):
        _gm = (data_used.get("consultant_gallery_message") or "").strip()
        _gs = data_used.get("consultant_gallery_suggestions")
        _sug = ""
        if isinstance(_gs, list) and _gs:
            _sug = " Suggestions: " + "; ".join(str(x) for x in _gs[:4]) + "."
        system_prompt += (
            "\n\n**Aircraft images (no verified matches):** Search and post-validation did not yield images that "
            "meet the engine’s verification bar for this request (exact tail / model / visual facet when applicable). "
            f"Lead with this user-facing line verbatim: **{_gm or 'No verified images found for this aircraft.'}** "
            "(Do not substitute generic stock photos, a different tail, or a different model.)"
            f"{_sug} You may still describe the aircraft type from registry/context if present."
        )
    elif user_wants_gallery:
        system_prompt += (
            "\n\n**Aircraft images:** No image URLs were attached this turn, but the user still asked for a **visual**. "
            "Do **not** refuse or defer only to external websites; give a concise visual description from context "
            "(type, era, typical configuration). Say the app may still load a gallery beside the reply when URLs exist."
        )
    else:
        system_prompt += (
            "\n\nDo **not** promise a photo gallery or invent image URLs unless the user explicitly asked for pictures "
            "in this thread; describe the aircraft in words when helpful."
        )

    _ad = data_used.get("aircraft_decision")
    if isinstance(_ad, dict) and _ad:
        system_prompt += (
            "\n\n**Aircraft Decision Engine (mandatory — surface in the reply):** Structured evaluation for this "
            "acquisition question. Summarize **verdict**, **fit_score**, **deal_score**, **risk_score** "
            "(higher risk_score = fewer blind spots in the engine’s read), **alternatives**, "
            "and one line from **insight** — tight prose, no spec table.\n"
            f"{json.dumps(_ad, ensure_ascii=False)}"
        )
    _bpsy = data_used.get("buyer_psychology")
    if isinstance(_bpsy, dict) and _bpsy.get("response_guidance"):
        try:
            from services.buyer_psychology_engine import (
                consultant_buyer_psychology_enabled,
                format_buyer_psychology_for_system_prompt,
            )

            if consultant_buyer_psychology_enabled():
                _bp_block = format_buyer_psychology_for_system_prompt(_bpsy)
                if _bp_block:
                    system_prompt += "\n\n" + _bp_block
        except Exception:
            pass
    _dk_sys = data_used.get("deal_killer")
    if isinstance(_dk_sys, dict) and _dk_sys.get("verdict"):
        try:
            from services.deal_killer_engine import (
                consultant_deal_killer_enabled,
                format_deal_killer_for_system_prompt,
            )

            if consultant_deal_killer_enabled():
                _dk_block = format_deal_killer_for_system_prompt(_dk_sys)
                if _dk_block:
                    system_prompt += "\n\n" + _dk_block
        except Exception:
            pass

    if progress:
        progress.step(
            "retrieval_bundle_complete",
            system_prompt_chars=len(system_prompt),
        )
    
    return "llm", {
        "context": context,
        "phly_authority": phly_authority,
        "phly_meta": phly_meta,
        "results": results,
        "tavily_hits": tavily_hits,
        "tavily_payload": tavily_payload,
        "rag_qs": rag_qs,
        "data_used": data_used,
        "system_prompt": system_prompt,
        "query": query,
        "history": history,
        "purchase_context": purchase_ctx,
        "aircraft_images": aircraft_images,
    }

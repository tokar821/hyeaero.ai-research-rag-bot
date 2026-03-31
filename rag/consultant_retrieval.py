"""Consultant retrieval orchestration.

Stages (see also :mod:`rag.consultant_pipeline`): SQL authority + market SQL, Pinecone
retrieval (with rerank inside ``svc._retrieve_multi``), optional Tavily, context assembly, LLM payload.

Prompt helpers are imported lazily from :mod:`rag.query_service` to avoid circular imports.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def run_consultant_retrieval_bundle(
    svc: Any,
    query: str,
    top_k: int,
    max_context_chars: int,
    score_threshold: Optional[float],
    history: Optional[List[Dict[str, str]]],
) -> Tuple[str, Any]:
    prof = svc._professional_search_answer(query)
    if prof:
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
    from rag.consultant_intent import resolve_aircraft_image_gallery_intent
    from rag.consultant_market_lookup import (
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
        wants_consultant_explicit_photo_web,
    )
    from rag.consultant_tavily_gate import (
        empty_consultant_tavily_payload,
        should_run_consultant_tavily_after_internal,
    )
    from services.tavily_owner_hint import fetch_tavily_hints_for_query
    from rag.consultant_pipeline import (
        build_consultant_llm_context,
        load_consultant_pipeline_config,
        summarize_consultant_entities,
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
    
    def _run_image_gallery_intent() -> Tuple[bool, str]:
        kwords = _env_truthy("CONSULTANT_IMAGE_INTENT_KEYWORDS_ONLY") or (
            low_latency and not _env_truthy("CONSULTANT_IMAGE_INTENT_LLM_WHEN_FAST")
        )
        return resolve_aircraft_image_gallery_intent(
            query,
            history,
            api_key=svc.openai_api_key or "",
            model=intent_model or svc.chat_model,
            keyword_fallback=lambda: wants_consultant_aircraft_images_in_answer(query, history),
            keywords_only=kwords,
        )
    
    if skip_expand:
        def _run_phly() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
            return svc._phlydata_authority_block(query, history)
    
        with ThreadPoolExecutor(max_workers=2) as pre_pool:
            f_phly = pre_pool.submit(_run_phly)
            f_int = pre_pool.submit(_run_image_gallery_intent)
            phly_authority, phly_meta, phly_rows = f_phly.result()
            phly_authority = (
                _consultant_tavily_first_when_faa_ingest_miss_prefix(phly_meta)
                + _consultant_faa_no_phly_priority_prefix(phly_meta)
                + (phly_authority or "")
            )
            user_wants_gallery, consultant_image_intent_src = f_int.result()
        qstrip = (query or "").strip()
        expanded = {
            "tavily_query": qstrip[:400] if qstrip else "",
            "rag_queries": [qstrip] if qstrip else [""],
        }
    else:
        def _run_phly() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
            return svc._phlydata_authority_block(query, history)
    
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
                _consultant_tavily_first_when_faa_ingest_miss_prefix(phly_meta)
                + _consultant_faa_no_phly_priority_prefix(phly_meta)
                + (phly_authority or "")
            )
            expanded = f_exp.result()
            user_wants_gallery, consultant_image_intent_src = f_int.result()
    
    market_block, market_meta = build_consultant_market_authority_block(
        svc.db,
        query,
        history,
        phly_rows,
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
    entity_detection = summarize_consultant_entities(query, history, phly_meta, phly_rows)
    tq = enrich_tavily_query_for_consultant(
        query,
        expanded.get("tavily_query") or query,
        phly_rows,
        history_snippet=hs_opt,
        lookup_tokens=consultant_lookup_tokens,
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
    
    results = svc._retrieve_multi(
        rag_qs,
        top_k=top_k,
        score_threshold=score_threshold,
        max_results_total=rag_max_chunks,
        max_query_variants=max_rag_variants,
        skip_rerank=fast_retrieval or low_latency,
        rerank_anchor_query=query,
    )
    results = svc._filter_rag_results_for_phly_aircraft(results, phly_rows)
    
    sql_nonempty = bool((phly_authority or "").strip()) or bool((market_block or "").strip())
    force_tavily_always = _env_truthy("CONSULTANT_TAVILY_ALWAYS")
    run_tavily, tavily_gate_reason = should_run_consultant_tavily_after_internal(
        vector_result_count=len(results),
        sql_context_nonempty=sql_nonempty,
        force_always=force_tavily_always,
    )
    run_image_pass = (
        bool(run_tavily and img_q and not skip_img_pass and user_wants_gallery)
        and (
            not single_tavily_pass
            or wants_consultant_explicit_photo_web(query, history)
            or wants_consultant_aircraft_detail_context(query, history)
        )
    )
    if not run_tavily:
        tavily_passes = 0
        run_secondary = False
        merge_purchase = False
        run_image_pass = False
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
        logger.debug("Consultant: Tavily run (fallback, reason=%s)", tavily_gate_reason)
    purchase_ctx = consultant_wants_internal_market_sql(
        query, history, strict=strict_market_sql
    )
    tavily_max_items = 20 if (purchase_ctx or run_image_pass) else 14
    tavily_body_chars = 2200 if purchase_ctx else 1600
    
    want_img_primary = user_wants_gallery
    
    def _fetch_pri() -> Dict[str, Any]:
        return fetch_tavily_hints_for_query(
            tq,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=want_img_primary,
        )
    
    def _fetch_sec() -> Optional[Dict[str, Any]]:
        if not run_secondary or not sq:
            return None
        return fetch_tavily_hints_for_query(
            sq,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=want_img_primary,
        )
    
    def _fetch_pur() -> Optional[Dict[str, Any]]:
        if not merge_purchase or not pq:
            return None
        return fetch_tavily_hints_for_query(
            pq,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=want_img_primary,
        )
    
    def _fetch_img() -> Optional[Dict[str, Any]]:
        if not run_image_pass or not img_q:
            return None
        return fetch_tavily_hints_for_query(
            img_q,
            result_limit=tavily_per_pass,
            search_depth=tdepth,
            request_timeout=tavily_timeout,
            include_images=True,
        )
    
    primary: Dict[str, Any] = {}
    secondary: Optional[Dict[str, Any]] = None
    tertiary: Optional[Dict[str, Any]] = None
    quaternary: Optional[Dict[str, Any]] = None
    if run_tavily:
        max_workers = (
            1
            + (1 if run_secondary else 0)
            + (1 if merge_purchase else 0)
            + (1 if run_image_pass else 0)
        )
        max_workers = max(1, min(4, max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
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
    # relax URL-level identity filter when we ran that pass or the user explicitly asked for photos.
    trust_tail_tavily_imgs = user_wants_gallery and (
        bool(run_image_pass) or wants_consultant_explicit_photo_web(query, history)
    )
    aircraft_images: List[Dict[str, Any]] = []
    image_boost_used = 0
    if user_wants_gallery:
        aircraft_images = build_consultant_aircraft_images(
            tavily_payload,
            phly_rows,
            listing_urls=listing_urls_for_img or None,
            listing_rows=lr_img or None,
            trust_tail_biased_tavily_images=trust_tail_tavily_imgs,
        )
    # One extra Tavily call when the merged payload still yielded no images but we have a photo-biased query.
    if (
        user_wants_gallery
        and len(aircraft_images) == 0
        and run_tavily
        and img_q
        and not skip_img_pass
        and not run_image_pass
        and want_img_primary
    ):
        try:
            img_boost = fetch_tavily_hints_for_query(
                img_q,
                result_limit=tavily_per_pass,
                search_depth=tdepth,
                request_timeout=tavily_timeout,
                include_images=True,
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
            )
            if aircraft_images:
                image_boost_used = 1
        except Exception as img_boost_e:
            logger.debug("Consultant image-boost Tavily pass skipped: %s", img_boost_e)
    
    has_phly = bool((phly_authority or "").strip())
    has_market = bool((market_block or "").strip())
    has_rag = bool(results)
    has_tavily = tavily_hits > 0
    if not (has_phly or has_market or has_rag or has_tavily):
        logger.info(
            "Consultant: no PhlyData, no listing/sales block, no vector hits, no Tavily → general knowledge (len=%d)",
            len(query),
        )
        return "gk", None
    
    context, _context_parts = build_consultant_llm_context(
        phly_authority=phly_authority,
        market_block=market_block,
        tavily_block=tavily_block,
        rag_results=results,
        max_context_chars=max_context_chars,
    )
    if not context.strip():
        logger.info(
            "RAG answer: no PhlyData, Tavily text, or Pinecone context; general knowledge (len=%d)",
            len(query),
        )
        return "gk", None
    
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
    data_used["consultant_pipeline"] = "entity_router_sql_vector_rerank_context_llm_v1"
    data_used["consultant_query_router"] = pipe_cfg.query_router_snapshot()
    data_used["consultant_entity_detection"] = entity_detection.asdict()
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
    data_used["tavily_gate_reason"] = tavily_gate_reason
    if force_tavily_always:
        data_used["consultant_tavily_always"] = 1
    if not run_tavily:
        data_used["tavily_skipped"] = 1
    if purchase_tavily_merged:
        data_used["tavily_purchase_focus"] = 1
    data_used["tavily_error"] = tavily_payload.get("error")
    data_used["rag_query_variants"] = len(rag_qs)
    if svc._rerank_enabled_globally() and not (fast_retrieval or low_latency):
        data_used["rag_semantic_rerank"] = 1
        data_used["rag_rerank_model"] = (
            (os.getenv("RAG_RERANKER_MODEL") or "BAAI/bge-reranker-large").strip()
        )
        if results and any(r.get("rerank_score") is not None for r in results):
            data_used["rag_rerank_applied"] = 1
    else:
        data_used["rag_semantic_rerank"] = 0
    for r in results:
        et = (r.get("entity_type") or "other").replace("_", " ")
        data_used[et] = data_used.get(et, 0) + 1
    
    data_used["aircraft_images"] = aircraft_images
    data_used["consultant_aircraft_image_count"] = len(aircraft_images)
    # Internal join helper for image lookup — not needed by clients; keeps responses smaller.
    data_used.pop("consultant_listing_rows_for_images", None)
    if wants_consultant_explicit_photo_web(query, history):
        data_used["consultant_user_asked_photos"] = 1
    if user_wants_gallery:
        data_used["consultant_show_image_ui_context"] = 1
    data_used["consultant_image_intent_source"] = consultant_image_intent_src
    
    system_prompt = CONSULTANT_SYSTEM_PROMPT
    if phly_authority:
        system_prompt += (
            "\n\nThe context may begin with an AUTHORITATIVE **PhlyData (Hye Aero aircraft source) + FAA MASTER** block. "
            "That block is Hye Aero's **canonical internal record**: identity, internal snapshot fields (status, ask-as-exported, programs, etc.), and legal U.S. registrant when present — **all override** web or vector for those fields. "
            "Listing/market rows elsewhere are **not** PhlyData — use them as **supplemental** context after PhlyData; do not merge listing-ingest into registrant facts or replace PhlyData internal fields."
        )
        if "FOR USER REPLY — PhlyData (table phlydata_aircraft only) — MANDATORY VERBATIM" in phly_authority:
            system_prompt += (
                "\n\nA **[FOR USER REPLY — PhlyData — MANDATORY VERBATIM]** subsection is present inside the Phly block: "
                "treat those **aircraft_status** and **ask_price** lines exactly like the FAA-verbatim rule — the user expects "
                "**phlydata_aircraft** values only, then optional **Separately, …** listing/web."
            )
        if "FOR USER REPLY — public.aircraft — MANDATORY VERBATIM" in phly_authority:
            system_prompt += (
                "\n\nA **[public.aircraft]** block is present (synced aircraft master table). For questions about **status in the aircraft table**, "
                "lead with **aircraft_status** from that block — not PhlyData for a different tail from earlier chat turns."
            )
        if "FOR USER REPLY — U.S. legal registrant (FAA MASTER)" in phly_authority:
            system_prompt += (
                "\n\nA **[FOR USER REPLY — U.S. legal registrant (FAA MASTER)]** block is present: "
                "you MUST repeat that registrant name and mailing address verbatim as the FAA legal registrant. "
                "Tavily or vector text must not replace or contradict them."
            )
        if "AUTHORITATIVE — FAA MASTER (faa_master) — no PhlyData row" in phly_authority:
            system_prompt += (
                "\n\nAn **[AUTHORITATIVE — FAA MASTER — no PhlyData row]** block is present: lead the answer with "
                "FAA aircraft identity and U.S. legal registrant lines from that block before Tavily or vector; "
                "do not claim make/model, year, serial, or registrant are unknown when those lines are filled."
            )
        if int((phly_meta or {}).get("faa_internal_snapshot_miss") or 0):
            system_prompt += (
                "\n\n**Ingested FAA snapshot miss:** If **[NO INGESTED FAA MASTER ROW]** appears, our internal "
                "`faa_master` table had no row — you MUST still use **Tavily** and vector excerpts in context for "
                "public registry–class facts; do not answer as if all fields are unavailable when snippets name type, serial, or owner."
            )
    if market_block:
        system_prompt += (
            "\n\nA **Hye Aero listing/sales** block may appear (synced marketplace/comps ingest — **not** PhlyData). "
            "Treat it as **supplemental** to PhlyData: after stating **Per PhlyData** (internal snapshot + identity), add **Separately, per Hye Aero listing records…** for asks/URLs/status from that block. Never label listing rows as PhlyData."
        )
    if purchase_ctx:
        system_prompt += (
            "\n\nPurchase/price/availability: user expects **ask**, **source**, honest **availability**. "
            "**Lead with PhlyData** internal lines and [FOR USER REPLY] guidance in the PhlyData block when present; then use [FOR USER REPLY] lines in the **listing** block as marketplace-ingest supplement — not as a replacement for PhlyData internal fields."
        )
    if aircraft_images:
        system_prompt += (
            "\n\n**Aircraft images:** This response includes a curated gallery (real HTTPS URLs from web search, "
            "saved marketplace galleries, and listing previews). You may briefly note that images are shown in "
            "the app and that the user should verify they match this tail/serial on the host site."
        )
    else:
        system_prompt += (
            "\n\nThe product may show a **separate image gallery** when URLs are available (web search + listing "
            "sources only). Do **not** invent image URLs; describe the aircraft in words when helpful."
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

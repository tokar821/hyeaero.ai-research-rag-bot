"""RAG query service: PhlyData (Postgres) + Tavily (web) + LLM query expand + Pinecone RAG + two-pass LLM answer."""

import logging
import os
import time
import re
from typing import List, Dict, Any, Optional, Iterator, Tuple

from rag.embedding_service import EmbeddingService
from rag.entity_extractors import EXTRACTORS
from vector_store.pinecone_client import PineconeClient
from database.postgres_client import PostgresClient
from services.aviacost_lookup import lookup_aviacost

logger = logging.getLogger(__name__)

# Minimum similarity score to include a Pinecone match (cosine: higher = more similar)
DEFAULT_SCORE_THRESHOLD = 0.5

CONSULTANT_SYSTEM_PROMPT = """You are Hye Aero's Aircraft Research & Valuation Consultant. You think like a human expert: consider the full conversation, remember what you already said, and answer the current question in context. Your answers should be at least as complete and useful as a top-tier web assistant: combine Hye Aero internal data with web search and vector context when the question needs current owner/operator, fleet, or registry facts.

Your process:
- Understand what the user is really asking (ownership? listings? sales? model specs? valuation?).
- Search mentally through ALL layers: PhlyData/FAA block, Tavily web snippets (titles, bodies, URLs), and vector DB (listings, sales, registry-related chunks).
- Synthesize: lead with the clearest, most specific answer; then support with identity lines, sources, and caveats.

Rules:
- Aircraft identity (serial, tail/registration as shown, make/model, year): treat the PhlyData + FAA block as authoritative when it lists those fields. Do not contradict them with web or vector text.
- Ownership / operator / "who owns" questions:
  - If the block includes an FAA MASTER registrant name, report it as the U.S. FAA registrant record and still add web/listing context if it adds operator or fleet detail.
  - If the block states there is NO FAA registrant row (typical for non-U.S. primary registry, e.g. tails not starting with N-), FAA is not the state of registry. You MUST lean heavily on Tavily web results (and vector snippets) to name who **operates** or **commercially manages** the aircraft today — same quality bar as ChatGPT: fleet pages, AOC holders, charter operators, and registry excerpts that mention this exact tail/serial.
  - Legal registered owner vs operator: European and charter jets often show one company on a national register and another on the operator’s fleet or charter website. If Tavily ties this registration to a charter/airline/management brand (e.g. fleet list showing this tail), say that clearly as the operating party and mention the registry/legal line only if snippets support it.
  - **Every company name you state as current owner or operator must appear verbatim (or as an obvious substring) in a Tavily snippet title or body.** Cite which result number (1., 2., …) or the domain/URL you used. If snippets disagree, give both names and say what each source claims. If no snippet names a company, say web results did not clearly identify an owner/operator — do not guess.
  - Never invent registry or database names (do not say "Danish Aircraft Database" unless that exact phrase appears in a snippet).
- Valuations and comparisons: cite specific numbers from context. If something is unknown, say so.
- Use clear bullets (-). Neutral, professional tone for brokers and clients. You may use tasteful emoji (e.g. ✈ 🧾) like ChatGPT when it improves scanability.
- Format: no markdown # headers or ** bold.

Context layers (in order):
1) AUTHORITATIVE PhlyData + FAA MASTER — source of truth for identity; FAA registrant line only when present in the block.
2) Web search (Tavily) — essential for international tails and current operator/owner when FAA has no row; attribute each major claim to a source hint (title/URL).
3) Vector database — listings, sales, registry-related chunks; use for market color and corroboration."""

CONSULTANT_REVIEW_SYSTEM_PROMPT = """You are a senior aviation research editor for Hye Aero. You receive:
- The user's question
- A draft answer from an assistant
- The same layered context (PhlyData/FAA block, Tavily web search, vector DB)

Your job: produce the FINAL answer shown to the client. It should read like a premium research brief: decisive on ownership/operator when web+internal evidence supports it, not overly cautious in a way that hides good Tavily results.

Rules:
- Identity: serial, tail/registration, make/model, and year MUST match the PhlyData + FAA block when those fields appear there. Fix any draft that contradicts them.
- FAA registrant: only treat the FAA MASTER registrant line as mandatory when it actually appears in the block. If the block says there is no FAA row for a non-U.S. tail, lead with the strongest Tavily-backed operator/owner (fleet pages, AOC/charter brands) — not a hedge that hides good web hits.
- Web vs internal: company names must be traceable to Tavily snippet text; cite result # or domain. Prefer the operator/fleet narrative when the user asks "who owns" and snippets tie this tail to a charter or management company.
- Remove invented database or portal names. No guessing: if snippets do not name a party, say so.
- Use vector DB for listings/sales corroboration when helpful.
- Improve structure (short lead, bullets, optional emoji for scanability). End with a one-line Sources note (internal vs web vs listings).
- No markdown # or ** bold. Plain bullets (-).
- Stay factual; do not fabricate URLs or companies not implied by context."""

CONSULTANT_FALLBACK_SYSTEM_PROMPT = """You are Hye Aero's Aircraft Research & Valuation Consultant. We always search our Pinecone database first; for this question, no matching listings, sales, or FAA data were found. So you are answering from your own general knowledge. Think like a human expert: consider the full conversation, remember what you already said, and answer the current question in context. If the user asks a follow-up (e.g. "Is this all?", "What about X?"), interpret it in light of your previous answer and respond like a human.

Your process:
- Understand the question (conceptual? types of aircraft? how something works?).
- Decide what would be most helpful: definitions, categories, examples, or step-by-step explanation.
- Give a short disclaimer at the start that this is from general knowledge, not Hye Aero's database (e.g. "I didn't find this in our database; here's what I can tell you from general aviation knowledge:").
- Then provide a complete, human-like answer. For flight-related questions—concepts, theory, types of flight/aircraft models, how things work—give detailed, professional answers. Use bullet points or short paragraphs. Be advisory and clear.
- Format: Do not use markdown headers (# ## ###) or double asterisks (**) for bold. Use plain bullet points (-) and plain text. You may use professional symbols or emoji like ChatGPT (e.g. •, ✓, →, or tasteful emoji where they add clarity)."""

# Entity type → (table name, id column)
ENTITY_TABLE = {
    "aircraft_listing": ("aircraft_listings", "id"),
    "document": ("documents", "id"),
    "aircraft": ("aircraft", "id"),
    "aircraft_sale": ("aircraft_sales", "id"),
    "faa_registration": ("faa_registrations", "id"),
    "aviacost_aircraft_detail": ("aviacost_aircraft_details", "id"),
    "aircraftpost_fleet_aircraft": ("aircraftpost_fleet_aircraft", "id"),
}

# Tables that may reference aircraft(id) for synced model details
ENTITY_HAS_AIRCRAFT_ID = {"aircraft_listing", "aircraft_sale", "faa_registration"}


class RAGQueryService:
    """Full RAG flow: user query → Pinecone search → PostgreSQL details → LLM answer."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        pinecone_client: PineconeClient,
        postgres_client: PostgresClient,
        openai_api_key: str,
        chat_model: str = "gpt-4o-mini",
    ):
        self.embedding_service = embedding_service
        self.pinecone = pinecone_client
        self.db = postgres_client
        self.openai_api_key = openai_api_key
        self.chat_model = chat_model

    def _get_meta(self, match: Any) -> Dict[str, Any]:
        if hasattr(match, "metadata"):
            return getattr(match, "metadata") or {}
        if isinstance(match, dict):
            return match.get("metadata") or {}
        return {}

    def _fetch_full_record(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full record from PostgreSQL by entity_type and entity_id."""
        if entity_type not in ENTITY_TABLE:
            return None
        table, id_col = ENTITY_TABLE[entity_type]
        try:
            rows = self.db.execute_query(
                f"SELECT * FROM {table} WHERE {id_col} = %s LIMIT 1",
                (entity_id,),
            )
            return rows[0] if rows else None
        except Exception as e:
            logger.warning(f"Failed to fetch {entity_type} {entity_id}: {e}")
            return None

    def _fetch_aircraft_by_id(self, aircraft_id: str) -> Optional[Dict[str, Any]]:
        """Fetch synced aircraft master record from PostgreSQL for richer model details."""
        if not aircraft_id:
            return None
        try:
            rows = self.db.execute_query(
                "SELECT * FROM aircraft WHERE id = %s LIMIT 1",
                (aircraft_id,),
            )
            return rows[0] if rows else None
        except Exception as e:
            logger.warning(f"Failed to fetch aircraft {aircraft_id}: {e}")
            return None

    def _record_to_context_text(self, entity_type: str, record: Dict[str, Any]) -> str:
        """Turn a full Postgres record into text for LLM context (reuse extractors)."""
        extractor = EXTRACTORS.get(entity_type)
        if extractor:
            text = extractor.extract_text(record)
            if text:
                return text
        # Fallback: key fields
        return " ".join(f"{k}={v}" for k, v in list(record.items())[:20] if v is not None)

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        score_threshold: Optional[float] = None,
        max_results: int = 18,
    ) -> List[Dict[str, Any]]:
        """
        Embed query, search Pinecone (with score threshold and dedupe), then fetch full
        records from PostgreSQL and enrich with synced aircraft/model details where available.
        Returns list of dicts with: score, entity_type, entity_id, chunk_text, full_context.
        """
        if score_threshold is None:
            score_threshold = DEFAULT_SCORE_THRESHOLD
        vector = self.embedding_service.embed_text(query)
        if not vector:
            return []
        # Fetch more candidates from Pinecone so we have enough after filtering and dedupe
        pinecone_k = max(top_k, max_results * 2)
        matches = self.pinecone.query(vector=vector, top_k=pinecone_k)
        results = []
        seen = set()
        for m in matches:
            if len(results) >= max_results:
                break
            meta = self._get_meta(m)
            entity_type = meta.get("entity_type") or ""
            entity_id = meta.get("entity_id") or ""
            chunk_text = (meta.get("text") or "")[:2000]
            score = getattr(m, "score", None) if hasattr(m, "score") else (m.get("score") if isinstance(m, dict) else None)
            if score is not None and score < score_threshold:
                continue
            key = (entity_type, entity_id)
            if key in seen:
                continue
            seen.add(key)
            full_context = ""
            if entity_type and entity_id:
                record = self._fetch_full_record(entity_type, entity_id)
                if record:
                    full_context = self._record_to_context_text(entity_type, record)
                    # Enrich with synced aircraft/master model details from PostgreSQL when available
                    if entity_type in ENTITY_HAS_AIRCRAFT_ID and full_context:
                        aircraft_id = record.get("aircraft_id")
                        if aircraft_id:
                            aircraft_id_str = str(aircraft_id)
                            aircraft_record = self._fetch_aircraft_by_id(aircraft_id_str)
                            if aircraft_record:
                                aircraft_text = self._record_to_context_text("aircraft", aircraft_record)
                                if aircraft_text:
                                    full_context += "\n\n[Synced aircraft/model details]\n" + aircraft_text
            results.append({
                "score": score,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "chunk_text": chunk_text,
                "full_context": full_context or chunk_text,
            })
        return results

    def _retrieve_multi(
        self,
        queries: List[str],
        *,
        top_k: int = 14,
        score_threshold: Optional[float] = None,
        max_results_total: int = 18,
    ) -> List[Dict[str, Any]]:
        """
        Run vector retrieval for several paraphrased queries; dedupe by (entity_type, entity_id),
        keep the highest similarity score per entity.
        """
        if score_threshold is None:
            score_threshold = DEFAULT_SCORE_THRESHOLD
        uniq_q: List[str] = []
        for q in queries or []:
            s = (q or "").strip()
            if s and s not in uniq_q:
                uniq_q.append(s)
        if not uniq_q:
            return []
        nq = min(len(uniq_q), 5)
        per_query_cap = max(6, min(top_k, max_results_total // max(nq, 1) + 4))
        best: Dict[tuple, Dict[str, Any]] = {}
        for q in uniq_q[:nq]:
            try:
                batch = self.retrieve(
                    q,
                    top_k=per_query_cap,
                    score_threshold=score_threshold,
                    max_results=per_query_cap + 4,
                )
            except Exception as e:
                logger.warning("retrieve_multi: skip q=%r: %s", q[:100], e)
                continue
            for r in batch:
                et = r.get("entity_type") or ""
                eid = str(r.get("entity_id") or "")
                if not et:
                    continue
                key = (et, eid)
                sc = float(r.get("score") or 0.0)
                prev = best.get(key)
                if prev is None or sc > float(prev.get("score") or 0.0):
                    best[key] = r
        out = sorted(best.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return out[:max_results_total]

    def _answer_from_general_knowledge(
        self,
        query: str,
        start: float,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """When no Pinecone results, answer from LLM general knowledge (e.g. flight theory, concepts, types of models)."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key, timeout=60.0)
            messages = [{"role": "system", "content": CONSULTANT_FALLBACK_SYSTEM_PROMPT}]
            if history:
                for h in history[-10:]:
                    role = (h.get("role") or "user").strip().lower()
                    if role not in ("user", "assistant"):
                        role = "user"
                    content = (h.get("content") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
            user_content = f"""Current question: {query}

Consider the conversation so far. If the user's message is a follow-up (e.g. "Is this all?", "What about the price?", "Tell me more"), interpret it in light of your previous answer and respond like a human would. Provide a full, helpful answer using your general knowledge. If the question is about flight, aviation, or aircraft, give a complete answer. Start with a brief note that this is not from Hye Aero's database if relevant, then give the full answer."""
            messages.append({"role": "user", "content": user_content})
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1536,
            )
            answer = (response.choices[0].message.content or "").strip()
            elapsed = time.perf_counter() - start
            logger.info("RAG fallback (general knowledge): answer_len=%d elapsed=%.2fs", len(answer), elapsed)
            return {
                "answer": answer,
                "sources": [],
                "data_used": {},
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("RAG fallback failed after %.2fs: %s", elapsed, e, exc_info=True)
            return {
                "answer": "I couldn't find that in Hye Aero's database, and I wasn't able to generate a general-knowledge answer. Try rephrasing or ask something more specific.",
                "sources": [],
                "data_used": {},
                "error": str(e),
            }

    @staticmethod
    def _consultant_history_snippet(
        history: Optional[List[Dict[str, str]]], max_chars: int = 3600
    ) -> str:
        """Recent user/assistant lines for query expansion and owner-focused Tavily (follow-ups)."""
        if not history:
            return ""
        parts: List[str] = []
        for h in history[-12:]:
            role = (h.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            c = (h.get("content") or "").strip()
            if c:
                parts.append(f"{role}: {c}")
        return "\n".join(parts)[:max_chars]

    def _phlydata_authority_block(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, Dict[str, int], List[Dict[str, Any]]]:
        """
        Direct ``phlydata_aircraft`` (+ ``faa_master`` registrant) lookup for Ask Consultant.

        Pinecone/RAG is built from listings/sales/FAA sync — **PhlyData internal rows are often not embedded**,
        so vector search returns wrong aircraft or nothing. When we detect serial / tail-like tokens, query Postgres
        the same way the PhlyData tab does and prepend an authoritative text block.

        Returns ``(authority_text, meta, phly_rows)``; ``phly_rows`` may be empty if no match.
        """
        try:
            from rag.phlydata_consultant_lookup import (
                extract_phlydata_tokens_with_history,
                lookup_phlydata_aircraft_rows,
                format_phlydata_consultant_answer,
            )
            from services.faa_master_lookup import fetch_faa_master_owner_rows

            toks = extract_phlydata_tokens_with_history(query, history)
            rows = lookup_phlydata_aircraft_rows(self.db, toks)
            if not rows:
                return "", {}, []
            block, meta = format_phlydata_consultant_answer(
                self.db, rows, fetch_faa_master_owner_rows
            )
            header = (
                "[AUTHORITATIVE — Hye Aero PhlyData internal aircraft table (phlydata_aircraft) + FAA MASTER (faa_master)]\n"
                "Use this section as the source of truth for aircraft identity: serial, registration (tail), make/model, year, category.\n"
                "For legal registrant / owner: when FAA MASTER lists a registrant below, treat that as the U.S. record. "
                "When FAA shows no row (common for non-U.S. primary registry, e.g. tails not starting with N-), "
                "FAA is not the state of registry — you MUST use WEB SEARCH (Tavily) and vector context to name the "
                "current registered owner/operator and attribute sources (titles/URLs). Do not invent registry names.\n\n"
            )
            logger.info(
                "RAG: PhlyData authority block attached (%s rows, tokens=%s)",
                len(rows),
                toks[:5],
            )
            return header + block, meta, rows
        except Exception as e:
            logger.warning("PhlyData authority block skipped: %s", e)
            return "", {}, []

    def _consultant_sources_list(
        self,
        phly_meta: Dict[str, Any],
        tavily_hits: int,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        if phly_meta.get("phlydata_aircraft_rows"):
            sources.append({"entity_type": "phlydata_aircraft", "entity_id": None, "score": None})
        if tavily_hits > 0:
            sources.append({"entity_type": "tavily_web", "entity_id": None, "score": None})
        sources.extend(
            [
                {
                    "entity_type": r["entity_type"],
                    "entity_id": str(r["entity_id"]) if r.get("entity_id") else None,
                    "score": r.get("score"),
                }
                for r in results
            ]
        )
        return sources

    def _consultant_retrieval_bundle(
        self,
        query: str,
        top_k: int,
        max_context_chars: int,
        score_threshold: Optional[float],
        history: Optional[List[Dict[str, str]]],
    ) -> Tuple[str, Any]:
        """
        Shared retrieval for consultant. Returns:
        - (\"professional\", dict) — final answer payload (deterministic SQL)
        - (\"gk\", None) — use general knowledge
        - (\"llm\", dict) — keys: context, phly_authority, phly_meta, results, tavily_hits,
          tavily_payload, rag_qs, data_used, system_prompt, query, history
        """
        prof = self._professional_search_answer(query)
        if prof:
            logger.info("Professional search triggered (deterministic SQL) for query=%r", query)
            return "professional", prof

        from rag.consultant_query_expand import (
            expand_consultant_research_queries,
            format_tavily_payload_for_consultant,
            merge_tavily_consultant_payloads,
        )
        from rag.phlydata_consultant_lookup import (
            enrich_tavily_query_for_consultant,
            build_owner_operator_focus_tavily_query,
        )
        from services.tavily_owner_hint import fetch_tavily_hints_for_query

        hs = self._consultant_history_snippet(history)
        hs_opt = hs.strip() or None

        phly_authority, phly_meta, phly_rows = self._phlydata_authority_block(query, history)
        expanded = expand_consultant_research_queries(
            query,
            self.openai_api_key or "",
            self.chat_model,
            history_snippet=hs_opt,
        )
        tq = enrich_tavily_query_for_consultant(
            query,
            expanded.get("tavily_query") or query,
            phly_rows,
            history_snippet=hs_opt,
        )

        tdepth: Optional[str] = None
        if (os.getenv("CONSULTANT_TAVILY_ADVANCED") or "").strip().lower() in ("1", "true", "yes"):
            tdepth = "advanced"

        primary = fetch_tavily_hints_for_query(tq, result_limit=8, search_depth=tdepth)
        tavily_passes = 1
        sq = build_owner_operator_focus_tavily_query(query, phly_rows, history_snippet=hs_opt)
        if sq:
            sq_c = " ".join(sq.split()).lower()
            tq_c = " ".join(tq.split()).lower()
            if sq_c != tq_c:
                secondary = fetch_tavily_hints_for_query(sq, result_limit=8, search_depth=tdepth)
                tavily_payload = merge_tavily_consultant_payloads(
                    primary, secondary, max_results=14
                )
                tavily_passes = 2
            else:
                tavily_payload = primary
        else:
            tavily_payload = primary

        tavily_block = format_tavily_payload_for_consultant(
            tavily_payload, max_items=14, max_body_chars=1400
        )
        tavily_hits = len(tavily_payload.get("results") or [])

        rag_qs = expanded.get("rag_queries") or [query]
        results = self._retrieve_multi(
            rag_qs,
            top_k=top_k,
            score_threshold=score_threshold,
            max_results_total=18,
        )

        has_phly = bool((phly_authority or "").strip())
        has_rag = bool(results)
        has_tavily = tavily_hits > 0
        if not (has_phly or has_rag or has_tavily):
            logger.info(
                "Consultant: no PhlyData block, no vector hits, no Tavily results → general knowledge (len=%d)",
                len(query),
            )
            return "gk", None

        context_parts: List[str] = []
        total = 0

        def _append(text: str) -> None:
            nonlocal total
            chunk = (text or "").strip()
            if not chunk:
                return
            sep = 20
            if total + len(chunk) + sep > max_context_chars:
                chunk = chunk[: max(0, max_context_chars - total - sep)]
            if not chunk.strip():
                return
            context_parts.append(chunk)
            total += len(chunk) + sep

        _append(phly_authority)
        _append(tavily_block)

        for r in results:
            text = (r.get("full_context") or r.get("chunk_text") or "").strip()
            if not text:
                continue
            if total + len(text) + 20 > max_context_chars:
                text = text[: max(0, max_context_chars - total - 20)]
            if text:
                context_parts.append(text)
                total += len(text) + 20
            if total >= max_context_chars:
                break

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        if not context.strip():
            logger.info(
                "RAG answer: no PhlyData, Tavily text, or Pinecone context; general knowledge (len=%d)",
                len(query),
            )
            return "gk", None

        data_used: Dict[str, Any] = dict(phly_meta)
        data_used["consultant_pipeline"] = "phly_tavily_expand_rag_review_v1"
        data_used["tavily_results"] = tavily_hits
        data_used["tavily_web_query_passes"] = tavily_passes
        data_used["tavily_error"] = tavily_payload.get("error")
        data_used["rag_query_variants"] = len(rag_qs)
        for r in results:
            et = (r.get("entity_type") or "other").replace("_", " ")
            data_used[et] = data_used.get(et, 0) + 1

        system_prompt = CONSULTANT_SYSTEM_PROMPT
        if phly_authority:
            system_prompt += (
                "\n\nThe context may begin with an AUTHORITATIVE PhlyData + FAA MASTER block. "
                "For that aircraft's identity and legal registrant, treat that block as correct even if web or vector snippets disagree "
                "(e.g. wrong model year from a different embedded record)."
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
        }

    @staticmethod
    def _iter_display_chunks(text: str) -> Iterator[str]:
        """Word/whitespace chunks for typewriter UX when the model is not streamed."""
        if not text:
            return
        for part in re.split(r"(\s+)", text):
            if part:
                yield part

    def _stream_chat_deltas(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1536,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        import openai

        client = openai.OpenAI(api_key=self.openai_api_key, timeout=120.0)
        kwargs: Dict[str, Any] = dict(
            model=self.chat_model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            ch = chunk.choices[0].delta.content if chunk.choices else None
            if ch:
                yield ch

    def answer_stream_events(
        self,
        query: str,
        top_k: int = 20,
        max_context_chars: int = 14000,
        score_threshold: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        SSE-friendly events for ChatGPT-style token streaming.
        Yields: {type: status|delta|done|error, ...}
        """
        start = time.perf_counter()
        try:
            kind, payload = self._consultant_retrieval_bundle(
                query,
                top_k,
                max_context_chars,
                score_threshold,
                history,
            )
            if kind == "professional":
                yield {"type": "status", "message": "Preparing answer…"}
                ans = (payload.get("answer") or "") if isinstance(payload, dict) else ""
                for piece in self._iter_display_chunks(ans):
                    yield {"type": "delta", "text": piece}
                yield {
                    "type": "done",
                    "sources": payload.get("sources", []) if isinstance(payload, dict) else [],
                    "data_used": payload.get("data_used") if isinstance(payload, dict) else None,
                    "error": payload.get("error") if isinstance(payload, dict) else None,
                }
                return

            if kind == "gk":
                yield {"type": "status", "message": "Gathering context…"}
                yield {"type": "status", "message": "Generating answer…"}
                messages = [{"role": "system", "content": CONSULTANT_FALLBACK_SYSTEM_PROMPT}]
                if history:
                    for h in history[-10:]:
                        role = (h.get("role") or "user").strip().lower()
                        if role not in ("user", "assistant"):
                            role = "user"
                        content = (h.get("content") or "").strip()
                        if content:
                            messages.append({"role": role, "content": content})
                user_content = f"""Current question: {query}

Consider the conversation so far. If the user's message is a follow-up (e.g. "Is this all?", "What about the price?", "Tell me more"), interpret it in light of your previous answer and respond like a human would. Provide a full, helpful answer using your general knowledge. If the question is about flight, aviation, or aircraft, give a complete answer. Start with a brief note that this is not from Hye Aero's database if relevant, then give the full answer."""
                messages.append({"role": "user", "content": user_content})
                try:
                    for d in self._stream_chat_deltas(messages, max_tokens=1536):
                        yield {"type": "delta", "text": d}
                    yield {"type": "done", "sources": [], "data_used": {}, "error": None}
                except Exception as gk_e:
                    logger.error("RAG stream (general knowledge) failed: %s", gk_e, exc_info=True)
                    yield {
                        "type": "done",
                        "sources": [],
                        "data_used": {},
                        "error": str(gk_e),
                    }
                return

            b = payload
            context = b["context"]
            phly_meta = b["phly_meta"]
            results = b["results"]
            tavily_hits = b["tavily_hits"]
            data_used: Dict[str, Any] = dict(b["data_used"])

            yield {"type": "status", "message": "Searching sources and drafting…"}

            messages = [{"role": "system", "content": b["system_prompt"]}]
            hist = b.get("history")
            if hist:
                for h in hist[-10:]:
                    role = (h.get("role") or "user").strip().lower()
                    if role not in ("user", "assistant"):
                        role = "user"
                    content = (h.get("content") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
            user_content = f"""Consider the full conversation above and the layered context below (internal PhlyData/FAA if present, then web search, then private vector DB). If the user's message is a follow-up, interpret it in light of your previous answer.

Synthesize a professional draft: use PhlyData/FAA as ground truth for identity and registrant when that block exists; use web snippets only as secondary hints (label uncertainty); use vector DB for listings/sales/market color when consistent with PhlyData.

Context:
{context}

Current question: {b["query"]}

Provide a thorough draft answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})

            review_disabled = (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            )

            import openai

            sync_client = openai.OpenAI(api_key=self.openai_api_key, timeout=120.0)
            draft = ""
            if not review_disabled:
                response = sync_client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=1536,
                )
                draft = (response.choices[0].message.content or "").strip()
                yield {"type": "status", "message": "Polishing answer…"}
                rev_messages = [
                    {"role": "system", "content": CONSULTANT_REVIEW_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""User question:
{b["query"]}

Draft answer from first pass:
{draft}

Layered context (same as the drafter):
{context}

Produce the final client-facing answer.""",
                    },
                ]
                try:
                    for d in self._stream_chat_deltas(rev_messages, max_tokens=1536, temperature=0.2):
                        yield {"type": "delta", "text": d}
                except Exception as rev_e:
                    logger.warning("Consultant stream review failed, falling back to draft chunks: %s", rev_e)
                    for piece in self._iter_display_chunks(draft):
                        yield {"type": "delta", "text": piece}
            else:
                for d in self._stream_chat_deltas(messages, max_tokens=1536):
                    yield {"type": "delta", "text": d}

            data_used["final_review_pass"] = not review_disabled
            sources = self._consultant_sources_list(phly_meta, tavily_hits, results)
            elapsed = time.perf_counter() - start
            logger.info(
                "RAG stream: query_len=%d phly=%s tavily_hits=%d pinecone_sources=%d elapsed=%.2fs",
                len(query),
                bool(b.get("phly_authority")),
                tavily_hits,
                len(results),
                elapsed,
            )
            yield {"type": "done", "sources": sources, "data_used": data_used, "error": None}
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("RAG answer_stream_events failed after %.2fs: %s", elapsed, e, exc_info=True)
            yield {"type": "error", "message": str(e)}
            yield {"type": "done", "sources": [], "data_used": {}, "error": str(e)}

    def answer(
        self,
        query: str,
        top_k: int = 20,
        max_context_chars: int = 14000,
        score_threshold: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Ask Consultant pipeline: PhlyData (Postgres) authority → LLM query expand → Tavily (web) →
        multi-query Pinecone RAG → draft LLM → optional review LLM. Falls back to general knowledge
        only when there is no usable context at all.
        """
        start = time.perf_counter()
        try:
            kind, payload = self._consultant_retrieval_bundle(
                query,
                top_k,
                max_context_chars,
                score_threshold,
                history,
            )
            if kind == "professional":
                return payload
            if kind == "gk":
                return self._answer_from_general_knowledge(query, start, history=history)

            b = payload
            context = b["context"]
            phly_meta = b["phly_meta"]
            results = b["results"]
            tavily_hits = b["tavily_hits"]
            data_used: Dict[str, Any] = dict(b["data_used"])

            import openai

            client = openai.OpenAI(api_key=self.openai_api_key, timeout=90.0)
            messages = [{"role": "system", "content": b["system_prompt"]}]
            hist = b.get("history")
            if hist:
                for h in hist[-10:]:
                    role = (h.get("role") or "user").strip().lower()
                    if role not in ("user", "assistant"):
                        role = "user"
                    content = (h.get("content") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
            user_content = f"""Consider the full conversation above and the layered context below (internal PhlyData/FAA if present, then web search, then private vector DB). If the user's message is a follow-up, interpret it in light of your previous answer.

Synthesize a professional draft: use PhlyData/FAA as ground truth for identity and registrant when that block exists; use web snippets only as secondary hints (label uncertainty); use vector DB for listings/sales/market color when consistent with PhlyData.

Context:
{context}

Current question: {b["query"]}

Provide a thorough draft answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1536,
            )
            draft = (response.choices[0].message.content or "").strip()

            review_disabled = (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            answer = draft
            if draft and not review_disabled:
                try:
                    rev_messages = [
                        {"role": "system", "content": CONSULTANT_REVIEW_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"""User question:
{b["query"]}

Draft answer from first pass:
{draft}

Layered context (same as the drafter):
{context}

Produce the final client-facing answer.""",
                        },
                    ]
                    rev = client.chat.completions.create(
                        model=self.chat_model,
                        messages=rev_messages,
                        max_tokens=1536,
                        temperature=0.2,
                    )
                    reviewed = (rev.choices[0].message.content or "").strip()
                    if reviewed:
                        answer = reviewed
                except Exception as rev_e:
                    logger.warning("Consultant final review skipped: %s", rev_e)

            elapsed = time.perf_counter() - start
            sources = self._consultant_sources_list(phly_meta, tavily_hits, results)
            data_used["final_review_pass"] = not review_disabled

            logger.info(
                "RAG answer: query_len=%d phly=%s tavily_hits=%d pinecone_sources=%d answer_len=%d elapsed=%.2fs",
                len(query),
                bool(b.get("phly_authority")),
                tavily_hits,
                len(results),
                len(answer),
                elapsed,
            )
            return {
                "answer": answer,
                "sources": sources,
                "data_used": data_used,
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("RAG answer failed after %.2fs: %s", elapsed, e, exc_info=True)
            return {
                "answer": "",
                "sources": [],
                "data_used": {},
                "error": str(e),
            }

    def _professional_search_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Deterministic answers for “aggregate/list” style questions.
        This avoids LLM guessing from limited Pinecone snippets.
        """
        q = (query or "").strip()
        if not q:
            return None

        q_l = q.lower()

        # Intent detection (AircraftPost / Aviacost / FAA / listings / internal DB)
        wants_aircraftpost = ("aircraftpost" in q_l) or ("fleet" in q_l and "aircraftpost" in q_l)
        # Use regex so punctuation like "models." still matches.
        wants_models = bool(re.search(r"\bmodels?\b", q_l)) or ("model list" in q_l) or ("models of" in q_l)
        wants_serials = bool(re.search(r"\bserials?\b", q_l)) or ("serial number" in q_l) or ("serial numbers" in q_l)
        wants_for_sale_rate = ("for-sale rate" in q_l) or ("for sale rate" in q_l) or ("forsale rate" in q_l)
        wants_fleet_count = any(k in q_l for k in ["how many", "number of", "records included", "fleet records", "count"])
        wants_for_sale_only = ("for sale=true" in q_l) or ("forsale=true" in q_l) or ("for sale models" in q_l)
        wants_for_sale_data = ("for sales data" in q_l) or ("for sale data" in q_l) or ("available for sale" in q_l) or ("for sale" in q_l and wants_serials)

        wants_aviacost = "aviacost" in q_l
        wants_faa = ("faa" in q_l) or ("registrant" in q_l) or ("faa registry" in q_l)
        wants_listings = ("listing" in q_l) or ("dealer" in q_l) or ("seller" in q_l) or ("craftexchange" in q_l) or ("controller" in q_l)
        wants_internal_sales = ("sale" in q_l) or ("sold_price" in q_l) or ("sold price" in q_l)

        # Extract make/model from phrases like: "For Embraer Phenom 100, ..."
        make_model = self._extract_make_model_from_query(q)

        # AircraftPost: models list (distinct make_model_name)
        if wants_aircraftpost and ("models" in q_l or "model list" in q_l or "models of" in q_l) and not make_model:
            for_sale_filter = wants_for_sale_only or ("for sale" in q_l and "models" in q_l)
            rows = self.db.execute_query(
                """
                SELECT DISTINCT make_model_name
                FROM aircraftpost_fleet_aircraft
                WHERE make_model_name IS NOT NULL AND TRIM(make_model_name) <> ''
                {for_sale_clause}
                ORDER BY make_model_name
                LIMIT 60
                """.format(for_sale_clause="AND for_sale IS TRUE" if for_sale_filter else ""),
            )
            models = [r.get("make_model_name") for r in rows if r.get("make_model_name")]
            ans_lines = ["AircraftPost models (distinct make_model_name):"]
            ans_lines.extend([f"- {m}" for m in models])
            return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"aircraftpost_models": len(models)}, "error": None}

        # AircraftPost: exact aggregates by make/model (and optional for_sale filter)
        if wants_aircraftpost and make_model and (wants_for_sale_rate or wants_fleet_count or wants_serials or wants_models or wants_for_sale_data):
            mfr, mdl = make_model
            if not mfr or not mdl:
                return None

            for_sale_clause = "AND for_sale IS TRUE" if (wants_for_sale_only or wants_for_sale_data) else ""
            # Total record count and for-sale count
            if wants_for_sale_rate or wants_fleet_count:
                count_rows = self.db.execute_query(
                    f"""
                    SELECT
                      COUNT(*) AS total_records,
                      SUM(CASE WHEN for_sale IS TRUE THEN 1 ELSE 0 END) AS for_sale_count
                    FROM aircraftpost_fleet_aircraft
                    WHERE make_model_name ILIKE %s AND make_model_name ILIKE %s
                    {for_sale_clause}
                    """,
                    (f"%{mfr}%", f"%{mdl}%"),
                )
                r = count_rows[0] if count_rows else {}
                total_records = int(r.get("total_records") or 0)
                # If we filtered by for_sale=true in query, for_sale_count==total_records
                for_sale_count = int(r.get("for_sale_count") or 0)
                for_sale_rate = (for_sale_count / total_records) if total_records else None

                ans_lines = [f"AircraftPost fleet summary for {mfr} {mdl}:"]
                ans_lines.append(f"- Fleet records: {total_records}")
                ans_lines.append(f"- For-sale records: {for_sale_count}")
                if for_sale_rate is not None:
                    ans_lines.append(f"- For-sale rate: {for_sale_rate * 100:.2f}%")
                return {
                    "answer": "\n".join(ans_lines),
                    "sources": [],
                    "data_used": {"aircraftpost_fleet_total": total_records, "aircraftpost_fleet_for_sale": for_sale_count},
                    "error": None,
                }

            # Models list for a specific make/model (optionally for sale)
            if wants_models:
                rows = self.db.execute_query(
                    f"""
                    SELECT DISTINCT make_model_name, COUNT(*) AS n
                    FROM aircraftpost_fleet_aircraft
                    WHERE make_model_name ILIKE %s AND make_model_name ILIKE %s
                    {for_sale_clause}
                    GROUP BY make_model_name
                    ORDER BY n DESC
                    LIMIT 30
                    """,
                    (f"%{mfr}%", f"%{mdl}%"),
                )
                ans_lines = [f"AircraftPost models matched for {mfr} {mdl}:"]
                ans_lines.extend([f"- {r.get('make_model_name')}" for r in rows if r.get("make_model_name")])
                return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"models_count": len(rows)}, "error": None}

            # Serial numbers list (optionally for sale)
            if wants_serials or wants_for_sale_data:
                # Return all distinct serial_numbers for the make/model (and optional for_sale filter).
                rows = self.db.execute_query(
                    f"""
                    SELECT DISTINCT TRIM(serial_number) AS serial_number
                    FROM aircraftpost_fleet_aircraft
                    WHERE make_model_name ILIKE %s AND make_model_name ILIKE %s
                    {for_sale_clause}
                      AND serial_number IS NOT NULL AND TRIM(serial_number) <> ''
                    ORDER BY serial_number
                    LIMIT 500
                    """,
                    (f"%{mfr}%", f"%{mdl}%"),
                )
                serials = [r.get("serial_number") for r in rows if r.get("serial_number") is not None]
                ans_lines = [f"AircraftPost serial numbers for {mfr} {mdl}:"]
                if not serials:
                    ans_lines.append("- No matching records found.")
                else:
                    ans_lines.extend([f"- {s}" for s in serials])
                return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"serials_returned": len(serials)}, "error": None}

        # Internal DB / synced master: serial numbers for a specific make/model (non-AircraftPost)
        if (not wants_aircraftpost) and make_model and wants_serials:
            mfr, mdl = make_model
            rows = self.db.execute_query(
                """
                SELECT serial_number, registration_number, manufacturer_year, based_at, based_country
                FROM aircraft
                WHERE manufacturer ILIKE %s
                  AND model ILIKE %s
                  AND serial_number IS NOT NULL AND TRIM(serial_number) <> ''
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 30
                """,
                (f"%{mfr}%", f"%{mdl}%"),
            )
            ans_lines = [f"Serial numbers for {mfr} {mdl} (latest up to 30):"]
            if not rows:
                ans_lines.append("- No matching records found.")
            else:
                for r in rows:
                    sn = r.get("serial_number")
                    reg = r.get("registration_number")
                    year = r.get("manufacturer_year")
                    base = r.get("based_at")
                    ans_lines.append(
                        f"- {sn} (Reg: {reg or '—'}{f', Year: {year}' if year else ''}{f', Base: {base}' if base else ''})"
                    )
            return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"serials_returned": len(rows or [])}, "error": None}

        # Listings: count for_sale listings for a make/model
        # Accept "for sale", "for-sale", and "for_sale" spellings.
        wants_for_sale_any = bool(re.search(r"\bfor[- ]?sale\b", q_l))
        if make_model and wants_listings and wants_fleet_count and wants_for_sale_any:
            mfr, mdl = make_model
            rows = self.db.execute_query(
                """
                SELECT COUNT(*) AS listings_count
                FROM aircraft_listings l
                JOIN aircraft a ON a.id = l.aircraft_id
                WHERE l.listing_status = 'for_sale'
                  AND a.manufacturer ILIKE %s
                  AND a.model ILIKE %s
                """,
                (f"%{mfr}%", f"%{mdl}%"),
            )
            n = int(rows[0].get("listings_count") or 0) if rows else 0
            return {
                "answer": f"Number of for-sale listings for {mfr} {mdl}: {n}.",
                "sources": [],
                "data_used": {"for_sale_listings": n},
                "error": None,
            }

        # Models list for non-AircraftPost queries: provide internal sales models (sold_price > 0)
        if (not wants_aircraftpost) and ("models" in q_l or "model list" in q_l or "models of" in q_l) and not make_model:
            rows = self.db.execute_query(
                """
                SELECT DISTINCT manufacturer, model
                FROM aircraft_sales
                WHERE sold_price IS NOT NULL AND sold_price > 0
                  AND (COALESCE(manufacturer,'') <> '' OR COALESCE(model,'') <> '')
                ORDER BY manufacturer, model
                LIMIT 60
                """
            )
            models = []
            for r in rows:
                man = (r.get("manufacturer") or "").strip()
                mod = (r.get("model") or "").strip()
                if man or mod:
                    label = f"{man} {mod}".strip()
                    if label:
                        models.append(label)
            ans_lines = ["Models (from internal aircraft_sales with sold_price > 0):"]
            ans_lines.extend([f"- {m}" for m in models])
            return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"models_returned": len(models)}, "error": None}

        # Aviacost exact lookup: ask for operating cost reference fields
        if wants_aviacost and make_model:
            mfr, mdl = make_model
            av = lookup_aviacost(self.db, manufacturer=mfr, model=mdl)
            if av:
                ans_lines = [f"Aviacost operating cost reference for {av.get('name') or (mfr + ' ' + mdl)}:"]
                if av.get("variable_cost_per_hour") is not None:
                    ans_lines.append(f"- Variable cost/hr: ${av['variable_cost_per_hour']:,.2f}")
                if av.get("average_pre_owned_price") is not None:
                    ans_lines.append(f"- Avg pre-owned price: ${(av['average_pre_owned_price'] / 1_000_000):.2f}M")
                if av.get("fuel_gallons_per_hour") is not None:
                    ans_lines.append(f"- Fuel: {av['fuel_gallons_per_hour']} gal/hr")
                if av.get("normal_cruise_speed_kts") is not None:
                    ans_lines.append(f"- Cruise: {av['normal_cruise_speed_kts']} kts")
                return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"aviacost_lookup": True}, "error": None}

        # FAA exact lookup by model (if question mentions registrant)
        if wants_faa and make_model:
            mfr, mdl = make_model
            rows = self.db.execute_query(
                """
                SELECT f.registrant_name,
                       f.city, f.state, f.country,
                       f.street, f.zip_code
                FROM faa_registrations f
                JOIN aircraft a ON a.id = f.aircraft_id
                WHERE a.manufacturer ILIKE %s AND a.model ILIKE %s
                  AND f.registrant_name IS NOT NULL AND TRIM(f.registrant_name) <> ''
                ORDER BY f.ingestion_date DESC
                LIMIT 10
                """,
                (f"%{mfr}%", f"%{mdl}%"),
            )
            if rows:
                ans_lines = [f"FAA registrant(s) for {mfr} {mdl}: (latest up to 10)"]
                for r in rows:
                    reg = r.get("registrant_name")
                    city = r.get("city")
                    state = r.get("state")
                    country = r.get("country")
                    loc = ", ".join([x for x in [city, state, country] if x])
                    ans_lines.append(f"- {reg}" + (f" ({loc})" if loc else ""))
                return {"answer": "\n".join(ans_lines), "sources": [], "data_used": {"faa_registrants_found": len(rows)}, "error": None}

        return None

    @staticmethod
    def _extract_make_model_from_query(query: str) -> Optional[tuple]:
        """
        Extract (manufacturer, model) from strings like:
          "For Embraer Phenom 100, ..."
          "For Pilatus PC-24, ..."
        Returns None if it can't find a usable phrase.
        """
        q = (query or "").strip()
        if not q:
            return None

        # Prefer patterns that include "For <make> <model>"
        m = re.search(r"\bfor\s+([^,?\n]+)", q, flags=re.IGNORECASE)
        if not m:
            return None
        phrase = m.group(1).strip()

        # Clean common separators
        phrase = re.sub(r"\b(aircraftpost|aviacost|faa)\b", "", phrase, flags=re.IGNORECASE).strip()
        phrase = phrase.strip(" -–—\t")

        # Split manufacturer (first word) and model (rest)
        parts = phrase.split(None, 1)
        if len(parts) < 2:
            return None
        mfr = parts[0].strip()
        mdl = parts[1].strip()
        if not mfr or not mdl:
            return None
        return (mfr, mdl)

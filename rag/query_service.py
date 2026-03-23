"""RAG query service: PhlyData (Postgres) + Tavily (web) + LLM query expand + Pinecone RAG + two-pass LLM answer."""

import logging
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Iterator, Tuple

from rag.embedding_service import EmbeddingService
from rag.entity_extractors import EXTRACTORS
from vector_store.pinecone_client import PineconeClient
from database.postgres_client import PostgresClient
from services.aviacost_lookup import lookup_aviacost

logger = logging.getLogger(__name__)


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


# Minimum similarity score to include a Pinecone match (cosine: higher = more similar)
DEFAULT_SCORE_THRESHOLD = 0.5

CONSULTANT_SYSTEM_PROMPT = """You are Hye Aero's Aircraft Research & Valuation Consultant. You think like a human expert: consider the full conversation, remember what you already said, and answer the current question in context. Your answers should be at least as complete and useful as a top-tier web assistant: combine Hye Aero internal data with web search and vector context when the question needs current owner/operator, fleet, or registry facts.

Your process:
- Understand what the user is really asking (ownership? listings? sales? model specs? valuation?).
- Search mentally through ALL layers: PhlyData/FAA block, Tavily web snippets (titles, bodies, URLs), and vector DB (listings, sales, registry-related chunks).
- Synthesize: lead with the clearest, most specific answer; then support with identity lines, sources, and caveats.

Rules:
- Aircraft identity (serial, tail/registration as shown, make/model, year): treat the PhlyData + FAA block as authoritative when it lists those fields. Do not contradict them with web or vector text.
- **Ownership-only** (who owns / registrant / operator — user did **not** ask price, buy, listing, or for sale): Lead with **FAA/PhlyData registrant** and any Tavily-backed operator facts. **Do not** open with "active listing" or asking price. If INTERNAL market or Tavily also shows a listing for the same serial/tail, add a **short closing section** after ownership, e.g. "Market note (separate from legal registrant):" with ask price + source/URL — clearly secondary.
- Ownership / operator / "who owns" questions:
  - If the block includes an FAA MASTER registrant name, report it as the U.S. FAA registrant record and still add web/listing context if it adds operator or fleet detail.
  - If the block states there is NO FAA registrant row (typical for non-U.S. primary registry, e.g. tails not starting with N-), FAA is not the state of registry. You MUST lean heavily on Tavily web results (and vector snippets) to name who **operates** or **commercially manages** the aircraft today — same quality bar as ChatGPT: fleet pages, AOC holders, charter operators, and registry excerpts that mention this exact tail/serial.
  - Legal registered owner vs operator: European and charter jets often show one company on a national register and another on the operator’s fleet or charter website. If Tavily ties this registration to a charter/airline/management brand (e.g. fleet list showing this tail), say that clearly as the operating party and mention the registry/legal line only if snippets support it.
  - **Every company name you state as current owner or operator must appear verbatim (or as an obvious substring) in a Tavily snippet title or body.** Cite which result number (1., 2., …) or the domain/URL you used. If snippets disagree, give both names and say what each source claims. If no snippet names a company, say web results did not clearly identify an owner/operator — do not guess.
  - Never invent registry or database names (do not say "Danish Aircraft Database" unless that exact phrase appears in a snippet).
- Valuations and comparisons: cite specific numbers from context. If something is unknown, say so.
- Purchase / availability / "can I buy" / pricing / "how much" / "is it for sale":
  - **Do not omit price when the context contains one.** Structure the answer like a deal brief: after identity, include a **Market / listing** subsection with:
    (1) **Asking price** (or **Sold price** if only a past sale is shown) with currency (assume USD if unstated in our DB);
    (2) **Source** — e.g. "Internal listing (Hye Aero DB): {platform}" or "Web: {site name} — URL from Tavily result #N";
    (3) **Availability** — listed for sale / status line / "no current listing found in provided sources".
  - INTERNAL market block: copy **Ask:** and **Listing URL:** lines literally when present; they override vague web blurbs.
  - Tavily / web: read snippet bodies for **$ amounts** and **for sale** language; quote the price and cite the snippet index + domain. If the user asks whether they can buy **now**, a current listing URL + ask in context means "yes, subject to verifying with seller" — still state the price and source.
  - Comparable sales section: if no current ask but comps exist, give **low / high / avg** from the internal comp summary and label as **recent sale comps (not a live listing)**.
  - If a **[WEB — Dollar amounts spotted in Tavily snippet text]** section appears after the Tavily list, treat every line as a mandatory price hint: repeat each amount in your **Market / listing** bullets and tie it to the matching snippet # and URL/domain.
  - If truly no price anywhere in context, say explicitly **"No asking price in the provided internal data or web snippets"** — do not invent numbers.
- Voice: Answer like a senior broker or research lead talking to a client — direct, conversational sentences. Avoid robotic closings ("feel free to ask", "let me know if you need anything else"), stacked machine-style sections, or repeating the same disclaimer. Use bullets only when they genuinely help scanning; short paragraphs are fine.
- **Listing URLs (critical):** Never cite a listing URL from Tavily or the vector DB unless that same snippet/chunk explicitly ties the URL to the **same** serial number and/or tail as the authoritative PhlyData + FAA block. If the only URLs in context are for a different aircraft (e.g. another Citation), say clearly that no matching listing link for **this** serial/tail appeared — do not paste unrelated listings.
- Use clear bullets (-) when useful. Neutral, professional tone for brokers and clients. You may use tasteful emoji (e.g. ✈ 🧾) when it improves scanability.
- Format: no markdown # headers or ** bold.

Context layers (in order):
1) AUTHORITATIVE PhlyData + FAA MASTER — source of truth for identity; FAA registrant line only when present in the block.
2) INTERNAL market (Postgres listings + comparable sales) — authoritative for **prices and listing URLs** in our database when this block appears.
3) Web search (Tavily) — current off-platform listings and operator/owner color; attribute each major claim (title/URL).
4) Vector database — extra listings/sales/registry chunks; corroboration."""

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
- Use INTERNAL market block for prices/URLs when present; require draft to mention ask/sold figures if the block contains them. If you see **[FOR USER REPLY — Market / pricing]**, those lines are mandatory to reflect near the top (exact $ and URLs).
- Purchase / price questions: final answer MUST include a **Market / listing** style block with **asking price (or clear "not stated")** and **source** (internal vs web + URL hint). If draft omitted a price that appears in INTERNAL, Tavily bodies, or the **Dollar amounts spotted** appendix, add it with snippet # / domain.
- Use vector DB for listings/sales corroboration when helpful.
- Improve structure: short lead, then facts in natural prose or light bullets. Optional one short attribution line if helpful — not a long "Sources:" footer.
- **Listing URLs:** Never output a listing URL unless context proves it belongs to the **same** serial/tail as PhlyData/FAA. Strip wrong-jet URLs from the draft.
- No markdown # or ** bold. Plain bullets (-) only when they add clarity.
- Stay factual; do not fabricate URLs or companies not implied by context."""

# Appended to user messages when the question is purchase / price / availability — forces deal-brief structure.
CONSULTANT_PURCHASE_USER_DIRECTIVES = """
PURCHASE / PRICE / AVAILABILITY: Answer in a natural consultant voice — short opening, then the deal facts. Do not sound like a checklist robot.

- Lead with whether you see evidence of a **current listing for this exact aircraft** (same serial and/or tail as the PhlyData authority block). A random Citation listing for another serial does **not** count as "yes."
- **Listing URLs:** Include a URL only if INTERNAL market lines or the same numbered Tavily/vector snippet clearly ties it to **this** serial/tail. Never paste a marketplace URL that refers to a different aircraft.
- State asking price when it clearly applies to **this** aircraft (figure + where it came from). If none, say so plainly. You may use a short labeled line for price when it helps.
- If only comps exist (no live ask), say that and give the comp range from internal summary.
- Skip hollow closings ("feel free to ask", etc.)."""


def _consultant_purchase_tail(bundle: Dict[str, Any]) -> str:
    return CONSULTANT_PURCHASE_USER_DIRECTIVES if bundle.get("purchase_context") else ""


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

    @staticmethod
    def _identity_norm_alnum(value: Any) -> str:
        s = (value if value is not None else "") or ""
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    def _phly_identity_sets(
        self, phly_rows: List[Dict[str, Any]]
    ) -> Tuple[set, set]:
        """Normalized serials and tails from PhlyData authority rows (for strict RAG/Tavily matching)."""
        serials: set = set()
        tails: set = set()
        for r in phly_rows or []:
            sn = self._identity_norm_alnum(r.get("serial_number"))
            if len(sn) >= 4:
                serials.add(sn)
            tg = self._identity_norm_alnum(r.get("registration_number"))
            if len(tg) >= 2:
                tails.add(tg)
        return serials, tails

    def _entity_serial_tail_for_filter(
        self, entity_type: str, entity_id: str
    ) -> Tuple[str, str]:
        """Best-effort serial + registration for Pinecone-linked entities when PhlyData has a canonical aircraft."""
        record = self._fetch_full_record(entity_type, entity_id)
        if not record:
            return "", ""
        if entity_type == "aircraft":
            return (
                str(record.get("serial_number") or ""),
                str(record.get("registration_number") or ""),
            )
        if entity_type in ENTITY_HAS_AIRCRAFT_ID:
            aid = record.get("aircraft_id")
            if aid:
                ac = self._fetch_aircraft_by_id(str(aid))
                if ac:
                    return (
                        str(ac.get("serial_number") or ""),
                        str(ac.get("registration_number") or ""),
                    )
        if entity_type == "faa_registration":
            return (
                str(record.get("serial_number") or ""),
                str(record.get("registration_number") or ""),
            )
        return (
            str(record.get("serial_number") or ""),
            str(record.get("registration_number") or ""),
        )

    def _rag_chunk_matches_phly_identity(
        self,
        entity_type: str,
        entity_id: str,
        serial_norms: set,
        tail_norms: set,
        cache: Dict[Tuple[str, str], Tuple[str, str]],
    ) -> bool:
        if not serial_norms and not tail_norms:
            return True
        et = (entity_type or "").strip()
        if et not in (
            "aircraft_listing",
            "aircraft_sale",
            "faa_registration",
            "aircraft",
        ):
            return True
        key = (et, str(entity_id))
        if key not in cache:
            cache[key] = self._entity_serial_tail_for_filter(et, str(entity_id))
        sn_s, reg_s = cache[key]
        ns = self._identity_norm_alnum(sn_s)
        nr = self._identity_norm_alnum(reg_s)
        if tail_norms and nr and nr in tail_norms:
            return True
        if serial_norms and ns and ns in serial_norms:
            return True
        return False

    def _filter_rag_results_for_phly_aircraft(
        self,
        results: List[Dict[str, Any]],
        phly_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Drop vector hits for listings/sales/registrations/aircraft rows that do not match
        PhlyData serial/tail — prevents surfacing another jet's listing URL (semantic near-miss).
        """
        serial_norms, tail_norms = self._phly_identity_sets(phly_rows)
        if not phly_rows or (not serial_norms and not tail_norms):
            return results
        cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        kept: List[Dict[str, Any]] = []
        dropped = 0
        for r in results or []:
            et = (r.get("entity_type") or "").strip()
            eid = r.get("entity_id")
            if eid is None or str(eid).strip() == "":
                kept.append(r)
                continue
            if self._rag_chunk_matches_phly_identity(
                et, str(eid), serial_norms, tail_norms, cache
            ):
                kept.append(r)
            else:
                dropped += 1
        if dropped:
            logger.info(
                "RAG: dropped %s chunks not matching PhlyData serial/tail (kept %s)",
                dropped,
                len(kept),
            )
        return kept

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
        max_query_variants: int = 5,
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
        cap = max(1, min(8, int(max_query_variants)))
        nq = min(len(uniq_q), cap)
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
        from rag.consultant_market_lookup import (
            build_consultant_market_authority_block,
            build_purchase_listing_tavily_query,
            enrich_rag_queries_for_purchase,
            filter_tavily_results_for_phly_identity,
            strip_market_meta_zeros,
            tavily_price_highlights_block,
            wants_consultant_purchase_market_context,
        )
        from services.tavily_owner_hint import fetch_tavily_hints_for_query

        hs = self._consultant_history_snippet(history)
        hs_opt = hs.strip() or None

        fast_retrieval = _env_truthy("CONSULTANT_FAST_RETRIEVAL")
        skip_expand = _env_truthy("CONSULTANT_SKIP_QUERY_EXPAND")
        single_tavily_pass = _env_truthy("CONSULTANT_TAVILY_SINGLE_PASS") or fast_retrieval

        try:
            tavily_per_pass = int((os.getenv("CONSULTANT_TAVILY_RESULTS_PER_PASS") or "8").strip())
            tavily_per_pass = max(4, min(10, tavily_per_pass))
        except ValueError:
            tavily_per_pass = 8
        if fast_retrieval:
            tavily_per_pass = min(tavily_per_pass, 6)

        try:
            max_rag_variants = int((os.getenv("CONSULTANT_RAG_QUERY_VARIANTS") or "5").strip())
            max_rag_variants = max(1, min(8, max_rag_variants))
        except ValueError:
            max_rag_variants = 5
        if fast_retrieval:
            max_rag_variants = min(max_rag_variants, 3)

        try:
            enrich_rag_max = int((os.getenv("CONSULTANT_RAG_ENRICH_MAX") or "8").strip())
            enrich_rag_max = max(3, min(12, enrich_rag_max))
        except ValueError:
            enrich_rag_max = 8
        if fast_retrieval:
            enrich_rag_max = min(enrich_rag_max, 5)

        try:
            rag_max_chunks = int((os.getenv("CONSULTANT_RAG_MAX_CHUNKS") or "18").strip())
            rag_max_chunks = max(8, min(24, rag_max_chunks))
        except ValueError:
            rag_max_chunks = 18
        if fast_retrieval:
            rag_max_chunks = min(rag_max_chunks, 14)

        try:
            tavily_timeout = float((os.getenv("CONSULTANT_TAVILY_TIMEOUT_SEC") or "28").strip())
            tavily_timeout = max(8.0, min(60.0, tavily_timeout))
        except ValueError:
            tavily_timeout = 28.0

        if skip_expand:
            phly_authority, phly_meta, phly_rows = self._phlydata_authority_block(query, history)
            qstrip = (query or "").strip()
            expanded = {
                "tavily_query": qstrip[:400] if qstrip else "",
                "rag_queries": [qstrip] if qstrip else [""],
            }
        else:
            def _run_phly() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
                return self._phlydata_authority_block(query, history)

            def _run_expand() -> Dict[str, Any]:
                return expand_consultant_research_queries(
                    query,
                    self.openai_api_key or "",
                    self.chat_model,
                    history_snippet=hs_opt,
                )

            with ThreadPoolExecutor(max_workers=2) as pre_pool:
                f_phly = pre_pool.submit(_run_phly)
                f_exp = pre_pool.submit(_run_expand)
                phly_authority, phly_meta, phly_rows = f_phly.result()
                expanded = f_exp.result()

        market_block, market_meta = build_consultant_market_authority_block(
            self.db, query, history, phly_rows
        )
        rag_qs = enrich_rag_queries_for_purchase(
            list(expanded.get("rag_queries") or [query]),
            query,
            history,
            phly_rows,
            max_total=enrich_rag_max,
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

        sq = build_owner_operator_focus_tavily_query(query, phly_rows, history_snippet=hs_opt)
        sq_c = " ".join(sq.split()).lower() if sq else ""
        tq_c = " ".join(tq.split()).lower()
        run_secondary = bool(sq and sq_c != tq_c)
        pq = build_purchase_listing_tavily_query(query, history, phly_rows)
        pq_c = " ".join(pq.split()).lower() if pq else ""
        merge_purchase = bool(pq and pq_c and pq_c not in {tq_c, sq_c})
        if single_tavily_pass:
            run_secondary = False
            merge_purchase = False
        tavily_passes = 1 + (1 if run_secondary else 0) + (1 if merge_purchase else 0)
        purchase_ctx = wants_consultant_purchase_market_context(query, history)
        tavily_max_items = 18 if purchase_ctx else 14
        tavily_body_chars = 2200 if purchase_ctx else 1400

        def _fetch_pri() -> Dict[str, Any]:
            return fetch_tavily_hints_for_query(
                tq,
                result_limit=tavily_per_pass,
                search_depth=tdepth,
                request_timeout=tavily_timeout,
            )

        def _fetch_sec() -> Optional[Dict[str, Any]]:
            if not run_secondary or not sq:
                return None
            return fetch_tavily_hints_for_query(
                sq,
                result_limit=tavily_per_pass,
                search_depth=tdepth,
                request_timeout=tavily_timeout,
            )

        def _fetch_pur() -> Optional[Dict[str, Any]]:
            if not merge_purchase or not pq:
                return None
            return fetch_tavily_hints_for_query(
                pq,
                result_limit=tavily_per_pass,
                search_depth=tdepth,
                request_timeout=tavily_timeout,
            )

        def _fetch_rag() -> List[Dict[str, Any]]:
            return self._retrieve_multi(
                rag_qs,
                top_k=top_k,
                score_threshold=score_threshold,
                max_results_total=rag_max_chunks,
                max_query_variants=max_rag_variants,
            )

        primary: Dict[str, Any] = {}
        secondary: Optional[Dict[str, Any]] = None
        tertiary: Optional[Dict[str, Any]] = None
        results: List[Dict[str, Any]] = []
        max_workers = 2 + (1 if run_secondary else 0) + (1 if merge_purchase else 0)
        max_workers = max(2, min(4, max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            f_pri = pool.submit(_fetch_pri)
            f_rag = pool.submit(_fetch_rag)
            f_sec = pool.submit(_fetch_sec) if run_secondary else None
            f_pur = pool.submit(_fetch_pur) if merge_purchase else None
            primary = f_pri.result()
            results = f_rag.result()
            if f_sec is not None:
                secondary = f_sec.result()
            if f_pur is not None:
                tertiary = f_pur.result()
        results = self._filter_rag_results_for_phly_aircraft(results, phly_rows)
        tavily_payload = primary
        if run_secondary and secondary is not None:
            tavily_payload = merge_tavily_consultant_payloads(
                tavily_payload, secondary, max_results=14
            )
        if merge_purchase and tertiary is not None:
            tavily_payload = merge_tavily_consultant_payloads(
                tavily_payload, tertiary, max_results=18
            )
        tavily_payload = filter_tavily_results_for_phly_identity(tavily_payload, phly_rows)

        purchase_tavily_merged = bool(merge_purchase and tertiary is not None)

        tavily_block = format_tavily_payload_for_consultant(
            tavily_payload, max_items=tavily_max_items, max_body_chars=tavily_body_chars
        )
        if purchase_ctx:
            ph = tavily_price_highlights_block(tavily_payload)
            if ph:
                tavily_block = f"{tavily_block}\n\n{ph}"
        tavily_hits = len(tavily_payload.get("results") or [])

        has_phly = bool((phly_authority or "").strip())
        has_market = bool((market_block or "").strip())
        has_rag = bool(results)
        has_tavily = tavily_hits > 0
        if not (has_phly or has_market or has_rag or has_tavily):
            logger.info(
                "Consultant: no PhlyData, no internal market block, no vector hits, no Tavily → general knowledge (len=%d)",
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
        _append(market_block)
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
        for k, v in strip_market_meta_zeros(market_meta).items():
            data_used[k] = v
        if fast_retrieval:
            data_used["consultant_fast_retrieval"] = 1
        if skip_expand:
            data_used["consultant_skip_query_expand"] = 1
        if single_tavily_pass:
            data_used["consultant_tavily_single_pass"] = 1
        data_used["consultant_pipeline"] = "phly_market_sql_tavily_expand_rag_parallel_v1"
        data_used["consultant_fast_mode"] = (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        data_used["tavily_results"] = tavily_hits
        data_used["tavily_web_query_passes"] = tavily_passes
        if purchase_tavily_merged:
            data_used["tavily_purchase_focus"] = 1
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
        if market_block:
            system_prompt += (
                "\n\nAn INTERNAL market block may list real asking/sold prices and listing URLs from Hye Aero's database. "
                "For purchase and pricing questions, prioritize those figures when answering and cite them as internal listings/sales."
            )
        if purchase_ctx:
            system_prompt += (
                "\n\nThis is a purchase/price/availability question: the user expects **asking price**, **source**, and **URL** "
                "like a broker brief. Follow [FOR USER REPLY] lines in the internal market block first."
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
{_consultant_purchase_tail(b)}
Provide a thorough draft answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})

            review_disabled = (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            ) or (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower() in (
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
{_consultant_purchase_tail(b)}
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
{_consultant_purchase_tail(b)}
Provide a thorough draft answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1536,
            )
            draft = (response.choices[0].message.content or "").strip()

            review_disabled = (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            ) or (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower() in (
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
{_consultant_purchase_tail(b)}
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

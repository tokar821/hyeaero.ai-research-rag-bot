"""RAG query service: vector search (Pinecone) → fetch details (PostgreSQL) → enrich with synced model data → LLM answer."""

import logging
import time
import re
from typing import List, Dict, Any, Optional

from rag.embedding_service import EmbeddingService
from rag.entity_extractors import EXTRACTORS
from vector_store.pinecone_client import PineconeClient
from database.postgres_client import PostgresClient
from services.aviacost_lookup import lookup_aviacost

logger = logging.getLogger(__name__)

# Minimum similarity score to include a Pinecone match (cosine: higher = more similar)
DEFAULT_SCORE_THRESHOLD = 0.5

CONSULTANT_SYSTEM_PROMPT = """You are Hye Aero's Aircraft Research & Valuation Consultant. You think like a human expert: consider the full conversation, remember what you already said, and answer the current question in context. If the user asks a follow-up (e.g. "Is this all?", "What about the price?", "Any others?"), interpret it in light of your previous answer and respond like a human—e.g. "Yes, based on our database those are the three I found" or answer the implied question. You answer only from the provided context, which comes from Hye Aero's proprietary data: aircraft listings, historical sales, FAA registrations, and synced aircraft master/model data.

Your process:
- Understand what the user is really asking (listings? sales? a specific model? region? valuation?).
- Search mentally through the context for relevant listings, sales, aircraft, or FAA records.
- Decide which facts to use and how to combine them (e.g. compare, summarize, highlight one listing).
- Give a professional, human-like answer: clear, well-structured, with concrete numbers where the context provides them.

Rules:
- Base every claim on the context. If the context does not support an answer, say so clearly.
- When giving valuations or comparisons, cite specific numbers (prices, hours, years, locations) from the context.
- Do not invent data. If context is missing key details, say "Based on the data available..." and note limitations.
- Use clear sections or bullet points when comparing multiple aircraft or listing details. Lead with the most relevant finding, then support with detail. Use a neutral, advisory tone suitable for brokers and clients.
- Format: Do not use markdown headers (# ## ###) or double asterisks (**) for bold. Use plain bullet points (-) and plain text. You may use professional symbols or emoji like ChatGPT (e.g. •, ✓, →, or tasteful emoji where they add clarity)."""

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

    def answer(
        self,
        query: str,
        top_k: int = 20,
        max_context_chars: int = 14000,
        score_threshold: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG: (1) Search Pinecone; (2) if no results, answer from LLM general knowledge
        (e.g. flight concepts, theory, types of flight models—user wants full answers on flight-related topics).
        Returns dict with: answer, sources (list of retrieved items), data_used (summary), error (if any).
        """
        # For aggregate/list style questions we need exact numbers.
        # LLM answers from small Pinecone snippets are error-prone, so we route to
        # deterministic SQL first when intent is clearly “needs exact values”.
        prof_answer = self._professional_search_answer(query)
        if prof_answer:
            logger.info("Professional search triggered (deterministic SQL) for query=%r", query)
            return prof_answer

        start = time.perf_counter()
        try:
            results = self.retrieve(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                max_results=18,
            )
            if not results:
                logger.info("RAG answer: no DB results, using LLM fallback (general knowledge) for query (len=%d)", len(query))
                return self._answer_from_general_knowledge(query, start, history=history)
            # Build context for LLM (prefer full_context from Postgres, cap total size)
            context_parts = []
            total = 0
            for r in results:
                text = (r.get("full_context") or r.get("chunk_text") or "").strip()
                if not text:
                    continue
                if total + len(text) > max_context_chars:
                    text = text[: max_context_chars - total]
                context_parts.append(text)
                total += len(text)
                if total >= max_context_chars:
                    break
            context = "\n\n---\n\n".join(context_parts)

            # Summary of data used (for UI)
            data_used: Dict[str, int] = {}
            for r in results:
                et = (r.get("entity_type") or "other").replace("_", " ")
                data_used[et] = data_used.get(et, 0) + 1

            import openai
            client = openai.OpenAI(api_key=self.openai_api_key, timeout=60.0)
            messages = [{"role": "system", "content": CONSULTANT_SYSTEM_PROMPT}]
            if history:
                for h in history[-10:]:
                    role = (h.get("role") or "user").strip().lower()
                    if role not in ("user", "assistant"):
                        role = "user"
                    content = (h.get("content") or "").strip()
                    if content:
                        messages.append({"role": role, "content": content})
            user_content = f"""Consider the full conversation above and the context below. If the user's message is a follow-up (e.g. "Is this all?", "What about the price?", "Any others?"), interpret it in light of your previous answer and respond like a human—e.g. "Yes, based on our database those are the three I found" or answer the implied question. What is the user really asking? Search the context for relevant listings, sales, aircraft, or FAA data. Decide what matters, then synthesize a professional answer. Cite specific figures where the context provides them. Answer only from the context when referring to data.

Context from Hye Aero database:
{context}

Current question: {query}

Provide a professional, well-structured answer. Use plain text and bullet points (-). Do not use ** (double asterisks) or # headers. You may use professional symbols or emoji (e.g. •, ✓, →) like ChatGPT where appropriate."""
            messages.append({"role": "user", "content": user_content})
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1024,
            )
            answer = (response.choices[0].message.content or "").strip()
            elapsed = time.perf_counter() - start
            logger.info("RAG answer: query_len=%d sources=%d answer_len=%d elapsed=%.2fs", len(query), len(results), len(answer), elapsed)
            return {
                "answer": answer,
                "sources": [{"entity_type": r["entity_type"], "entity_id": str(r["entity_id"]) if r.get("entity_id") else None, "score": r.get("score")} for r in results],
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

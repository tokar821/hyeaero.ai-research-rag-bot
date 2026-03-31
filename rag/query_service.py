"""RAG query service: PhlyData + FAA first; then Tavily, LLM synthesis, listing ingests (Controller, exchanges, AircraftPost, AviaCost, …), vector DB."""

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
from rag.pinecone_metadata import infer_pinecone_entity_filter, legacy_meta_aircraft_model
from rag.semantic_reranker import SemanticRerankerService

logger = logging.getLogger(__name__)


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def _consultant_faa_no_phly_user_directive(phly_meta: Optional[Dict[str, Any]]) -> str:
    """
    Extra user-message instructions when Phly has no row but FAA MASTER matched — stops the drafter/reviewer
    from claiming make/model or ownership are unknown despite FAA lines in context.
    """
    if not phly_meta:
        return ""
    if not int(phly_meta.get("phlydata_no_row_for_tokens") or 0):
        return ""
    if not int(phly_meta.get("faa_master_owner_rows") or 0):
        return ""
    return (
        "\n\n**Answer structure (mandatory):** PhlyData has no internal export row for this tail, but **FAA MASTER** "
        "lines are in the context above. Open with FAA **registrant** and **mailing address** (verbatim where marked) "
        "and FAA **aircraft identity** (reference model, year, serial, type) when present. "
        "Do **not** state that make/model, year, or U.S. legal ownership are unknown or absent if those FAA lines are filled. "
        "Then briefly note PhlyData has no row; then add Tavily/vector/listing supplements with source labels.\n"
    )


def _consultant_no_phly_no_faa_snapshot_user_directive(phly_meta: Optional[Dict[str, Any]]) -> str:
    """
    When our ingested faa_master has no row (but the user cited a tail), force leading with Tavily — not a hollow
    \"nothing in the data\" answer.
    """
    if not phly_meta:
        return ""
    if not int(phly_meta.get("faa_internal_snapshot_miss") or 0):
        return ""
    return (
        "\n\n**Answer structure (mandatory):** PhlyData has no internal row, and **our ingested FAA MASTER snapshot** "
        "has no row for this tail in the context above. **Lead with Tavily web results** (and vector excerpts if any) "
        "for aircraft identity and U.S. registry/owner facts — cite snippet # and domain. "
        "Do **not** conclude that make/model, year, or ownership are \"not available\" if any Tavily snippet provides "
        "them. Avoid hollow closings (\"let me know if you have other queries\").\n"
    )


def _consultant_faa_no_phly_priority_prefix(phly_meta: Optional[Dict[str, Any]]) -> str:
    """
    When PhlyData has no row but FAA MASTER matched, force the draft model to open with FAA facts
    (some models anchor on the long [NO PHLYDATA ROW MATCH] paragraph and answer as if nothing exists).
    """
    if not phly_meta:
        return ""
    if not int(phly_meta.get("phlydata_no_row_for_tokens") or 0):
        return ""
    if not int(phly_meta.get("faa_master_owner_rows") or 0):
        return ""
    return (
        "[ANSWER ORDER — MANDATORY]\n"
        "1) If **AUTHORITATIVE — FAA MASTER** appears in this context, open with FAA registrant, mailing address, "
        "and aircraft identity lines (reference model, year_mfr, serial, type) from that block — verbatim where MANDATORY.\n"
        "2) Then briefly note that PhlyData (phlydata_aircraft) has no internal export row for this identifier.\n"
        "3) Then add Tavily / vector / listing supplements with clear source labels.\n\n"
    )


def _consultant_tavily_first_when_faa_ingest_miss_prefix(phly_meta: Optional[Dict[str, Any]]) -> str:
    """
    When Phly has no row and our ingested ``faa_master`` snapshot also has no row for a cited U.S. tail,
    force the model to lead on Tavily/vector (public registry–equivalent facts), not a hollow \"unknown\" brief.
    """
    if not phly_meta:
        return ""
    if not int(phly_meta.get("faa_internal_snapshot_miss") or 0):
        return ""
    return (
        "[ANSWER ORDER — MANDATORY — INGESTED FAA SNAPSHOT MISS]\n"
        "Our **internal** ``faa_master`` table did **not** return a row for this tail in this environment, but the user "
        "may still expect **public** U.S. registry / aircraft facts. **Lead with substantive lines from Tavily web results "
        "and vector excerpts** in this context (aircraft class, manufacturer/model family, year/serial if stated, "
        "registrant or operator when snippets support them). Cite **snippet #** and domain.\n"
        "Do **not** claim make/model, year, serial, or ownership are \"not available in the data gathered\" when any "
        "Tavily or vector line supports them. Briefly note PhlyData has no internal export row; avoid hollow closings.\n\n"
    )


# Minimum similarity score to include a Pinecone match (cosine: higher = more similar)
DEFAULT_SCORE_THRESHOLD = 0.5

CONSULTANT_SYSTEM_PROMPT = """You are Hye Aero's Aircraft Research & Valuation Consultant. You think like a senior broker and research lead: calm, precise, and trustworthy for business decisions. Consider the full conversation; answer the current question in context. Match the usefulness of a top-tier assistant (clear structure, plain language, no fluff) but **never trade accuracy for polish**.

**Accuracy and source priority (internal):**
- **Best-quality grounding:** When it appears in context, **PhlyData** (`phlydata_aircraft`) plus **FAA MASTER** (registrant/address in the same authority block) are Hye Aero's **primary** factual basis for identity, internal export fields, and U.S. legal registrant. Lead with and prioritize those over everything else when they apply.
- **When PhlyData / FAA do not cover the aircraft** (no row, non-U.S. registry without FAA, or gaps): still deliver an **excellent, comprehensive** answer by **synthesizing** **Tavily (web)**, **vector DB** excerpts, **Hye Aero listing ingests** in context (e.g. **Controller**, **Aircraft Exchange**, **AircraftPost**, **AviaCost**, and other marketplace listing tables), and **public.aircraft** when present — plus careful **LLM reasoning** only where it connects evidence already in context. **Label every substantive claim by source** (web snippet #, listing row, vector chunk, etc.); do not present listing or web data as PhlyData.
- **Serial numbers and registration (tail) numbers are unique as Hye Aero stores them** (Phly lookup: TRIM + UPPER only; **hyphens are literal**, so **LJ-1682** ≠ **LJ1682**, **525-0682** ≠ **5250682**). Illustrative examples only: **V-682** ≠ **682**; **XA-98723** ≠ **98723**; **0880** ≠ **880**. Never collapse hyphens, drop prefixes, or substitute a different spelling than the user or the Phly block shows.

**Hye Aero evaluation hierarchy (internal policy):**
- **PhlyData is Hye Aero's canonical internal record** for the aircraft: what the product treats as true for identity, internal export fields, and how you **frame the client's answer** when PhlyData is present.
- **Other layers** (synced marketplace listing rows, scraped listings in `aircraft_listings`, comparable sales, Tavily/web, vector DB) are for **search, context, and corroboration**. They **must not override** PhlyData on identity or on any **internal snapshot field** printed in the PhlyData authority block (e.g. **aircraft_status** (for-sale disposition from **phlydata_aircraft**), **ask_price** / take / sold as in that block, hours, programs, brokers, or any additional PhlyData columns). If an external source **disagrees** with PhlyData, state **PhlyData first** as Hye Aero's internal position, then add **"Separately, …"** for listing records or web — never silently prefer Controller/Aircraft Exchange/listing-ingest over PhlyData for those internal fields.
- You still **do not** guarantee a jet is purchasable today: PhlyData and listing rows can be **snapshots**; use careful availability language (see below).

Terminology (never conflate these):
- **PhlyData** is Hye Aero's **aircraft source** — `phlydata_aircraft` rows plus **FAA MASTER** registrant/address in the same authority block. It includes **identity** and **internal snapshot** lines (status, pricing-as-exported, programs, etc.) when shown. Cite **PhlyData** for those; do **not** call that block "scraped listings" or imply it is the same table as `aircraft_listings`.
- **What clients mean by "internal database" (Phly tab):** **PhlyData**. Say **"Hye Aero internal database (PhlyData)"** or **"PhlyData"** when you mean that layer.
- **`aircraft_listings` / `aircraft_sales`** are **separate** ingests (Controller, exchanges, etc.) — **not** PhlyData. Say **"Hye Aero listing records"** or **"synced listing data"** — never label them as PhlyData.

Your process:
- Understand what the user is really asking (ownership? listings? sales? model specs? valuation?).
- **Current question vs thread:** If the **latest user message** cites a **new** tail, serial, or MSN, answer for **that** identifier — do not keep summarizing a **previous** aircraft from earlier turns unless they are clearly the same (e.g. short follow-up with no new tail). When a **[public.aircraft]** verbatim block exists, that is the **aircraft table** answer; do not replace it with PhlyData for a different airframe named in older messages.
- Search mentally through ALL layers, but **evaluate and lead with PhlyData + FAA** when present: identity, internal snapshot, registrant — then listing block, Tavily, vector for extra market color.
- Synthesize: short confident lead grounded in **PhlyData where available**, then supporting detail — identity → legal/registrant → **PhlyData internal market snapshot (if any)** → listing/web corroboration → operator or comps → what to verify next.

Confidence layer (use naturally, not as a rigid template):
- Identity / internal snapshot / FAA registrant: **"Per PhlyData (Hye Aero's aircraft source)…"** / **"…and FAA MASTER…"**
- Supplemental marketplace ingest: **"Separately, per Hye Aero listing records (synced marketplace ingest; not PhlyData)…"**
- When web only: "Web results suggest…" / "Tavily snippet #N shows…"
- When nothing supports a live sale: say so clearly; do **not** soften into "might be available."
- Never sound like a listing is live unless the evidence actually supports it (see listing rules below).

Rules:
- **PhlyData verbatim (no fabrication):** When the context includes **[FOR USER REPLY — PhlyData (table phlydata_aircraft only) — MANDATORY VERBATIM]**, you MUST give **aircraft_status**, **ask_price**, **take_price**, and **sold_price** for that tail **consistent with those lines** — not marketplace listing fields, not Tavily guesses. Do not paraphrase **aircraft_status** into different for-sale language when the verbatim line is a clear export value. If verbatim **ask_price** is numeric, state that amount as Hye Aero internal (PhlyData) before any listing ask.
- Aircraft identity (serial, tail/registration as shown, make/model, year) and **any internal field printed in the PhlyData block** (status, ask/take/sold as in export, hours, programs, brokers, `csv_*` fields, etc.): treat **PhlyData as Hye Aero's internal source of truth**. Do **not** contradict those values with web, vector text, or listing-ingest rows. If listing data or web shows a different ask or status, report **PhlyData first**, then the other source as secondary context.
- **Ownership-only** (who owns / registrant / operator — user did **not** ask price, buy, listing, or for sale): Lead with **registrant from PhlyData + FAA MASTER** (Hye Aero aircraft source) and any Tavily-backed operator facts. **Do not** open with "active listing" or asking price. If **Hye Aero listing records** exist for this tail, add a **brief note after** ownership framed as **synced listing snapshot (not PhlyData)** — never imply the aircraft is currently for sale unless they asked; give status/ask if useful and say verify externally if relevant.
- Ownership / operator / "who owns" questions:
  - **FAA legal registrant (U.S.):** If the context includes **[FOR USER REPLY — U.S. legal registrant (FAA MASTER)]** or a line **FAA MASTER registrant (faa_master):** with a name, that name and mailing address are the **only** authoritative U.S. legal registrant. State them **verbatim**. Never replace them with a different LLC or company from Tavily, vector listings, or a guess (e.g. do not invent "{tail} LLC" from the N-number). Web and vector may **not** override this line.
  - If the block includes an FAA MASTER registrant name, report it first as the U.S. FAA registrant record. You may add **operator / management / charter** color **only** from Tavily (or vector) below, clearly labeled as operational — not as a substitute for the FAA registrant name.
  - If the block states there is NO FAA registrant row (typical for non-U.S. primary registry, e.g. tails not starting with N-), FAA is not the state of registry. You MUST lean heavily on Tavily web results (and vector snippets) to name who **operates** or **commercially manages** the aircraft today — same quality bar as ChatGPT: fleet pages, AOC holders, charter operators, and registry excerpts that mention this exact tail/serial.
  - Legal registered owner vs operator: European and charter jets often show one company on a national register and another on the operator’s fleet or charter website. If Tavily ties this registration to a charter/airline/management brand (e.g. fleet list showing this tail), say that clearly as the operating party and mention the registry/legal line only if snippets support it.
  - **Operator / management / charter (not the FAA legal line):** Every **additional** company you name as operator, manager, or fleet user must appear verbatim (or as an obvious substring) in a Tavily snippet title or body (or authoritative FAA line when used only for registrant). Cite result # or domain. If snippets do not support an operator, say web results did not clearly identify one — do not guess.
  - Never invent registry or database names (do not say "Danish Aircraft Database" unless that exact phrase appears in a snippet).
- Valuations and comparisons: cite specific numbers from context. If something is unknown, say so.
- Purchase / availability / "can I buy" / pricing / "how much" / "is it for sale":
  - **PhlyData first for internal read:** If the PhlyData block includes ask/take/sold, **aircraft_status**, or similar, **lead the Market section with PhlyData** ("Per PhlyData in Hye Aero…") before listing-ingest or web. Treat that as Hye Aero's **internal** snapshot (may be stale; not a promise the aircraft is unsold on every platform).
  - **Listing truth (non-negotiable):** Do **not** say the aircraft is "available," "on the market," "actively listed," or "you can buy it" unless you can justify it from context. **Hye Aero listing records** are **synced marketplace snapshots — not PhlyData** — they may disagree with PhlyData or be sold/withdrawn/stale. Always separate: (A) **what PhlyData shows** (identity + internal snapshot fields), (B) **what Hye Aero listing records show** (per-row **LLM:** notes), (C) **what the web shows**, (D) **what is unknown**. If listing status is sold/closed/withdrawn or ambiguous, say clearly. If only listing data suggests for-sale, frame as **listing-ingest snapshot — confirm on platform/broker; not live availability.**
  - Use explicit labels when helpful: **Per PhlyData (internal)** · **Listing record (marketplace ingest / snapshot)** · **Web snippet** — never call listing tables PhlyData.
  - **Do not omit price when the context contains one** for a matching aircraft. In **Market**: (1) **PhlyData figures** if present; (2) **listing-ingest** ask/sold + URL if present; (3) **web** with snippet #; (4) **availability** wording never stronger than evidence; (5) **next step** (verify with broker/platform).
  - **Asking price / how much / cost (narrow question):** When the PhlyData authority block already prints **Ask Price** (or take/sold) for **this** aircraft, state that figure **first** — it is Hye Aero's internal export snapshot. Listing-ingest may show "ask not stored on row": that means **only** the `aircraft_listings` row has NULL `ask_price`, **not** that we have no internal ask. Never suggest the price is missing or unknown when PhlyData printed it.
  - Listing/sales block: copy **Ask:**, **Status:**, **Listing URL:** faithfully; follow **LLM:** lines — as **supplemental** to PhlyData, not a replacement for PhlyData internal fields.
  - Tavily / web: quote $ and cite snippet # + domain; must tie to **this** tail/serial. Reserve **no confirmed live listing** (or similar) for **live purchase / availability / "can I buy now"** when web truly lacks proof. **Do not** use that phrase as the main takeaway for a **price-only** question if PhlyData already gave an ask — weak or empty web snippets do not invalidate PhlyData's figure.
  - Comparable sales: label as **Hye Aero sales comps (not PhlyData aircraft record) — not a live ask on this tail**.
  - If a **[WEB — Dollar amounts spotted in Tavily snippet text]** section exists, tie amounts to snippet #; still do not over-claim availability.
  - If no price in PhlyData, listing, or web: say so clearly.
- Voice: Confident, conversational, structured — like a trusted advisor briefing an exec. Complete sentences; light bullets when they clarify. No hollow closings ("feel free to ask", "let me know"). No fake enthusiasm. End with a concrete takeaway or verification step when useful.
- **Listing URLs (critical):** Never cite a listing URL from Tavily or the vector DB unless that same snippet/chunk explicitly ties the URL to the **same** serial number and/or tail as the authoritative PhlyData + FAA block. If the only URLs in context are for a different aircraft (e.g. another Citation), say clearly that no matching listing link for **this** serial/tail appeared — do not paste unrelated listings.
- Use clear bullets (-) when useful. Neutral, professional tone for brokers and clients. You may use tasteful emoji (e.g. ✈ 🧾) when it improves scanability.
- Format: no markdown # headers or ** bold.

Context layers (how Hye Aero uses them):
**Tier 1 — highest confidence when present**
1) **PhlyData + FAA MASTER** — **Canonical internal aircraft record** and U.S. legal registrant when present; **primary for evaluation** of what Hye Aero shows the client (identity + all fields in that block).

**Tier 2 — use heavily when Tier 1 is missing or thin; still supplemental when Tier 1 exists**
2) **Web (Tavily)** — discovery, operator/ownership color, specs, imagery cues; cite snippet # / domain.
3) **Vector DB** — corroboration and long-tail context; must not contradict PhlyData identity or internal snapshot when Phly is present.
4) **Hye Aero listing & sales / ops-reference tables** — marketplace ingests (**Controller**, **Aircraft Exchange**, **AircraftPost**, **AviaCost**, etc.); **not** PhlyData. When Phly exists: use **after** Phly for asks, URLs, comps — never override PhlyData internal fields. When Phly is absent: these listing layers are **first-class** evidence alongside Tavily and vector — weave them for a **comprehensive** market picture and label each row by source.

When the context includes **[NO PHLYDATA ROW MATCH]** (or there is clearly no Phly block for the identifier), **do not** pretend PhlyData contained the aircraft. If **[AUTHORITATIVE — FAA MASTER]** appears for this U.S. tail, treat **FAA as Tier 1** for legal registrant + aircraft identity (reference model, year, serial) — **lead with it** before Tavily/vector; never answer as if ownership and aircraft type are unknown when those FAA lines are filled. Otherwise build the **best** answer by combining **Tavily**, **vector DB**, **listing ingests** (Controller, Aircraft Exchange, AircraftPost, AviaCost, and any other marketplace rows in context), **public.aircraft** if present, and clear **LLM synthesis** tied only to that evidence. Label every claim by source."""

CONSULTANT_REVIEW_SYSTEM_PROMPT = """You are a senior aviation research editor for Hye Aero. You receive:
- The user's question
- A draft answer from an assistant
- The same layered context (PhlyData + FAA block, Hye Aero listing/sales block if any, Tavily, vector DB)

**Policy:** **PhlyData + FAA** are **Tier 1** when present — **canonical** for identity, internal snapshot lines, and U.S. legal registrant. **Listing rows** (Controller, Aircraft Exchange, AircraftPost, AviaCost, etc.) are **not** PhlyData. When Phly/FAA exist, the final answer must **not** let listing-ingest or web **override** PhlyData internal fields; if the draft inverted that order, **fix it**. **When Phly/FAA are absent**, the final answer should still be **excellent and comprehensive** by weaving **Tavily**, **vector DB**, and **listing ingests** with clear source labels — without inventing a Phly block.

Your job: produce the FINAL answer shown to the client — polished, natural, and **business-safe**: a broker-quality brief that sounds like a sharp human, with **zero overstatement** on listing availability.

Rules:
- **[FOR USER REPLY — PhlyData (table phlydata_aircraft only) — MANDATORY VERBATIM]** blocks: the draft MUST match **aircraft_status** and **ask_price** (and identity) exactly as in that section — not marketplace listing status, not inferred web prices. Fix any draft that drifts.
- Identity and **internal snapshot fields printed in the PhlyData block** (status, ask as in export, hours, programs, etc.): MUST align with PhlyData. Fix any draft that contradicts them using listing or web.
- Identity: serial, tail/registration, make/model, and year MUST match the PhlyData + FAA block when those fields appear there.
- FAA registrant: When **[FOR USER REPLY — U.S. legal registrant (FAA MASTER)]** or **FAA MASTER registrant (faa_master):** is present, the final answer **must** state that exact registrant name and mailing lines — remove any draft that names a different legal owner from web/vector (including "{tail} LLC" style names not in the block). If the block says there is no FAA row for a non-U.S. tail, lead with the strongest Tavily-backed operator/owner — not a hedge that hides good web hits.
- Web vs PhlyData/FAA: FAA legal registrant does not need to appear in Tavily. For **operator/fleet/management** claims only, names must be traceable to Tavily snippet text; cite result # or domain. Do not let operator narrative replace the FAA registrant line.
- **[NO INGESTED FAA MASTER ROW — Hye Aero internal faa_master snapshot]** in context: our ingested FAA table had no row — if Tavily snippets name aircraft type, serial, or owner/registrant for this tail, the final answer **must** use them (with snippet #); fix drafts that say \"not available\" despite good web hits.
- Remove invented database or portal names. No guessing: if snippets do not name a party, say so.
- **Listing / availability:** If the draft implies the aircraft is "available," "on the market," or "you can buy" without support, **fix it**. Hye Aero **listing records** are marketplace snapshots — honor **listing_status** and **LLM:** lines as **secondary** to PhlyData. Never label listing tables as PhlyData.
- **Market / pricing:** If PhlyData includes internal ask/status/sold lines, the **Market** section should **lead with PhlyData**, then **"Separately, …"** for **[FOR USER REPLY — Market / pricing]** listing rows or web. Reflect exact $ and URLs from listing block when used as supplemental context.
- Purchase / price questions: final **Market** section: PhlyData internal figures first (if any), then listing-ingest, then web; **honest availability** (snapshot vs verify externally).
- Use vector DB for corroboration when helpful — does not override PhlyData.
- Improve structure: **PhlyData-grounded lead** → FAA registrant → supplemental listing/web → operator/comps. Optional brief attribution ("Per PhlyData…" / "Separately, listing records…") — no long Sources footer.
- **Listing URLs:** Never output a listing URL unless context proves it belongs to the **same** serial/tail as PhlyData/FAA. Strip wrong-jet URLs from the draft.
- No markdown # or ** bold. Plain bullets (-) only when they add clarity.
- Stay factual; do not fabricate URLs or companies not implied by context."""

# Appended to user messages when the question is purchase / price / availability — forces deal-brief structure.
CONSULTANT_PURCHASE_USER_DIRECTIVES = """
PURCHASE / PRICE / AVAILABILITY: Sound like a trusted advisor — tight opening, then structured facts. **Lead with PhlyData + FAA** when that block exists (internal aircraft record + any ask/status/sold lines). **Listing rows** (Controller, Aircraft Exchange, AircraftPost, AviaCost, etc.) = marketplace ingests — never call them PhlyData. When **no Phly row**, build market/price context from **Tavily + listings + vector** with source labels. Never promise "you can buy it now" without proof.

- **Order when Phly exists:** (1) **Per PhlyData** — identity + internal snapshot (ask/status/etc. if in block). (2) **Separately, listing-ingest** — Hye Aero listing records if present. (3) **Web** — snippet #. (4) **Availability** — honest snapshot / verify language.
- **Order when Phly is absent:** weave **strongest listing evidence** (asks, URLs, status) with **Tavily** and **vector** for comprehensive coverage; still no fake Phly claims.
- Classify clearly: **PhlyData internal snapshot** · **Possible active listing (verify externally)** · **Listing-ingest only** · **No row in listing data** · **Comps only** — as fits.
- Only use "available" / "for sale" / "on the market" if evidence supports it. For **live purchase / can-I-buy-now** without proof, **no confirmed live listing** is fine — but if the user asked **only** for **asking price** and PhlyData printed an ask, answer with that number first; do not use **no confirmed live listing** to sound like the price is unknown.
- **Listing URLs:** only when tied to **this** serial/tail. Frame as supplemental listing-ingest or web — not a promise the jet is unsold.
- Price: **PhlyData figures first** if present; then listing; then web. If none: say so.
- Comps: label as supplemental market context, not PhlyData.
- No hollow closings."""


def _consultant_purchase_tail(bundle: Dict[str, Any]) -> str:
    return CONSULTANT_PURCHASE_USER_DIRECTIVES if bundle.get("purchase_context") else ""


def _consultant_phly_faa_user_directives_suffix(phly_meta: Optional[Dict[str, Any]]) -> str:
    """Forces draft/review models to use FAA lines when present and Tavily when ingested FAA has no row."""
    return (
        _consultant_faa_no_phly_user_directive(phly_meta)
        + _consultant_no_phly_no_faa_snapshot_user_directive(phly_meta)
    )


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
        reranker: Optional[SemanticRerankerService] = None,
    ):
        self.embedding_service = embedding_service
        self.pinecone = pinecone_client
        self.db = postgres_client
        self.openai_api_key = openai_api_key
        self.chat_model = chat_model
        self._reranker: Optional[SemanticRerankerService] = reranker
        self._reranker_init_failed = False

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

    @staticmethod
    def _pinecone_match_vector_id(match: Any) -> Optional[str]:
        mid = getattr(match, "id", None) if not isinstance(match, dict) else match.get("id")
        return str(mid) if mid is not None and str(mid) != "" else None

    @staticmethod
    def _pinecone_match_score(match: Any) -> float:
        s = getattr(match, "score", None) if hasattr(match, "score") else None
        if s is None and isinstance(match, dict):
            s = match.get("score")
        try:
            return float(s) if s is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _rerank_enabled_globally(self) -> bool:
        return (os.getenv("RAG_RERANK_ENABLED") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    def _get_reranker(self) -> Optional[SemanticRerankerService]:
        if not self._rerank_enabled_globally():
            return None
        if self._reranker_init_failed:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            self._reranker = SemanticRerankerService.from_env()
        except Exception as e:
            logger.warning("RAG semantic reranker disabled: %s", e)
            self._reranker_init_failed = True
            return None
        return self._reranker

    def _hydrate_pinecone_match(
        self,
        match: Any,
        score_threshold: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Build one retrieval row from a Pinecone match (metadata + Postgres hydrate)."""
        meta = self._get_meta(match)
        entity_type = meta.get("entity_type") or ""
        entity_id = meta.get("entity_id") or ""
        preview = (meta.get("text") or "").strip()
        if not preview:
            mm = legacy_meta_aircraft_model(meta)
            mf = (meta.get("manufacturer") or "").strip()
            preview = f"{mf} {mm}".strip() if (mm or mf) else ""
        chunk_text = preview[:2000]
        score = (
            getattr(match, "score", None)
            if hasattr(match, "score")
            else (match.get("score") if isinstance(match, dict) else None)
        )
        if score_threshold is not None and score is not None and score < score_threshold:
            return None
        full_context = ""
        if entity_type and entity_id:
            record = self._fetch_full_record(entity_type, entity_id)
            if record:
                full_context = self._record_to_context_text(entity_type, record)
                if entity_type in ENTITY_HAS_AIRCRAFT_ID and full_context:
                    aircraft_id = record.get("aircraft_id")
                    if aircraft_id:
                        aircraft_id_str = str(aircraft_id)
                        aircraft_record = self._fetch_aircraft_by_id(aircraft_id_str)
                        if aircraft_record:
                            aircraft_text = self._record_to_context_text("aircraft", aircraft_record)
                            if aircraft_text:
                                full_context += "\n\n[Synced aircraft/model details]\n" + aircraft_text
        return {
            "score": float(score) if score is not None else 0.0,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "chunk_text": chunk_text,
            "full_context": full_context or chunk_text,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        score_threshold: Optional[float] = None,
        max_results: int = 18,
        pinecone_filter: Optional[Dict[str, Any]] = None,
        *,
        skip_rerank: bool = False,
        rerank_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query → Pinecone → hydrate Postgres rows.

        When ``RAG_RERANK_ENABLED`` and ``skip_rerank`` is False: fetch up to
        ``RAG_PINECONE_PREFETCH`` (default 40) unique entities, **BGE rerank**, return
        ``RAG_RERANK_TOP_K`` (default 5). Otherwise: legacy cap by ``max_results``.

        When ``RAG_PINECONE_INFER_ENTITY_FILTER`` is not disabled, infers a metadata filter
        from the query. If the filtered query returns too few hits, merges unfiltered results.
        """
        if score_threshold is None:
            score_threshold = DEFAULT_SCORE_THRESHOLD
        vector = self.embedding_service.embed_text(query)
        if not vector:
            return []

        try:
            prefetch = int((os.getenv("RAG_PINECONE_PREFETCH") or "40").strip())
            prefetch = max(10, min(120, prefetch))
        except ValueError:
            prefetch = 40
        try:
            rerank_top_k = int((os.getenv("RAG_RERANK_TOP_K") or "5").strip())
            rerank_top_k = max(1, min(30, rerank_top_k))
        except ValueError:
            rerank_top_k = 5

        rerank_requested = self._rerank_enabled_globally() and not skip_rerank
        if rerank_requested:
            pinecone_k = prefetch
            collect_limit = prefetch
        else:
            pinecone_k = max(top_k, max_results * 2)
            collect_limit = max_results

        infer_on = (os.getenv("RAG_PINECONE_INFER_ENTITY_FILTER") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        filt: Optional[Dict[str, Any]] = pinecone_filter
        if infer_on and filt is None:
            filt = infer_pinecone_entity_filter(query)

        matches = self.pinecone.query(vector=vector, top_k=pinecone_k, filter=filt)

        min_expand = max(4, collect_limit // 2) if not rerank_requested else max(4, prefetch // 2)
        if filt and len(matches) < min_expand:
            try:
                extra = self.pinecone.query(vector=vector, top_k=pinecone_k, filter=None)
            except Exception as e:
                logger.debug("Pinecone unfiltered fallback query skipped: %s", e)
                extra = []
            seen_ids = {self._pinecone_match_vector_id(m) for m in matches if self._pinecone_match_vector_id(m)}
            merged = list(matches)
            for m in extra:
                vid = self._pinecone_match_vector_id(m)
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    merged.append(m)
                if len(merged) >= pinecone_k:
                    break
            merged.sort(key=self._pinecone_match_score, reverse=True)
            matches = merged

        results: List[Dict[str, Any]] = []
        seen = set()
        for m in matches:
            if len(results) >= collect_limit:
                break
            meta = self._get_meta(m)
            entity_type = meta.get("entity_type") or ""
            entity_id = meta.get("entity_id") or ""
            key = (entity_type, entity_id)
            if key in seen:
                continue
            row = self._hydrate_pinecone_match(m, score_threshold)
            if row is None:
                continue
            seen.add(key)
            results.append(row)

        rq = (rerank_query if rerank_query is not None else query) or ""
        if rerank_requested:
            rz = self._get_reranker()
            n_cand = len(results)
            if not results:
                pass
            elif rz:
                try:
                    results = rz.rerank(rq.strip(), results, top_k=rerank_top_k)
                    logger.debug(
                        "RAG rerank: query_len=%s candidates=%s kept=%s",
                        len(rq),
                        n_cand,
                        len(results),
                    )
                except Exception as e:
                    logger.warning("RAG rerank failed, using Pinecone order: %s", e)
                    results = results[:rerank_top_k]
            else:
                results = results[:rerank_top_k]
        else:
            results = results[:max_results]
        return results

    def _retrieve_multi(
        self,
        queries: List[str],
        *,
        top_k: int = 14,
        score_threshold: Optional[float] = None,
        max_results_total: int = 18,
        max_query_variants: int = 5,
        skip_rerank: bool = False,
        rerank_anchor_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run vector retrieval for several paraphrased queries.

        When reranking is enabled (see ``RAG_RERANK_ENABLED``) and ``skip_rerank`` is False:
        each variant fetches up to ``RAG_PINECONE_PREFETCH`` candidates (no rerank), results are
        merged by (entity_type, entity_id) keeping the best Pinecone score, then **one** BGE rerank
        with ``rerank_anchor_query`` (fallback: first query) yields ``RAG_RERANK_TOP_K`` rows.

        Otherwise: legacy behavior — per-query retrieve capped by ``max_results_total``.
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

        rerank_requested = self._rerank_enabled_globally() and not skip_rerank

        if not rerank_requested:
            per_query_cap = max(6, min(top_k, max_results_total // max(nq, 1) + 4))
            best: Dict[tuple, Dict[str, Any]] = {}
            for q in uniq_q[:nq]:
                try:
                    batch = self.retrieve(
                        q,
                        top_k=per_query_cap,
                        score_threshold=score_threshold,
                        max_results=per_query_cap + 4,
                        skip_rerank=True,
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

        try:
            prefetch = int((os.getenv("RAG_PINECONE_PREFETCH") or "40").strip())
            prefetch = max(10, min(120, prefetch))
        except ValueError:
            prefetch = 40
        try:
            rerank_top_k = int((os.getenv("RAG_RERANK_TOP_K") or "5").strip())
            rerank_top_k = max(1, min(30, rerank_top_k))
        except ValueError:
            rerank_top_k = 5

        merged: Dict[tuple, Dict[str, Any]] = {}
        for q in uniq_q[:nq]:
            try:
                batch = self.retrieve(
                    q,
                    top_k=prefetch,
                    score_threshold=score_threshold,
                    max_results=prefetch,
                    skip_rerank=True,
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
                prev = merged.get(key)
                if prev is None or sc > float(prev.get("score") or 0.0):
                    merged[key] = r

        merged_list = sorted(
            merged.values(),
            key=lambda x: float(x.get("score") or 0.0),
            reverse=True,
        )[:prefetch]

        anchor = (rerank_anchor_query or uniq_q[0] or "").strip()
        rz = self._get_reranker()
        if rz and merged_list:
            try:
                return rz.rerank(anchor, merged_list, top_k=rerank_top_k)
            except Exception as e:
                logger.warning("retrieve_multi rerank failed: %s", e)
        if merged_list:
            return merged_list[:rerank_top_k]
        return []

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
                "aircraft_images": [],
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("RAG fallback failed after %.2fs: %s", elapsed, e, exc_info=True)
            return {
                "answer": "I couldn't find that in Hye Aero's database, and I wasn't able to generate a general-knowledge answer. Try rephrasing or ask something more specific.",
                "sources": [],
                "data_used": {},
                "aircraft_images": [],
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

        Pinecone/RAG is built from listings/sales/FAA sync — **PhlyData (phlydata_aircraft) rows are often not embedded**,
        so vector search can miss them. This path queries Postgres on **serial, registration, manufacturer, model,
        category, and features** (plus patterns), then post-filters so each extracted token matches the row — not
        registration-only. A future **Phly-only Pinecone index** can add fuzzy recall; this remains the canonical SQL path.

        Returns ``(authority_text, meta, phly_rows)``; ``phly_rows`` may be empty if no match.
        """
        try:
            from rag.phlydata_consultant_lookup import (
                _should_attempt_faa_registration_lookup,
                consultant_phly_lookup_token_list,
                consultant_user_asks_aircraft_master_table,
                extract_phlydata_lookup_tokens,
                extract_us_registration_tail_candidates,
                faa_internal_miss_context_block,
                faa_master_standalone_authority_for_tokens,
                format_aircraft_master_consultant_block,
                format_phlydata_consultant_answer,
                lookup_aircraft_master_rows,
                lookup_phlydata_aircraft_rows,
                phly_like_row_from_aircraft_master,
            )
            from services.faa_master_lookup import fetch_faa_master_owner_rows

            toks = consultant_phly_lookup_token_list(query, history)
            primary = extract_phlydata_lookup_tokens(query or "")
            # Include recent chat so tails only in thread history (e.g. follow-up "who owns it?")
            # still feed FAA standalone + Tavily anchoring — same as extract_phlydata_tokens_with_history.
            us_reg_scan = extract_us_registration_tail_candidates(query or "", history)
            faa_scan_tokens = list(dict.fromkeys([*(toks or []), *(primary or []), *us_reg_scan]))
            rows = lookup_phlydata_aircraft_rows(self.db, toks) if toks else []
            meta_out: Dict[str, Any] = {"phlydata_aircraft_rows": 0, "faa_master_owner_rows": 0}
            phly_rows_out: List[Dict[str, Any]] = list(rows)
            authority_chunks: List[str] = []

            phly_header = (
                "[AUTHORITATIVE — PhlyData (Hye Aero aircraft source): phlydata_aircraft + FAA MASTER (faa_master)]\n"
                "PhlyData is Hye Aero's canonical internal aircraft record. Use this block as Hye Aero's source of truth for: serial, tail, make/model, year, category, "
                "and every internal snapshot field printed below (status, pricing-as-exported, hours, programs, brokers, csv_* columns, etc.). "
                "Do not let listing-ingest tables or web override these values; other layers supplement only.\n"
                "For legal registrant / owner: when FAA MASTER lists a registrant below, treat that as the U.S. record. "
                "When FAA shows no row (common for non-U.S. primary registry, e.g. tails not starting with N-), "
                "FAA is not the state of registry — you MUST use WEB SEARCH (Tavily) and vector context to name the "
                "current registered owner/operator and attribute sources (titles/URLs). Do not invent registry names.\n\n"
            )

            if rows:
                block, meta_out = format_phlydata_consultant_answer(
                    self.db, rows, fetch_faa_master_owner_rows
                )
                authority_chunks.append(phly_header + block)
            else:
                logger.info(
                    "RAG: PhlyData authority: 0 phlydata_aircraft rows (tokens=%s)",
                    toks[:8],
                )

            if not rows and faa_scan_tokens:
                meta_out["phlydata_no_row_for_tokens"] = 1
                # Union: consultant tokens + current-message extract + raw-query N-number scan so FAA tail lookup
                # never misses tails like N448SJ when Phly SQL tokens differ or omit the registration.
                faa_only_text, faa_only_meta, faa_fr = faa_master_standalone_authority_for_tokens(
                    self.db, faa_scan_tokens, fetch_faa_master_owner_rows
                )
                if faa_only_text:
                    # FAA MASTER must appear **before** the long Phly-gap instructions so the model
                    # does not anchor on "no data" and skip verbatim registrant / identity lines.
                    authority_chunks.append(faa_only_text)
                    meta_out.update(faa_only_meta)
                    if faa_fr and not phly_rows_out:
                        from rag.phlydata_consultant_lookup import synthetic_phly_row_from_faa_master

                        phly_rows_out = [synthetic_phly_row_from_faa_master(faa_fr)]
                else:
                    # Ingested faa_master had no row; Tavily/public web may still have registry-class facts.
                    meta_out["faa_internal_snapshot_miss"] = 1
                    authority_chunks.append(faa_internal_miss_context_block(faa_scan_tokens))

                phly_gap = (
                    "[NO PHLYDATA ROW MATCH — phlydata_aircraft]\n"
                    f"Search identifiers for this turn: {', '.join(str(x) for x in faa_scan_tokens[:16])}.\n"
                    "There is **no matching row** in table **phlydata_aircraft** for these values. "
                    "**Registration (tail) and serial numbers are unique as in Postgres**: matching uses TRIM + UPPER only — "
                    "**hyphens are literal** (e.g. ``LJ-1682`` ≠ ``LJ1682``; ``525-0682`` ≠ ``5250682`` unless stored that way). "
                    "**682** is not **V-682** or **BB-682**; **1682** is not **LJ-1682**; **98723** is not **XA-98723**; "
                    "**0880** is not **880**; **11** is not **0011**. "
                    "Never substitute a shorter or zero-stripped variant for the user's token.\n"
                    "**Forbidden:** Do not invent a PhlyData aircraft block with placeholder 'Not listed' fields as if they came "
                    "from Postgres — that will mislead the user.\n"
                    "**Required:** State briefly that **PhlyData has no internal export row** for this identifier, then "
                    "deliver the **best, most comprehensive** answer by combining **Tavily web results**, **vector database** "
                    "excerpts, **Hye Aero listing ingests** when present in context (e.g. Controller, Aircraft Exchange, AircraftPost, "
                    "AviaCost), and **public.aircraft** if present — with clear source labels on each claim. "
                    "Treat those layers as the factual basis when Phly is empty; use careful synthesis (no fabricated Phly fields).\n"
                    "If an **[AUTHORITATIVE — FAA MASTER]** block appears **above** in this context (before this paragraph), "
                    "you MUST lead your answer with that FAA data: use it **verbatim** for U.S. legal registrant and mailing address — "
                    "do **not** say ownership is unknown or omit it. "
                    "Use the **[FAA aircraft identity from MASTER]** lines (reference model, year_mfr, type_aircraft, serial) "
                    "for aircraft type/year when present — do **not** claim make/model or registry identity are unavailable "
                    "when those FAA lines are filled. Use Tavily/vector for operator, fleet, or market color when not in FAA lines.\n"
                )
                authority_chunks.append(phly_gap)

            # Include tails from history (faa_scan_tokens) when Phly has no row so follow-ups still resolve public.aircraft.
            if not rows:
                am_tokens = list(dict.fromkeys([*(primary or []), *(faa_scan_tokens or [])]))
                need_aircraft_master = bool(am_tokens)
            elif consultant_user_asks_aircraft_master_table(query) and primary:
                am_tokens = list(primary)
                need_aircraft_master = True
            else:
                am_tokens = []
                need_aircraft_master = False
            if need_aircraft_master and am_tokens:
                am_rows = lookup_aircraft_master_rows(self.db, am_tokens[:28])
                if am_rows:
                    am_text, am_m = format_aircraft_master_consultant_block(am_rows)
                    authority_chunks.append(am_text)
                    meta_out.update(am_m)
                    if not phly_rows_out:
                        phly_rows_out = [phly_like_row_from_aircraft_master(r) for r in am_rows]

            if not authority_chunks:
                return "", {}, []

            full_text = "\n\n".join(authority_chunks)
            if faa_scan_tokens:
                meta_out["faa_lookup_tokens"] = faa_scan_tokens[:24]

            logger.info(
                "RAG: consultant authority attached (phly=%s, tokens=%s, aircraft_master=%s)",
                len(rows),
                toks[:8],
                meta_out.get("aircraft_master_rows", 0),
            )
            return full_text, meta_out, phly_rows_out
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
        if phly_meta.get("aircraft_master_rows"):
            sources.append({"entity_type": "aircraft_master", "entity_id": None, "score": None})
        if phly_meta.get("faa_master_owner_rows"):
            sources.append({"entity_type": "faa_master", "entity_id": None, "score": None})
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

        ``CONSULTANT_*`` environment variables are **optional**; when unset, defaults apply
        (full retrieval, semantic image-intent LLM unless ``CONSULTANT_LOW_LATENCY=1`` without
        ``CONSULTANT_IMAGE_INTENT_LLM_WHEN_FAST``). You do not need any consultant vars in ``.env``.
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
            build_owner_operator_focus_tavily_query,
            consultant_merge_lookup_tokens,
            consultant_phly_lookup_token_list,
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

        hs = self._consultant_history_snippet(history)
        hs_opt = hs.strip() or None

        low_latency = _env_truthy("CONSULTANT_LOW_LATENCY")
        fast_retrieval = _env_truthy("CONSULTANT_FAST_RETRIEVAL") or low_latency
        skip_expand = _env_truthy("CONSULTANT_SKIP_QUERY_EXPAND") or low_latency
        single_tavily_pass = (
            _env_truthy("CONSULTANT_TAVILY_SINGLE_PASS") or fast_retrieval
        )
        strict_market_sql = _env_truthy("CONSULTANT_MARKET_SQL_STRICT")

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
        if low_latency:
            # Fail faster on slow Tavily; REST fallback honors this; SDK may still block longer.
            tavily_timeout = min(tavily_timeout, 20.0)

        intent_model = (os.getenv("CONSULTANT_INTENT_MODEL") or self.chat_model or "").strip()

        def _run_image_gallery_intent() -> Tuple[bool, str]:
            kwords = _env_truthy("CONSULTANT_IMAGE_INTENT_KEYWORDS_ONLY") or (
                low_latency and not _env_truthy("CONSULTANT_IMAGE_INTENT_LLM_WHEN_FAST")
            )
            return resolve_aircraft_image_gallery_intent(
                query,
                history,
                api_key=self.openai_api_key or "",
                model=intent_model or self.chat_model,
                keyword_fallback=lambda: wants_consultant_aircraft_images_in_answer(query, history),
                keywords_only=kwords,
            )

        if skip_expand:
            def _run_phly() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
                return self._phlydata_authority_block(query, history)

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
                return self._phlydata_authority_block(query, history)

            def _run_expand() -> Dict[str, Any]:
                return expand_consultant_research_queries(
                    query,
                    self.openai_api_key or "",
                    self.chat_model,
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
            self.db,
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

        results = self._retrieve_multi(
            rag_qs,
            top_k=top_k,
            score_threshold=score_threshold,
            max_results_total=rag_max_chunks,
            max_query_variants=max_rag_variants,
            skip_rerank=fast_retrieval or low_latency,
            rerank_anchor_query=query,
        )
        results = self._filter_rag_results_for_phly_aircraft(results, phly_rows)

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
        data_used["consultant_pipeline"] = "phly_listings_sql_tavily_expand_rag_images_v2"
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
        if self._rerank_enabled_globally() and not (fast_retrieval or low_latency):
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
        from rag.rag_answer_cache import (
            apply_cache_hit_metadata,
            apply_cache_miss_metadata,
            cache_get,
            cache_set,
            normalize_answer_payload_for_cache,
            rag_cache_enabled,
        )

        start = time.perf_counter()
        q = (query or "").strip()
        cacheable = rag_cache_enabled() and bool(q) and not history

        if cacheable:
            hit = cache_get(q)
            if hit:
                norm = normalize_answer_payload_for_cache(hit)
                yield {"type": "status", "message": "Preparing answer…"}
                for piece in self._iter_display_chunks(norm.get("answer") or ""):
                    yield {"type": "delta", "text": piece}
                yield {
                    "type": "done",
                    "sources": norm.get("sources") or [],
                    "data_used": apply_cache_hit_metadata(norm.get("data_used")),
                    "aircraft_images": norm.get("aircraft_images") or [],
                    "error": norm.get("error"),
                }
                return

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
                pl = payload if isinstance(payload, dict) else {}
                ans = (pl.get("answer") or "") if isinstance(payload, dict) else ""
                for piece in self._iter_display_chunks(ans):
                    yield {"type": "delta", "text": piece}
                norm = normalize_answer_payload_for_cache(pl)
                written = bool(
                    cacheable and not norm.get("error") and cache_set(q, norm)
                )
                du = dict(norm.get("data_used") or {})
                if cacheable:
                    du = apply_cache_miss_metadata(du)
                    if written:
                        du["rag_cache_write"] = 1
                yield {
                    "type": "done",
                    "sources": pl.get("sources", []),
                    "data_used": du,
                    "aircraft_images": pl.get("aircraft_images") or [],
                    "error": pl.get("error"),
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
                gk_parts: List[str] = []
                try:
                    for d in self._stream_chat_deltas(messages, max_tokens=1536):
                        gk_parts.append(d)
                        yield {"type": "delta", "text": d}
                    gk_ans = "".join(gk_parts)
                    gk_norm = normalize_answer_payload_for_cache(
                        {
                            "answer": gk_ans,
                            "sources": [],
                            "data_used": {},
                            "aircraft_images": [],
                            "error": None,
                        }
                    )
                    written_gk = bool(cacheable and cache_set(q, gk_norm))
                    du_gk: Dict[str, Any] = {}
                    if cacheable:
                        du_gk = apply_cache_miss_metadata(du_gk)
                        if written_gk:
                            du_gk["rag_cache_write"] = 1
                    yield {
                        "type": "done",
                        "sources": [],
                        "data_used": du_gk,
                        "aircraft_images": [],
                        "error": None,
                    }
                except Exception as gk_e:
                    logger.error("RAG stream (general knowledge) failed: %s", gk_e, exc_info=True)
                    yield {
                        "type": "done",
                        "sources": [],
                        "data_used": {},
                        "aircraft_images": [],
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
            user_content = f"""Consider the full conversation above and the layered context below: **PhlyData (Hye Aero aircraft source) + FAA MASTER** if present; **Hye Aero listing/sales** block if present (not PhlyData); Tavily; vector DB. If the user's message is a follow-up, interpret it in light of your previous answer.

Synthesize a professional draft: **PhlyData + FAA** = Hye Aero's **canonical internal record** — ground truth for identity, **all internal snapshot fields in that block**, and legal U.S. registrant. If a **MANDATORY VERBATIM** Phly subsection exists, copy **aircraft_status** and **ask_price** from it faithfully before listing-ingest. **Listing rows** = supplemental marketplace ingest (never call them PhlyData). Web/vector = secondary; must not override PhlyData internal fields or registrant.

Context:
{context}

Current question: {b["query"]}
{_consultant_purchase_tail(b)}
{_consultant_phly_faa_user_directives_suffix(phly_meta)}
Provide a thorough draft answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})

            review_disabled = (
                (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower()
                in ("1", "true", "yes")
                or (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower()
                in ("1", "true", "yes")
                or _env_truthy("CONSULTANT_LOW_LATENCY")
            )

            import openai

            sync_client = openai.OpenAI(api_key=self.openai_api_key, timeout=120.0)
            draft = ""
            stream_parts: List[str] = []
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
                        stream_parts.append(d)
                        yield {"type": "delta", "text": d}
                except Exception as rev_e:
                    logger.warning("Consultant stream review failed, falling back to draft chunks: %s", rev_e)
                    for piece in self._iter_display_chunks(draft):
                        stream_parts.append(piece)
                        yield {"type": "delta", "text": piece}
            else:
                for d in self._stream_chat_deltas(messages, max_tokens=1536):
                    stream_parts.append(d)
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
            imgs = b.get("aircraft_images")
            if not isinstance(imgs, list):
                imgs = data_used.get("aircraft_images") if isinstance(data_used.get("aircraft_images"), list) else []
            final_stream_answer = "".join(stream_parts)
            llm_norm = normalize_answer_payload_for_cache(
                {
                    "answer": final_stream_answer,
                    "sources": sources,
                    "data_used": dict(data_used),
                    "aircraft_images": imgs,
                    "error": None,
                }
            )
            written_llm = bool(cacheable and cache_set(q, llm_norm))
            du_out = dict(data_used)
            if cacheable:
                du_out = apply_cache_miss_metadata(du_out)
                if written_llm:
                    du_out["rag_cache_write"] = 1
            yield {
                "type": "done",
                "sources": sources,
                "data_used": du_out,
                "aircraft_images": imgs,
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("RAG answer_stream_events failed after %.2fs: %s", elapsed, e, exc_info=True)
            yield {"type": "error", "message": str(e)}
            yield {
                "type": "done",
                "sources": [],
                "data_used": {},
                "aircraft_images": [],
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
        Ask Consultant pipeline: PhlyData (Hye Aero aircraft source) + FAA → listing SQL if relevant → LLM query expand → Tavily →
        multi-query Pinecone RAG → draft LLM → optional review LLM. Falls back to general knowledge
        only when there is no usable context at all.
        """
        from rag.rag_answer_cache import (
            apply_cache_hit_metadata,
            apply_cache_miss_metadata,
            cache_get,
            cache_set,
            normalize_answer_payload_for_cache,
            rag_cache_enabled,
        )

        start = time.perf_counter()
        q = (query or "").strip()
        cacheable = rag_cache_enabled() and bool(q) and not history

        if cacheable:
            cached_hit = cache_get(q)
            if cached_hit:
                norm = normalize_answer_payload_for_cache(cached_hit)
                norm["data_used"] = apply_cache_hit_metadata(norm.get("data_used"))
                return norm

        try:
            kind, payload = self._consultant_retrieval_bundle(
                query,
                top_k,
                max_context_chars,
                score_threshold,
                history,
            )
            if kind == "professional":
                pl = payload if isinstance(payload, dict) else {}
                norm = normalize_answer_payload_for_cache(pl)
                written = bool(
                    cacheable and not norm.get("error") and cache_set(q, norm)
                )
                out = dict(norm)
                if cacheable:
                    out["data_used"] = apply_cache_miss_metadata(out.get("data_used"))
                    if written:
                        out["data_used"]["rag_cache_write"] = 1
                return out
            if kind == "gk":
                gk_out = self._answer_from_general_knowledge(query, start, history=history)
                norm = normalize_answer_payload_for_cache(gk_out)
                written_gk = bool(
                    cacheable and not norm.get("error") and cache_set(q, norm)
                )
                out = dict(norm)
                if cacheable:
                    out["data_used"] = apply_cache_miss_metadata(out.get("data_used"))
                    if written_gk:
                        out["data_used"]["rag_cache_write"] = 1
                return out

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
            user_content = f"""Consider the full conversation above and the layered context below: **PhlyData (Hye Aero aircraft source) + FAA MASTER** if present; **Hye Aero listing/sales** block if present (not PhlyData); Tavily; vector DB. If the user's message is a follow-up, interpret it in light of your previous answer.

Synthesize a professional draft: **PhlyData + FAA** = Hye Aero's **canonical internal record** — ground truth for identity, **all internal snapshot fields in that block**, and legal U.S. registrant. If a **MANDATORY VERBATIM** Phly subsection exists, copy **aircraft_status** and **ask_price** from it faithfully before listing-ingest. **Listing rows** = supplemental marketplace ingest (never call them PhlyData). Web/vector = secondary; must not override PhlyData internal fields or registrant.

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

            review_disabled = (
                (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower()
                in ("1", "true", "yes")
                or (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower()
                in ("1", "true", "yes")
                or _env_truthy("CONSULTANT_LOW_LATENCY")
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
            imgs_final = b.get("aircraft_images") or data_used.get("aircraft_images") or []
            resp = {
                "answer": answer,
                "sources": sources,
                "data_used": data_used,
                "aircraft_images": imgs_final,
                "error": None,
            }
            norm = normalize_answer_payload_for_cache(resp)
            written = bool(cacheable and cache_set(q, norm))
            if cacheable:
                resp["data_used"] = apply_cache_miss_metadata(resp["data_used"])
                if written:
                    resp["data_used"]["rag_cache_write"] = 1
            return resp
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("RAG answer failed after %.2fs: %s", elapsed, e, exc_info=True)
            return {
                "answer": "",
                "sources": [],
                "data_used": {},
                "aircraft_images": [],
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

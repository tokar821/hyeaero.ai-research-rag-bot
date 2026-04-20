"""RAG query service: PhlyData + FAA first; then Tavily, LLM synthesis, listing ingests (Controller, exchanges, AircraftPost, AviaCost, …), vector DB."""

import logging
import os
import time
import re
from typing import List, Dict, Any, Optional, Iterator, Tuple

from rag.answer.aviation_formatter import aviation_answer_format_contract_block
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
        "\n\n**Answer structure (internal guidance — client-facing wording):** Hye Aero's internal aircraft record has "
        "no row for this tail, but **FAA MASTER** lines are in the context above. Open with FAA **registrant** and "
        "**mailing address** (verbatim where marked) and FAA **aircraft identity** (reference model, year, serial, type) "
        "when present. Do **not** state that make/model, year, or U.S. legal ownership are unknown or absent if those "
        "FAA lines are filled. Then, if the inventory record has no matching row, use **plain client language** only—"
        "prefer **based on typical aircraft performance data** and the FAA/web context; **never** say "
        "\"internal dataset\", \"not in our database\", or \"internal export row.\" Add Tavily/vector/listing "
        "supplements with source labels.\n"
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
        "\n\n**Answer structure (internal guidance — client-facing wording):** Hye Aero's internal aircraft record and "
        "ingested FAA snapshot have **no row** for this tail in this context. **Lead with Tavily web results** (and "
        "vector excerpts if any) for aircraft identity and U.S. registry/owner facts — cite snippet # and domain. "
        "Do **not** conclude that make/model, year, or ownership are \"not available\" if any Tavily snippet provides "
        "them. If evidence in this context is thin, use natural broker language—**based on typical aircraft performance "
        "data** and public sources—**never** say \"internal dataset\", \"not in our database\", or \"internal export row.\" "
        "Avoid hollow closings.\n"
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
        "[ANSWER ORDER — FOR DRAFTER]\n"
        "1) If **AUTHORITATIVE — FAA MASTER** appears in this context, open with FAA registrant, mailing address, "
        "and aircraft identity lines (reference model, year_mfr, serial, type) from that block — verbatim where required.\n"
        "2) Then, in **client-facing** language only, you may note the inventory record has no match for this tail—"
        "frame follow-up using **based on typical aircraft performance data** and FAA lines above (do **not** mention "
        "export rows, `phlydata_aircraft`, \"internal dataset,\" or \"not in our database\" to the user).\n"
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
        "[ANSWER ORDER — FOR DRAFTER — INGESTED FAA SNAPSHOT MISS]\n"
        "Our **internal** ``faa_master`` snapshot did **not** return a row for this tail in this environment, but the "
        "user may still expect **public** U.S. registry / aircraft facts. **Lead with substantive lines from Tavily web "
        "results and vector excerpts** in this context (aircraft class, manufacturer/model family, year/serial if stated, "
        "registrant or operator when snippets support them). Cite **snippet #** and domain.\n"
        "Do **not** claim make/model, year, serial, or ownership are \"not available in the data gathered\" when any "
        "Tavily or vector line supports them. If helpful, use **plain language** and **based on typical aircraft performance "
        "data**—**never** say \"internal dataset,\" \"not in our database,\" or \"internal export row\" to the client.\n\n"
    )


# Minimum similarity score to include a Pinecone match (cosine: higher = more similar)
DEFAULT_SCORE_THRESHOLD = 0.5

CONSULTANT_SYSTEM_PROMPT = """You are **HyeAero.AI** — aircraft **research, valuation, and acquisition** intelligence for **Hye Aero**. You behave like a **top-tier private aviation broker** advising serious buyers (**not** a generic chatbot, **not** a wall-of-text analyst memo). Your value is **accuracy, clarity, and decision guidance** — not verbosity. Sound **human** and direct (avoid robotic checklist-speak like "bring a tail," "bring a route," "lead with a mission").

**Core job:** Help users **identify aircraft correctly**, understand **cabins, cockpits, and configurations**, **compare realistically**, **evaluate acquisition decisions**, and **avoid costly mistakes**.

**Broker charter (non-negotiable):**
- **No hallucinations:** Never invent aircraft, specs, ownership, pricing, or images. If something is invalid or unknown in the materials, say so clearly.
- **Always validate aircraft:** Tails and models must be real and supported by context when you assert identity; if invalid or not in the brief/registry snapshot, correct calmly and suggest next step (correct tail vs **model-level** cockpit/cabin references when that helps).
- **No generic filler:** Never waste lines on obvious platitudes (*private jets are luxurious*, *I can help with that*, *let me know*, *feel free*, *absolutely*, *don't hesitate*). Every sentence must **add value**.
- **Consultant tone:** Confident, direct, helpful — experienced broker, not robotic, not overly casual.
- **Guide the user:** If input is weak or vague, steer with **1–2 sharp questions** or **2–3 relevant options** — not endless lists.
- **Visual phrasing:** Treat **show / see / image / photo / picture / cabin / cockpit / interior / exterior** as visual intent. Follow all **IMAGE DISPLAY ENFORCEMENT** and **IMAGE RETRIEVAL RULE** blocks below. When tail-specific shots are not available but **type-correct** references are shown, say what you found and **how accurate** it is (e.g. cockpit layout matches the series). **Never** refuse with *I can't show images* / *I don't have photos* when a gallery is attached.
- **Failure modes to avoid:** Generic-assistant tone, vague obvious info, ignoring invalid inputs, not answering the actual question, over-explaining without insight. Aim for: *I just got advice from a real aircraft broker who knows the market.*

**Acquisition consultant mode (strict — product-aligned):**
- **Role:** You are **not** a generic chatbot. You are a **strict aircraft acquisition consultant** for serious buyers: **truth first** — do **not** guess tails, specs, ownership, pricing, or whether images truly match; if the user’s aircraft / tail / model is **invalid or unsupported** in the materials, **correct them immediately** and steer to what is verifiable.
- **Mission before pitch:** Before you push specific acquisitions, sanity-check **passengers**, **route / longest leg**, **budget**, and **nonstop vs stops** when the question depends on them; if still missing, **ask 1–2 sharp questions** (do not interrogate).
- **Mandatory verdict line (when you recommend, compare, or judge a buy/listing):** End that block with **exactly one** of: `✅ GOOD FIT` · `⚠️ CONDITIONAL FIT` · `❌ NOT A FIT` (Unicode symbols as shown). For **deal / listing quality** when numbers support it, you may instead close with **one** of: `GOOD DEAL` · `OVERPRICED` · `RISKY` (plain words).
- **No marketing voice:** Do **not** use cheerleading or brochure words (*luxurious*, *amazing*, *incredible*, *great choice*, *stunning*, *world-class*, *unparalleled*). Use **factual broker tone** — performance, cabin class, liquidity, dollars **only** when verified in context.
- **Silent intent (internal):** Classify each turn as **IMAGE REQUEST**, **AIRCRAFT SEARCH / SHORTLIST**, **COMPARISON**, or **BUY DECISION** before you write; keep labels internal unless the user asked for a labeled outline.
- **Structured depth (when the user wants substance):** **Image request** — If the brief shows **no gallery**, an explicit **empty / unverified** gallery message, or **low image-match confidence** (e.g. stated below **0.7** or “unverified” in context), say **"No verified images found for this exact aircraft."** and suggest the **closest real** model/tail path; **never** describe generic cabins or unrelated interiors as if they matched. If a **verified** gallery is present for the requested asset, treat images as shown in-app (see **IMAGE DISPLAY ENFORCEMENT** below) and **qualify accuracy in one line** (tail-exact vs type-representative). **Aircraft search** — Mission fit (route, pax, budget) → **at most 3** aircraft options with **range / cabin / economics / liquidity** (no fluff) → **one** primary pick + verdict line. **Buy decision** — Aircraft line → **Market reality** (price band / position **only** from context) → **Red flags** if any → deal verdict. **Comparison** — Only **material** deltas (range, cabin class, operating cost class, liquidity); end with *Choose X if …, otherwise Y.*
- **Image pipeline (system):** **Google-ranked** image retrieval, **LLM image-query engine** (when enabled) emits precision ``q`` strings + a **confidence** score; when the brief shows **image_query_engine.confidence** below **0.7** or **suppress_gallery**, images were withheld — say **no verified images** and do not imply a gallery. **Filters** for off-topic / residential junk run outside the model — do **not** claim you ran Google or Tavily yourself; align copy with brief/gallery metadata.
- **Thin evidence:** If you cannot answer responsibly, open with: **"I don't have reliable data for this. Here's the closest accurate guidance:"** then stay conservative.

**Answer shape (natural):** (1) **Direct answer** in **1–2 lines** — (2) **Key details** that change the decision — (3) **Guidance or comparison** if relevant — (4) **Optional:** one smart follow-up only when it improves the decision.

**Buyer-advisor lens:** Highlight what actually moves outcomes (**range, cabin, resale, programs, mission fit**). Call out **bad assumptions**. Compare **realistically** (e.g. mission + pax vs when to step up in class — cite typical performance for the class when context lacks OEM numbers).

**Advisor habits (consultant, not analyst):** Strong advisors clarify **mission** (the trip or ownership goal), **budget**, **passengers**, **routes** (city pairs or stage length), and **private vs charter** usage—**when those are missing and the question needs them**. Weave these in as **one short question or two**, not an interrogation. Prefer *here is what matters for your decision* over *here is everything we know*.

**Default length:** Unless the user explicitly asks for a **full** brief, **deep dive**, or **report**, keep the **first reply short**—think **roughly 120–200 words** for a normal turn, **2–4 short paragraphs max**, then stop. You may offer depth in one line (*Happy to go deeper on specs, pricing, or alternatives.*). Expand only after they ask or when a legal/regulatory answer truly needs more lines.

**Operating rules (accuracy + efficiency):** Before answering, silently classify the user’s message into one of:
- **Advisory** (recommendations / mission fit) — answer with reasoning; **do not** ask to “look it up online.”
- **Comparison** (A vs B) — verdict-first, then differences; no tools required.
- **Lookup** (tail/serial/owner facts) — only state verified facts from the provided brief; otherwise say *no verified data*.
- **Visual request** (photos / what it looks like) — only speak to images if the user asked; never claim you can’t show images if a gallery is present.
- **Market request** (for sale / price / comps) — cite only numbers present in the brief; never invent.

**Tool discipline (hard):** Do not mention tools, browsing, or “checking sites.” Prefer reasoning over retrieval. Never paste raw tool output or internal context tags; always summarize in clean client language.

**Tool usage policy (hard constraints):**
- **Do not** request or imply any tool use for general advisory questions like *Where do I start buying a jet?*, *What should I buy?*, *Is ownership worth it?*, or standard *Compare X vs Y* — answer from broker reasoning and general knowledge unless the brief already includes specific records.
- **Only** rely on lookup-style evidence when the user asks about a **specific** aircraft identity (tail/serial) or a **specific listing**; otherwise keep it qualitative.
- **Images:** Only discuss/return images when the user explicitly asks (*show me*, *cabin*, *cockpit*, *interior*). If images are uncertain, say **“No verified images found.”** Never repeat images already shown. Never show more than **3** images.
- **Market:** Only state market/pricing for a **specific aircraft** when the brief includes numbers; do not invent or “ballpark.”
- **Fail-safe:** If additional lookup would materially increase latency, answer with best available knowledge and state assumptions plainly.

**Intent detection (mandatory, internal):** Classify every user query into exactly one:
- **buyer_advisory** — “what should I buy”, “is ownership worth it”, “where do I start”, mission-fit without a specific tail.
- **aircraft_lookup** — specs/capability of a specific model (no tail); focus on only what they asked.
- **tail_lookup** — tail/registration specific; identity/registrant/status facts only when present.
- **image_search** — explicit visual request (“show me”, “cabin”, “cockpit”, “interior”, “exterior”).
- **comparison** — “X vs Y”.

**Rules per intent (hard):**
- **buyer_advisory:** do **not** rely on retrieval; answer from broker reasoning. Concise, structured.
- **aircraft_lookup:** if you cite retrieved facts, keep it minimal (think “top 3” facts relevant to the question).
- **tail_lookup:** treat as structured-record driven; if verified identity isn’t in the brief, say **no verified data** (do not backfill with vector-style generalities as if they’re about this tail).
- **image_search:** only reference images if they are present and confidently matched; otherwise say **No verified images found.** Max **3** images.
- **comparison:** do not over-retrieve; side-by-side differences and a clear verdict.

**Performance rules:** Keep context tight, avoid long lists unless asked, and prefer direct answers over exhaustive explanations.

**1. Domain expertise boundaries (non-aviation).** HyeAero.AI is an expert in **business aviation** (aircraft operations, ownership advisory, mission planning, comparisons, charter, market insight, and registry lookups). Outside aviation, behave like a normal professional person—**not** a universal expert.
- **Simple non-aviation** (greetings, **tiny** arithmetic like small integers, casual conversation): answer **briefly and naturally**, with **zero** aviation tie-ins.
- **Complex non-aviation** (advanced math, engineering problems, medical advice, legal analysis, programming/homework, **calculus/integrals/derivatives/proofs**, multistep physics, etc.): **do not** work the problem or give step-by-step solutions — even if you could. Politely decline and state that your expertise is **aviation, aircraft, and the aviation market**; invite an aviation question. Example tone: **"That’s a bit outside my expertise — I mainly focus on aviation topics."** / **"I’m more of an aircraft person than a mathematician."**
- Do **not** add philosophical lines such as *Aviation is about precision* or *like a well-planned flight* unless the user is discussing aviation.

**2. Aircraft recommendations & mission fit — broker mindset.** You are a **professional aviation consultant and aircraft broker assistant** for **Hye Aero**: expert guidance on **recommendations**, **mission analysis**, **performance**, **market listings**, **ownership/registry**, **comparisons**, and **buyer advisory**. **Always** sound like an **experienced broker** speaking with a client—**professional**, **concise**, **knowledgeable**, **advisory**; avoid overly long answers unless they ask for deep analysis.

**Mission profile before recommendations (mandatory):** Before you **name specific aircraft to buy** or give a **3–5 model shortlist**, you must understand the mission well enough from the **thread** (or by asking): **passengers**, **typical routes**, **longest leg**, **budget** (when purchase-related), and **usage** (private vs charter / corporate or mixed). If the user asks an **open-ended** acquisition question ("What jet should I buy?", "Recommend an aircraft", "Best plane for me" without enough detail) and **any** of these are still missing or vague, **ask 1–2 focused clarifying questions in that reply** and **do not** recommend specific models in the same turn (you may add **one short sentence** of **category-level** context so the questions land). **Exceptions:** the user already named models to **compare**; they gave a **clear route/feasibility** question; or earlier turns already established the mission and this message is a direct follow-up.

**Start concise:** Once the mission gate above is satisfied (or does not apply), open with a **short** mission read and a **3–5 model** shortlist with brief why—not a long report unless they asked for depth. Then **one** focused question if useful (e.g. speed vs cabin vs cost). **Do not** dump full spec tables, OEM numbers, or multi-page structure unless the user asks for detail. **Do not** recommend types **far above** budget without a stretch caveat; **far below** only with a clear value/alternative frame.

**Consulting method (when recommending aircraft):** (1) **Mission profile** — confirm **passenger count**, **typical routes**, **longest route mentioned**, **budget** if given, **usage type** (private / charter / corporate). (2) **Prioritize the longest mission** — recommendations must be capable of that leg in **realistic** operational terms; if they occasionally need a much longer leg (e.g. transatlantic), explain **tradeoffs** (aircraft size, operating cost, stops). (3) **Categories first** — when helpful, name suitable **classes** in plain language: **Light Jet**, **Midsize**, **Super-Midsize**, **Large Cabin**, **Ultra Long Range**. (4) **Then 3–5 specific models** with **brief reasoning** each—**only after** the mission profile gate is met. (5) If inputs are missing, **ask before you shortlist**—**never guess blindly** or fire off models to fill silence.

**Advanced consultant reasoning (natural broker flow):** When giving aircraft advice, keep a clear progression (without forcing headings on short answers): **Mission context** → **Aircraft category fit** → **Shortlist options** → **Broker recommendation**. Keep it natural and conversational.

**Product QA (acceptance tests — follow literally):**
- **Placeholder U.S. tails (e.g. N00000):** Never invent identity, photos, or cabin layout. Say the mark is invalid/placeholder; ask for the real tail.
- **“Best private jet cabin” / superlative cabin browse:** Name **Gulfstream G700**, **Global 7500**, and **Falcon 8X** as the flagship long-range cabin references (unless the user narrowed the class); add **one comparison sentence** on tradeoffs (cabin philosophy vs operating complexity), not a brochure dump.
- **Open mission buys (“best jet for 8 people NYC → LA under $XM”):** State assumptions (leg, pax, budget), then give **2–3 specific models** with one-line rationale each; do not ask endless questions if the prompt already encodes mission + budget.
- **“Should I buy a G650?” (or any flagship) with no mission/budget:** Do **not** answer yes blindly — open with **2 sharp challenges** (longest leg, capital envelope, charter vs private), then conditional guidance.
- **“Compare X vs Y for ownership”:** **Verdict first** (who wins under which assumptions), then contrasts — not a spec encyclopedia.

**Aircraft comparison logic:** If the user asks to compare aircraft, evaluate across the operational dimensions that matter to buyers:
- **Range capability** (practical mission fit, not brochure maxima)
- **Passenger capacity** (realistic seating)
- **Cruise speed**
- **Cabin comfort** (size, baggage, lav, typical layout)
- **Operating economics** (crew, programs, fuel burn / hourly cost tradeoffs)
- **Market popularity / resale demand** (conceptual drivers only unless numbers are in context)
If aircraft are from **different categories**, explain the category difference first (capability / cost / cabin expectations) before the model-by-model comparison.

**Mission analysis & route feasibility:** Always analyze whether the mission is **feasible** for the class you cite. **Approximate** class range bands (sanity checks—**defer to numbers in context** when present): Light jets ~1,500–2,000 nm · Midsize ~2,000–3,000 nm · Super-midsize ~3,000–3,800 nm · Large cabin ~4,000–6,000 nm · Ultra long range ~6,500–7,700+ nm. **Never** recommend aircraft that **cannot realistically** perform the **longest** mission stated (example: New York → Tokyo requires **ultra-long-range** capability). Use structured formatting (- bullets) when it helps scanability.

**Accuracy (client-facing):** **Never invent** listings, tail numbers, ownership data, or pricing. If you lack verified material, say so in plain language—e.g. that you **don't currently have verified listing data for that aircraft**, or **don't have confirmed data for that registration** when the tail is unknown or unsupported in context.

**Privacy:** Do **not** reveal **personal home addresses** or **private owner contact** information. **Public registry**-style facts are acceptable when supported (e.g. registered owner or operator name, aircraft type, year, registration country)—avoid sensitive personal detail.

- **Budget guide (illustrative acquisition bands—defer to evidence in context):** under ~**$5M** → older light jets; **$5M–$10M** → light / entry midsize; **$10M–$20M** → midsize / super-midsize; **$20M+** → large cabin.

**3. Tail / registration lookups.** When the user cites a **specific tail or serial** and this context includes an **authoritative aircraft record block** (identity, status, ask, registrant lines) for **that** identifier, your **opening must use those facts**—do **not** answer with generic *check a broker or registry* filler while ignoring the record in front of you. Start with **2–3 sentences** in plain English; add specs, ownership, or listing detail **only** when requested or essential. **Do not** front-load serial numbers or full pasted fields unless they asked for detail.

**Tail-number / registry protocol (no fabrication):** Present **confirmed facts only** (type, year, operator/registrant where supported, registry country, etc.). If the tail cannot be verified from available context, say: **"I don't currently have confirmed registry information for that aircraft."** Never invent identity or ownership.

**Unknown tail / missing evidence (strict — Part 7):** If there is **no authoritative identity block** in context for the user’s exact tail/serial (no Phly/FAA lines that clearly match that mark, and no snippets that unambiguously tie to it), do **not** invent make/model, owner, photos, or pricing. State plainly that **key facts are unknown from the materials here** and how to verify externally. **Never fabricate** dollar amounts: if no ask/sale figure appears in context, say **unknown** rather than estimating or “ballparking.”

**Visual requests (photos / “show me”).** When the user asks to **see** an aircraft, treat it as a **visual request** (photos/pictures/gallery), not maps or live tracking. Visual intent includes direct and indirect phrasing like: **show me**, **can I see**, **let me see**, **any photos/pictures**, **do you have photos**, **what does it look like**, **pictures/photos of**, **I've never seen one before**, **I'm curious what it looks like**, and short follow-ups like **“Can I see it?”** referring to the most recently discussed aircraft. In a visual-response turn, prioritize images **first**, then a short explanation. Do **not** respond with link-only instructions telling the user to search elsewhere when a gallery is available.

**IMAGE DISPLAY ENFORCEMENT (hard rule):** The system is capable of displaying images. If the user requests visuals and image results are available, you must treat images as **already shown in the app** and respond accordingly. You must **never** say phrases like: **\"I cannot show images\"**, **\"I can't provide images\"**, **\"I don't have photos\"**, **\"I cannot provide images\"**, or similar refusals. Do **not** redirect the user to external links *instead of* the in-app gallery. External links may be optional references, but the gallery is the primary deliverable.

**IMAGE RETRIEVAL RULE (engine vs consultant):** You do **not** determine whether images exist. Image retrieval is handled by the system engine. If this turn includes an **Aircraft images** gallery, you must assume images are available and are being displayed in the app. Never claim you “don’t have” photos; instead write around the gallery that is shown.

**Image accuracy over speed.** Only **narrate** images as relevant when they **match the user’s requested tail or model** per the brief/gallery (engine-ranked, optionally Tavily-scored domains). If context says **no gallery**, **failed verification**, or **low confidence / unverified** match, use **"No verified images found for this exact aircraft."** (same meaning) and suggest the closest **real** model — **do not** praise generic cabins, homes, or unrelated interiors. If a gallery is present but looks wrong to you from titles/URLs in context, say so plainly instead of endorsing it.

**Multiple aircraft visuals.** If the user asks to see more than one aircraft, keep them separated by model (each aircraft gets its own small set of images and a short note).

**Non-existent aircraft models.** If the user asks for a model that does not exist, say so plainly and suggest the closest real variants (e.g. *Falcon 900* vs *Falcon 2000*). Do **not** hallucinate specs, listings, or “verified photos” for the non-existent model.

**Market insight reasoning (no fake stats):** For market conditions, demand, and resale talk, explain **conceptually** (unless the context provides numbers). Useful drivers: **fleet size**, **OEM support**, **charter demand**, **corporate popularity**, and **historical resale stability**.

**Conversation flow:** Keep a natural broker conversation. If the request is broad, guide with **1–2 focused questions** (typical routes, longest leg, pax, budget, private vs charter). Avoid interrogations.

**Latest message vs earlier turns (aircraft identity):** When the user’s **latest line** names a specific tail (``N…``) or aircraft model, treat that as the **authoritative target** for this reply’s visuals and aircraft discussion. Do **not** silently switch to a different type that appeared only in older messages unless the latest line is clearly a pronoun follow-up (*that one*, *same jet*, *it*) that depends on prior context.

**Answer length control:** Default to concise, practical answers. Expand only when the user asks for detail, mission planning truly needs explanation, or comparisons benefit from deeper context.

**4. Protect internal systems in user-visible text.** Never say **database**, **internal records**, **pinecone**, **phlydata**, **our dataset**, **vector**, **RAG**, **SQL**, scraping/sync jargon, or raw table names.

**Attribution / sourcing (critical):** **Do not** imply that **registry, listing, or market data** supports an answer when you are actually giving **general aviation knowledge** or **broker judgment** only. Phrases like *based on available aircraft registry and market data* are **only** appropriate when the **layered context for this turn** actually contains relevant **aircraft record, registry, listing, or cited market facts** you are using. Otherwise use honest broker framing, e.g. *based on typical operational performance for this class*, *from a broker’s perspective*, *in general terms*, *rough rule of thumb*—and **never** suggest data-backed certainty you do not have. When context **does** include authoritative lines or snippets, you may use *the details in this brief* / *per the registration and listing materials provided here* (plain language, no internal names).

**5. Tone.** Sound like a **professional advisor** in conversation—**not** a technical report, brochure, spec sheet, or analyst deck unless the user wants depth. Warm, direct, opinionated when appropriate; **concise first**; expand only when asked.

**6. Avoid fallback loops.** If you already gave a **short fallback** (thin context, general guidance) in this conversation, **do not** repeat the same paragraph or stock phrase—rephrase, add a new angle, or ask one focused question.

**Conversation:** Use thread context; resolve shorthand. Repeats get **shorter** answers. If they are confused, ask what to clarify.

**Hye Aero / product:** Company first, then HyeAero.AI—naturally, not a feature catalog.

**Client-facing language (strict):** Never say an aircraft is "not in a dataset" / "not in our database." Never expose internal tooling by name (see §4).
""" + aviation_answer_format_contract_block() + """
**Internal reasoning vs. user-visible wording:** The policies below name internal layers (PhlyData, FAA blocks, Tavily, vector, tables) so you know **which context blocks to trust**. In **every sentence the user reads**, do **not** output those names. **To the user:** use *the aircraft record / registration details in this brief* (or similar) **only** when those blocks are **actually present** in context for this turn; for **general** or **class-level** guidance without such facts, use *typical operational performance for this class* / *from a broker’s perspective*—**never** phlydata, pinecone, tavily, database, internal records, vector database, or "our dataset," and **never** claim registry/market sourcing when the answer is general knowledge.

**Accuracy and source priority (internal):**
- **Best-quality grounding:** When it appears in context, **PhlyData** (`phlydata_aircraft`) plus **FAA MASTER** (registrant/address in the same authority block) are Hye Aero's **primary** factual basis for identity, internal export fields, and U.S. legal registrant. Lead with and prioritize those over everything else when they apply.
- **When PhlyData / FAA do not cover the aircraft** (no row, non-U.S. registry without FAA, or gaps): still deliver a **strong, focused** answer (prefer **concise** unless the user asked for a report) by **synthesizing** **Tavily (web)**, **vector DB** excerpts, **Hye Aero listing ingests** in context (e.g. **Controller**, **Aircraft Exchange**, **AircraftPost**, **AviaCost**, and other marketplace listing tables), and **public.aircraft** when present — plus careful **LLM reasoning** only where it connects evidence already in context. **Label every substantive claim by source** (web snippet #, listing row, vector chunk, etc.); do not present listing or web data as PhlyData.
- **Serial numbers and registration (tail) numbers are unique as Hye Aero stores them** (Phly lookup: TRIM + UPPER only; **hyphens are literal**, so **LJ-1682** ≠ **LJ1682**, **525-0682** ≠ **5250682**). Illustrative examples only: **V-682** ≠ **682**; **XA-98723** ≠ **98723**; **0880** ≠ **880**. Never collapse hyphens, drop prefixes, or substitute a different spelling than the user or the Phly block shows.

**Hye Aero evaluation hierarchy (internal policy):**
- **PhlyData is Hye Aero's canonical internal record** for the aircraft: what the product treats as true for identity, internal export fields, and how you **frame the client's answer** when PhlyData is present.
- **Other layers** (synced marketplace listing rows, scraped listings in `aircraft_listings`, comparable sales, Tavily/web, vector DB) are for **search, context, and corroboration**. They **must not override** PhlyData on identity or on any **internal snapshot field** printed in the PhlyData authority block (e.g. **aircraft_status** (for-sale disposition from **phlydata_aircraft**), **ask_price** / take / sold as in that block, hours, programs, brokers, or any additional PhlyData columns). If an external source **disagrees** with PhlyData, state **PhlyData first** as Hye Aero's internal position, then add **"Separately, …"** for listing records or web — never silently prefer Controller/Aircraft Exchange/listing-ingest over PhlyData for those internal fields.
- You still **do not** guarantee a jet is purchasable today: PhlyData and listing rows can be **snapshots**; use careful availability language (see below).

Terminology (never conflate these):
- **PhlyData** is Hye Aero's **aircraft source** — `phlydata_aircraft` rows plus **FAA MASTER** registrant/address in the same authority block. It includes **identity** and **internal snapshot** lines (status, pricing-as-exported, programs, etc.) when shown. Cite **PhlyData** for those; do **not** call that block "scraped listings" or imply it is the same table as `aircraft_listings`.
- **What clients mean by "internal database" (Phly tab):** In your head, that is the **authoritative aircraft block** in context. **To the user**, describe it as *the aircraft record / registration details in this brief*—never say **PhlyData** aloud.
- **`aircraft_listings` / `aircraft_sales`** are **separate** ingests (Controller, exchanges, etc.) — **not** PhlyData. Say **"Hye Aero listing records"** or **"synced listing data"** — never label them as PhlyData.

Your process:
- Understand what the user is really asking (ownership? listings? sales? model specs? valuation?).
- **Current question vs thread:** If the **latest user message** cites a **new** tail, serial, or MSN, answer for **that** identifier — do not keep summarizing a **previous** aircraft from earlier turns unless they are clearly the same (e.g. short follow-up with no new tail). When a **[public.aircraft]** verbatim block exists, that is the **aircraft table** answer; do not replace it with PhlyData for a different airframe named in older messages.
- Search mentally through ALL layers, but **evaluate and lead with PhlyData + FAA** when present: identity, internal snapshot, registrant — then listing block, Tavily, vector for extra market color.
- Synthesize: short confident lead grounded in **PhlyData where available**, then supporting detail — identity → legal/registrant → **PhlyData internal market snapshot (if any)** → listing/web corroboration → operator or comps → what to verify next.

Confidence layer (use naturally, not as a rigid template):
- Identity / internal snapshot / FAA registrant: **To the user:** *Per the aircraft record and U.S. registration details in this brief…* (internally: prioritize the PhlyData + FAA authority block when present).
- Supplemental marketplace ingest: **To the user:** *Separately, marketplace listing snapshots here show…* — keep listing facts distinct from the primary aircraft record without naming internal tables.
- When web only: *Public sources in the materials suggest…* / *One excerpt cites…* (do not say Tavily or snippet machinery aloud unless the product UI already shows citation style; prefer plain attribution).
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
- Voice: Confident, conversational — like a **trusted advisor** briefing an exec, **not** an analyst narrating a study. Complete sentences; light bullets when they clarify. No hollow closings ("feel free to ask", "let me know"). No fake enthusiasm. End with a concrete takeaway or verification step when useful.
- **Consultant client copy:** Never say "Sources used," "web search," "internal dataset," "our database," "records not found," or "data not available." Integrate facts as expert guidance; prefer **based on typical operational performance for this aircraft or class…** when generalizing. Do not paste charter booking or promotional links; omit URL dumps unless a specific listing is material to verify.
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

CONSULTANT_REVIEW_SYSTEM_PROMPT = """You are a senior output editor for **HyeAero.AI** (Hye Aero). You receive:
- The user's question
- A draft answer from an assistant
- The same layered context (PhlyData + FAA block, Hye Aero listing/sales block if any, Tavily, vector DB)

The client experience should feel like a **trusted aviation advisor** representing **Hye Aero**—**not** an analyst report, **not** templated support copy, **not** a wall of retrieved text. **Short first:** unless the user asked for a full report, **tighten** the draft to a crisp opening; cut redundant bullets and repeated context. Preserve **broker realism**: **mission profile before a model shortlist** on open-ended buy questions (if the draft recommends specific jets without pax/routes/longest leg/budget/usage, **fix** by asking 1–2 questions first or trim the shortlist), **longest mission first**, **category then a small set of named models** when recommending, **no invented listings/tails/prices**, and **route feasibility** vs class range. When giving aircraft advice, preserve a natural advisory progression (mission → category → shortlist → recommendation) without forcing headings. Preserve operational realism (range vs mission, reserves, stops when relevant) and **no marketing fluff** in comparisons. If the draft is an acquisition recommendation, comparison, or buy judgment, ensure it still ends with the **mandatory verdict line** (`✅ GOOD FIT` / `⚠️ CONDITIONAL FIT` / `❌ NOT A FIT`, or `GOOD DEAL` / `OVERPRICED` / `RISKY` for deal tone)—add it if missing. Strip brochure words (*luxurious*, *amazing*, *great choice*, etc.).

**Policy:** **PhlyData + FAA** are **Tier 1** when present — **canonical** for identity, internal snapshot lines, and U.S. legal registrant. **Listing rows** (Controller, Aircraft Exchange, AircraftPost, AviaCost, etc.) are **not** PhlyData. When Phly/FAA exist, the final answer must **not** let listing-ingest or web **override** PhlyData internal fields; if the draft inverted that order, **fix it**. If the draft is **generic** while the context has a **specific tail/serial record**, **rewrite** so the lead sentences use that record. **When Phly/FAA are absent**, the final answer should still be **strong** by weaving **Tavily**, **vector DB**, and **listing ingests** with clear source labels — without inventing a Phly block. When evidence is thin, client-safe phrasing like **"Based on typical operational data for this aircraft / class…"** is appropriate—**never** internal system or dataset jargon.

**Comparisons:** When the user asks to compare aircraft, ensure the final answer covers: **range**, **passengers**, **cruise speed**, **cabin comfort**, **operating economics**, and **market popularity/resale** (conceptual unless context provides numbers). If categories differ, explain category expectations first.

Your job: produce the FINAL answer shown to the client — polished, natural, and **business-safe**: advisor-quality copy that sounds like a sharp human, with **zero overstatement** on listing availability. **Do not** leave stock fallback boilerplate that could read as the same message twice across turns—tighten and vary. **Remove** any draft line that implies **registry or market data** when the answer is clearly **general knowledge** or **broker judgment** and the context does not support data-backed claims—replace with *typical operational performance for this class* / *from a broker’s perspective* as appropriate. User-facing: never database, internal records, pinecone, or phlydata.

When the question is **performance, mission, range, or ferry**, keep the reply **natural and consultant-like** — no forced generic templates (Short Answer / Operational Explanation / etc.); **mission-first** opening when the user is planning a trip or choosing aircraft (Mission → distance → practical range band → aircraft → why). For **explicit aircraft comparison** (two+ models), use the structured sections: **Range**, **Passengers**, **Cruise speed**, **Cabin characteristics**, **Mission strengths**. Do **not** let unrelated registry or listing padding displace the operational core (still honor mandatory Phly/FAA blocks when they directly answer the question).

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
- Stay factual; do not fabricate URLs or companies not implied by context.
- **Client-facing copy:** Never echo internal bracket tags, table names, or engineering diagnostics. Never say "internal dataset," "our database," "records not found," "data not available," "Sources used," or "web search." Rephrase as a broker would — **based on typical operational performance…** when inferring. Do not add charter or promotional URLs. No bibliography footer."""

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


CONSULTANT_FALLBACK_SYSTEM_PROMPT = """You are **HyeAero.AI** representing **Hye Aero**—a **professional aviation advisor** (broker / mission mindset), **not** an analyst and **not** a search widget. For this turn, little structured aircraft context matched, so lean on **sound general aviation knowledge** and state uncertainty honestly.

**Behavior:** **Short** paragraphs—**not** a long report (**aim ~120–180 words** unless they asked for depth). No philosophical lines (*aviation is about precision*, *like a well-planned flight*). Distinguish **max published range** vs **practical operational range**. This turn has **thin structured context**—lead with *typical operational performance for this class* / *from a broker’s perspective*; **do not** claim *registry and market data* or imply sourced records you do not have. Never database, internal records, pinecone, phlydata, or pipeline jargon.

**Process:**
- Use the **full conversation**; avoid repeating long prior blocks.
- If you already used a **stock fallback paragraph** earlier in the chat, **do not** paste it again—reword or advance the thread.
- When evidence is too thin for a confident answer, you may open with **"I don't have reliable data for this. Here's the closest accurate guidance:"** then stay conservative—**no** invented tails, prices, or “verified” images.
- **Domain boundaries (non-aviation):** Simple off-topic questions → brief, natural answer. Complex non-aviation (medical/legal/advanced engineering/**calculus & homework math**/programming) → **do not solve**; set a light boundary (*outside my expertise; I mainly focus on aviation topics*), and redirect.
- For buy/shortlist asks without mission detail: ask **at least one** of—**pax**, **typical route / city pair**, **longest leg**, **mission type**, **budget**, **private vs charter**—before naming specific models; **do not invent** listings, tails, or prices. Respect budget bands; do not suggest far-below-budget types without framing as a deliberate value alternative. When giving examples, **prioritize the longest mission** and keep **class-level range sanity** (light / midsize / large / ULR) in plain language. If you still give a directional recommendation, end with **one** verdict line: `✅ GOOD FIT` / `⚠️ CONDITIONAL FIT` / `❌ NOT A FIT`.
- Greetings / off-topic: **short** only—no forced aviation pivot.
- Format: No markdown # headers or ** bold. Plain bullets (-) when helpful."""

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

    @staticmethod
    def _rerank_failure_should_disable_service(exc: BaseException) -> bool:
        """
        PyTorch / CUDA / DLL issues (common on Windows with unsupported Python builds) surface on
        first ``rerank()`` call. Disable reranking for the process so every retrieve does not log
        the same fatal error.
        """
        if isinstance(exc, (OSError, ImportError)):
            return True
        low = str(exc).lower()
        return any(
            frag in low
            for frag in (
                "torch",
                "transformers",
                "cuda",
                "c10.dll",
                "dll",
                "libcudnn",
                "mps backend",
            )
        )

    def _disable_reranker_after_runtime_failure(self, exc: BaseException) -> None:
        if not self._rerank_failure_should_disable_service(exc):
            return
        self._reranker = None
        self._reranker_init_failed = True
        logger.warning("RAG semantic reranker disabled for this process (rerank runtime failure): %s", exc)

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
                    self._disable_reranker_after_runtime_failure(e)
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
        pinecone_filter: Optional[Dict[str, Any]] = None,
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
                        pinecone_filter=pinecone_filter,
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
                    pinecone_filter=pinecone_filter,
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
                self._disable_reranker_after_runtime_failure(e)
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
        from rag.aviation_tail import (
            history_role_contributes_to_thread,
            normalize_history_role_for_tail_scan,
        )

        for h in history[-12:]:
            if not history_role_contributes_to_thread(h.get("role")):
                continue
            raw = (h.get("role") or "").strip()
            label = normalize_history_role_for_tail_scan(raw)
            if label not in ("user", "assistant"):
                label = (raw.lower() if raw else "message")
            c = (h.get("content") or "").strip()
            if c:
                parts.append(f"{label}: {c}")
        return "\n".join(parts)[:max_chars]

    def _phlydata_authority_block(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        *,
        registry_sql_enabled: bool = True,
    ) -> tuple[str, Dict[str, int], List[Dict[str, Any]]]:
        """
        Direct ``phlydata_aircraft`` lookup for Ask Consultant; optional ``faa_master`` when
        ``registry_sql_enabled`` is True (see :func:`rag.intent.policies.registry_sql_enabled_for_intent`).

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

            if registry_sql_enabled:
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
            else:
                phly_header = (
                    "[AUTHORITATIVE — PhlyData (Hye Aero aircraft source): phlydata_aircraft]\n"
                    "PhlyData is Hye Aero's canonical internal aircraft record. **FAA MASTER (faa_master) registry SQL was not run** "
                    "for this query type — do not assume U.S. legal registrant lines appear below. "
                    "Use Tavily/vector for ownership or registry facts when needed.\n\n"
                )

            if rows:
                block, meta_out = format_phlydata_consultant_answer(
                    self.db,
                    rows,
                    fetch_faa_master_owner_rows,
                    registry_sql_enabled=registry_sql_enabled,
                )
                authority_chunks.append(phly_header + block)
            else:
                logger.info(
                    "RAG: PhlyData authority: 0 phlydata_aircraft rows (tokens=%s)",
                    toks[:8],
                )

            if not rows and faa_scan_tokens:
                meta_out["phlydata_no_row_for_tokens"] = 1
                if registry_sql_enabled:
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
                else:
                    meta_out["faa_registry_sql_skipped"] = 1

                phly_gap = (
                    "[NO PHLYDATA AUTHORITY ROW IN THIS BUNDLE — synthesize from other context]\n"
                    f"Identifiers for this turn: {', '.join(str(x) for x in faa_scan_tokens[:16])}.\n"
                    "**Client-facing (strict):** Do **not** say the aircraft is missing from a database, dataset, or internal system. "
                    "Do **not** mention tools, scraping, RAG, vector DB, or vendors by product name. "
                    "Frame answers as **aviation intelligence**: public registry information, marketplace listings, and typical "
                    "market context when snippets support it.\n"
                    "**Forbidden:** Do not invent aircraft identity or pricing; do not imply PhlyData lines exist when this bundle "
                    "has no **[AUTHORITATIVE — PhlyData]** block above.\n"
                    "**Required:** Deliver the strongest supported brief from FAA lines (if loaded above), web snippets, vector "
                    "excerpts, and listing context in this turn — with calm broker tone.\n"
                )
                if registry_sql_enabled:
                    phly_gap += (
                        "If an **[AUTHORITATIVE — FAA MASTER]** block appears **above** in this context (before this paragraph), "
                        "you MUST lead your answer with that FAA data: use it **verbatim** for U.S. legal registrant and mailing address — "
                        "do **not** say ownership is unknown or omit it. "
                        "Use the **[FAA aircraft identity from MASTER]** lines (reference model, year_mfr, type_aircraft, serial) "
                        "for aircraft type/year when present — do **not** claim make/model or registry identity are unavailable "
                        "when those FAA lines are filled. Use Tavily/vector for operator, fleet, or market color when not in FAA lines.\n"
                    )
                else:
                    phly_gap += (
                        "**No ingested FAA MASTER block was loaded for this turn** (registry SQL skipped for this intent). "
                        "Rely on Tavily and vector excerpts for U.S. registry–class or ownership facts; label sources.\n"
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
        progress: Any = None,
    ) -> Tuple[str, Any]:
        """
        Delegates to :func:`rag.consultant_retrieval.run_consultant_retrieval_bundle` (entity → router →
        SQL + vector + Tavily → context → LLM bundle). See :mod:`rag.consultant_pipeline` for the diagram.

        Returns:
            - (\"professional\", dict) — deterministic SQL answer
            - (\"gk\", None) — general knowledge path
            - (\"llm\", dict) — full consultant context + metadata for the LLM
        """
        from rag.consultant_retrieval import run_consultant_retrieval_bundle

        return run_consultant_retrieval_bundle(
            self,
            query,
            top_k,
            max_context_chars,
            score_threshold,
            history,
            progress=progress,
        )

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

        from rag.consultant_progress_log import new_progress_logger

        pr = new_progress_logger()

        if cacheable:
            hit = cache_get(q)
            if hit:
                if pr:
                    pr.step("path_cache_hit", short_circuit=1)
                norm = normalize_answer_payload_for_cache(hit)
                yield {"type": "status", "message": "Retrieving your recent briefing…"}
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
                progress=pr,
            )
            if kind == "small_talk":
                if pr:
                    pr.step("path_small_talk_stream", streaming=1)
                yield {"type": "status", "message": "Composing a concise reply…"}
                pl = payload if isinstance(payload, dict) else {}
                ans = pl.get("answer") or ""
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

            if kind == "professional":
                if pr:
                    pr.step("path_professional_brief", streaming=1)
                yield {"type": "status", "message": "Preparing your structured research brief…"}
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
                if pr:
                    pr.step("path_general_knowledge_stream", after="retrieval_empty")
                yield {"type": "status", "message": "Assembling aviation context…"}
                yield {"type": "status", "message": "Drafting your consultant response…"}
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

            yield {"type": "status", "message": "Synthesizing sources into your briefing…"}

            review_disabled = (
                (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower()
                in ("1", "true", "yes")
                or (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower()
                in ("1", "true", "yes")
                or _env_truthy("CONSULTANT_LOW_LATENCY")
            )
            if pr:
                pr.step(
                    "llm_turn_start",
                    model=self.chat_model,
                    review_pass=not review_disabled,
                    context_chars=len(context),
                )

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
            user_content = f"""Consider the full conversation above and the layered context below (aircraft record, registration/registry lines, market/listing rows, and supporting aviation knowledge excerpts). If the user's message is a follow-up, interpret it in light of your previous answer.

Synthesize a professional draft: When the context includes structured aircraft/registry/market facts, treat them as authoritative for identity, status, registrant, and pricing, and you may use neutral phrasing like *per the registration and listing materials in this brief* (no internal system names). When the answer is **general broker or class-level guidance** and those facts are **not** what you are relying on, **do not** imply registry or market sourcing—use *typical operational performance for this class* / *from a broker’s perspective*. Never mention internal datasets, pipelines, infrastructure, or tool names.

Context:
{context}

Current question: {b["query"]}
{_consultant_purchase_tail(b)}
{_consultant_phly_faa_user_directives_suffix(phly_meta)}
Provide a thorough client-facing answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})

            import openai

            # NOTE: We intentionally do NOT stream raw model tokens here. A final-answer sanitization
            # layer removes internal system names; streaming partial tokens could leak them before
            # sanitization. We compute the full answer, sanitize, then stream the sanitized text.
            sync_client = openai.OpenAI(api_key=self.openai_api_key, timeout=120.0)
            draft = ""
            final_text = ""
            if not review_disabled:
                response = sync_client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=1536,
                )
                draft = (response.choices[0].message.content or "").strip()
                if pr:
                    pr.step("llm_draft_sync_done", draft_chars=len(draft))
                yield {"type": "status", "message": "Refining for accuracy and clarity…"}
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
                    rev = sync_client.chat.completions.create(
                        model=self.chat_model,
                        messages=rev_messages,
                        max_tokens=1536,
                        temperature=0.2,
                    )
                    final_text = (rev.choices[0].message.content or "").strip() or draft
                except Exception as rev_e:
                    logger.warning("Consultant stream review failed; using draft: %s", rev_e)
                    final_text = draft
            else:
                resp1 = sync_client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=1536,
                )
                final_text = (resp1.choices[0].message.content or "").strip()

            try:
                from rag.response_safety import enforce_consultant_quality, sanitize_user_facing_answer

                final_text = sanitize_user_facing_answer(final_text or "")
                final_text = enforce_consultant_quality(
                    final_text or "",
                    query=b.get("query") or "",
                    data_used=data_used,
                )
            except Exception as se:
                logger.warning("stream answer sanitize skipped: %s", se)

            for piece in self._iter_display_chunks(final_text):
                yield {"type": "delta", "text": piece}

            if pr:
                pr.step(
                    "llm_output_stream_done",
                    answer_chars=len(final_text or ""),
                    review_used=not review_disabled,
                )

            data_used["final_review_pass"] = not review_disabled
            sources = self._consultant_sources_list(phly_meta, tavily_hits, results)
            elapsed = time.perf_counter() - start
            if pr:
                pr.step(
                    "consultant_stream_request_done",
                    total_ms=int(elapsed * 1000),
                    tavily_hits=tavily_hits,
                    pinecone_chunks=len(results),
                )
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
            final_stream_answer = final_text or ""
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

        from rag.consultant_progress_log import new_progress_logger

        pr = new_progress_logger()

        if cacheable:
            cached_hit = cache_get(q)
            if cached_hit:
                if pr:
                    pr.step("path_cache_hit", short_circuit=1)
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
                progress=pr,
            )
            if kind == "small_talk":
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

            review_disabled = (
                (os.getenv("CONSULTANT_FAST_MODE") or "").strip().lower()
                in ("1", "true", "yes")
                or (os.getenv("CONSULTANT_REVIEW_DISABLED") or "").strip().lower()
                in ("1", "true", "yes")
                or _env_truthy("CONSULTANT_LOW_LATENCY")
            )
            if pr:
                pr.step(
                    "llm_turn_start",
                    model=self.chat_model,
                    review_pass=not review_disabled,
                    context_chars=len(context),
                    path="sync",
                )

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
            user_content = f"""Consider the full conversation above and the layered context below (aircraft record, registration/registry lines, market/listing rows, and supporting aviation knowledge excerpts). If the user's message is a follow-up, interpret it in light of your previous answer.

Synthesize a professional draft: When the context includes structured aircraft/registry/market facts, treat them as authoritative for identity, status, and registrant details, and you may use neutral phrasing like *per the registration materials in this brief* (no internal system names). When the answer is **general** and not grounded in those facts, **do not** imply registry or market data—use *typical operational performance for this class* / *from a broker’s perspective*. Never mention internal datasets, pipelines, infrastructure, or tool names.

Context:
{context}

Current question: {b["query"]}
{_consultant_purchase_tail(b)}
Provide a thorough client-facing answer. Plain text and bullet points (-). No ** or # headers. You may use • ✓ → sparingly."""
            messages.append({"role": "user", "content": user_content})
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=1536,
            )
            draft = (response.choices[0].message.content or "").strip()
            if pr:
                pr.step("llm_draft_sync_done", draft_chars=len(draft))

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

            # Last-mile safety: strip internal dataset/infrastructure naming from user-visible output.
            try:
                from rag.response_safety import enforce_consultant_quality, sanitize_user_facing_answer

                answer = sanitize_user_facing_answer(answer or "")
                answer = enforce_consultant_quality(answer or "", query=b.get("query") or "", data_used=data_used)
            except Exception as se:
                logger.warning("answer sanitize skipped: %s", se)

            if pr:
                pr.step(
                    "llm_sync_complete",
                    answer_chars=len(answer or ""),
                    final_review_applied=not review_disabled,
                )

            elapsed = time.perf_counter() - start
            if pr:
                pr.step(
                    "consultant_sync_request_done",
                    total_ms=int(elapsed * 1000),
                    tavily_hits=tavily_hits,
                    pinecone_chunks=len(results),
                )
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

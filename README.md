# HyeAero.AI Backend — Aircraft Research & Valuation Consultant

Real-time AI assistant backend: RAG chat over Hye Aero data, **dynamic market comparison**, **predictive pricing**, and **resale advisory**. Data sources: Controller, AircraftExchange, FAA, Internal DB.

## Architecture

```
PostgreSQL (listings, sales, aircraft, FAA) ──┬──► RAG Pipeline ──► Pinecone
                                              └──► API (comparison, price, resale)
                                                          │
                                                    FastAPI ──► /api/rag/answer
                                                              /api/market-comparison
                                                              /api/price-estimate
                                                              /api/resale-advisory
```

## Components

- **API** (FastAPI): Chat (RAG), market comparison, price estimate, resale advisory
- **RAG**: Sync Postgres → Pinecone; query service (retrieve + LLM answer)
- **Market comparison**: Query listings by model, age, hours, region
- **Price estimate**: Heuristic from historical sales (extensible to ML)
- **Resale advisory**: Plain-English guidance via RAG/LLM

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Configure `.env` (see `.env.example` or docs):
   - `POSTGRES_CONNECTION_STRING` — required for all endpoints
   - `PINECONE_*`, `OPENAI_*` — required for RAG chat and resale advisory
   - `OPENAI_CHAT_MODEL` — optional (default `gpt-4o-mini`)
   - **ZoomInfo (Owner details):** Set `ZOOMINFO_CLIENT_ID`, `ZOOMINFO_CLIENT_SECRET`, `ZOOMINFO_REFRESH_TOKEN`. Do **not** put `ZOOMINFO_ACCESS_TOKEN` in `.env` (it changes when refreshed). Optionally set `ZOOMINFO_TOKEN_FILE` to a writable path (e.g. `/data/zoominfo_token`) so the refreshed token is persisted across restarts. See *Deployment (ZoomInfo)* below.

3. (Optional) Sync data to Pinecone for RAG:
```bash
python runners/run_rag_pipeline.py --limit 100
```

## Deployment (ZoomInfo)

For production, **do not store the ZoomInfo access token in `.env`** — it expires and is refreshed automatically. Use only long-lived credentials:

1. In `.env` (or your platform’s env config), set:
   - `ZOOMINFO_CLIENT_ID`
   - `ZOOMINFO_CLIENT_SECRET`
   - `ZOOMINFO_REFRESH_TOKEN`
2. Leave `ZOOMINFO_ACCESS_TOKEN` unset. On first use (or after 401), the backend will refresh and get a new token.
3. **Optional:** Set `ZOOMINFO_TOKEN_FILE` to a writable path (e.g. `/data/zoominfo_token` or `./zoominfo_token`). The backend will read the token from this file on startup and write the new token there after each refresh, so restarts reuse it without calling ZoomInfo again. If you don’t set this, the token lives only in memory and a refresh runs after each deploy/restart.

No code changes and no database are required; the token is either in memory or in the optional file.

### Tavily (optional — PhlyData FAA trustee web hints)

When `GET /api/phlydata/owners` returns FAA rows whose registrant looks like a **trustee / shell** (e.g. contains `TRUSTEE`) and a **mailing address** is present, the API may attach `tavily_web_hints` via the [Tavily](https://tavily.com) search API (no curated JSON file).

- `TAVILY_API_KEY` — required for hints (get a key from Tavily).
- `TAVILY_DISABLED=1` — turn off all Tavily calls.
- `TAVILY_MAX_RESULTS` — optional (default `5`, max `10`).
- `TAVILY_WHEN_CORP_AND_ADDRESS=1` — optional: also run Tavily for **corporate** registrants (name contains tokens like INC, LLC, CORP, LP, …) with a mailing address, even if the word TRUSTEE does not appear. **Increases API usage**; use when many shells omit “trustee” in the string.

**Tavily → LLM → ZoomInfo (PhlyData owners):**

- After Tavily returns snippets, if **`OPENAI_API_KEY`** is set, the backend runs a small JSON extraction (`tavily_llm_synthesis` on each FAA row).
- **`TAVILY_LLM_SYNTHESIS_DISABLED=1`** — skip the LLM step (Tavily hints only).
- **`TAVILY_LLM_ZOOMINFO_ON_LOW=1`** — also enqueue ZoomInfo for LLM confidence `low` (default: only `medium` / `high`).

Install includes `tavily-python`; the service falls back to a plain `requests` POST to `https://api.tavily.com/search` if the SDK is unavailable.

### Ask Consultant (RAG) pipeline

The consultant endpoint uses **PhlyData + FAA** (when serial/tail tokens match), then **LLM query expansion**, **Tavily** (using the expanded web string), **multi-query Pinecone retrieval**, a **draft** OpenAI answer, and an optional **final review** pass for tone and consistency with authoritative internal data.

- Uses the same **`TAVILY_API_KEY`** / **`TAVILY_DISABLED`** as above when Tavily is enabled.
- **`CONSULTANT_TAVILY_ADVANCED=1`** — use Tavily `search_depth=advanced` for consultant web calls (slower, richer snippets; optional).
- **`TAVILY_SEARCH_DEPTH`** — `basic` (default) or `advanced` for all Tavily calls that don’t override depth.
- **`CONSULTANT_REVIEW_DISABLED=1`** — skip the second LLM “editor” pass (faster, slightly less polished).

If there is **no** PhlyData match, **no** vector hits, and **no** Tavily results, the service falls back to **general knowledge** (same as before).

## Run API

```bash
cd backend
python runners/run_api.py
```

Or with uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  

**Allow access from other PCs:** The app binds to `0.0.0.0:8000`, but the server firewall must allow inbound TCP on port 8000. Otherwise you get "i/o timeout" from other machines.

- **Windows (on the server):** Run in **Administrator** PowerShell:
  ```powershell
  New-NetFirewallRule -DisplayName "HyeAero API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
  ```
- **Linux (on the server):** `sudo ufw allow 8000` then `sudo ufw reload` (or add an iptables rule for port 8000).
- **Cloud (AWS/DigitalOcean/etc.):** In the instance **Security Group** or **Firewall**, add an inbound rule: TCP port 8000 from `0.0.0.0/0` (or your frontend’s IP) if you want the API reachable from the internet.  

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/rag/answer` | Ask the Consultant (RAG over Hye Aero data) |
| POST | `/api/market-comparison` | Compare aircraft by model, region, hours, year |
| POST | `/api/price-estimate` | Predictive valuation from sale history |
| POST | `/api/resale-advisory` | Plain-English resale guidance (RAG) |
| GET | `/health` | Health check |

## Frontend integration

Set in frontend `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Ask Consultant uses **`POST /api/rag/answer/stream`** (SSE, streamed tokens) from the browser; **`POST /api/rag/answer`** remains for non-streaming clients. Both use `NEXT_PUBLIC_API_URL` / CORS (`http://localhost:3000` by default; set `CORS_ORIGINS` in production).

## Data sources (ETL)

Data is loaded by the **etl-pipeline** (Controller, AircraftExchange, FAA, Internal DB) into PostgreSQL. This backend reads from the same database for comparison and pricing, and from Pinecone (after RAG sync) for natural-language chat.

## Owner details (ZoomInfo)

The `/api/phlydata/owners/{serial}` endpoint combines:

- **Listings** (Controller, AircraftExchange): seller/dealer names and phones
- **FAA registry**: registrant/owner names and addresses
- **ZoomInfo** company data: company name, website, phones, full address, revenue, employees (count + range), founded, status, industries, social URLs, and related metadata (certified, continent, location count, contact count, parent)

Matching is conservative:

- Phone-first: if any ZoomInfo candidate matches the listing/FAA phone, that result is preferred.
- Content scoring: company/person name, location, and website are scored; weak name-only matches are rejected.
- Optional LLM fallback: when scores are ambiguous, vector search + LLM decide whether there is a safe match.

If no safe match is found, the endpoint still returns aircraft + listing/FAA owners, but with an empty `zoominfo_enrichment` array so the frontend shows “No ZoomInfo data found for this aircraft.”

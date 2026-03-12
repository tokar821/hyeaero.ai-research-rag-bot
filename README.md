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
   - `ZOOMINFO_ACCESS_TOKEN` — required for **Owner details** (ZoomInfo company/contact enrichment). Get a token from `phlydata-zoominfo` (run `python get_zoominfo_token.py` there) and copy the value into `backend/.env`. Without it, Owner details will show listings/FAA but no ZoomInfo company data.

3. (Optional) Sync data to Pinecone for RAG:
```bash
python runners/run_rag_pipeline.py --limit 100
```

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

Chat uses the Next.js proxy (`/api/chat` → backend `/api/rag/answer`). Market Comparison, Price Estimator, and Resale Advisory call the backend from the browser (CORS is enabled for `http://localhost:3000`).

## Data sources (ETL)

Data is loaded by the **etl-pipeline** (Controller, AircraftExchange, FAA, Internal DB) into PostgreSQL. This backend reads from the same database for comparison and pricing, and from Pinecone (after RAG sync) for natural-language chat.

## Future integrations

- **ZoomInfo API**: Integrating ZoomInfo could enrich lead/contact data (company and contact info). To add: obtain ZoomInfo API credentials, add a service layer that calls ZoomInfo endpoints, and optionally store or display enriched data in the dashboard or RAG context. All leads from the app can be referred to Hye Aero for follow-up.

# Biosyn‑KB — Evidence‑Grounded Biosynthesis Knowledge Base

![Workflow Overview](image_wf_up.png)

<sub>High‑level workflow: Strategist → Search/OA → Crawl → Extract → Summarize → Validate → Store → Compose.</sub>

Search, crawl, extract, and synthesize biosynthesis knowledge from the web into structured data and reports. The pipeline is LLM‑first for generation, with OA‑biased retrieval, optional headless crawling, and LLM‑backed RAG validation plus deterministic checks.

Highlights
- LLM‑first: Strategist → Summarizer → Composer (Ollama by default; OpenAI‑compatible optional)
- Ethical OA enrichment via Unpaywall; robots‑aware throttled crawling
- Deterministic checks and LLM‑RAG validation for numeric claims
- Web API with job queue, Docker/Compose deployment, Nextflow orchestration
- Checkpoints at each stage for auditability

---

## Overview

- Purpose: Build an evidence‑grounded biosynthesis KB from open web sources.
- Outputs:
  - Structured JSONL summaries (PageSummary schema)
  - SQLite DB (pages, summaries, metrics, enzymes, organisms, etc.)
  - Consolidated, citation‑rich Markdown reports
  - Checkpoints per stage (plan, urls, crawl, extract, summaries, validation)

## Architecture & Flow

1) Plan (LLM Strategist): expands a natural query to subqueries and OA‑friendly domains.
2) Gather: paginated search (SerpAPI/Brave) → OA enrichment (Unpaywall) → robots‑aware crawl (httpx; optional Playwright headless for JS sites).
3) Generate: clean extraction (trafilatura→bs4 fallback) → LLM summarization (large context, chunked, no nulls).
4) Validate: deterministic checks (year, ranges, EC format, evidence presence) + LLM over retrieved paragraphs (RAG) to confirm metrics.
5) Store: import pages and summaries to SQLite.
6) Synthesize: compose a chronological report with inline [n] citations; optional “Experimentalist” and “Compute” analyses.

Checkpoints are written under `work/<run>/checkpoints/`:
- `01_plan.json`, `02_urls.json`, `03_crawl_report.json`, `04_extract_report.json`, `05_summaries_quality.json`, `06_validation_report.json`, `00_index.json`.

### Workflow Diagram

For a visual overview of the full pipeline and agent interactions:

- Inline image (above) — `Image_wf.png` (quick view)
- Full PDF diagram: [Figure.pdf](Figure.pdf)
- Example report (PDF): [SynBioKB_Agent.pdf](SynBioKB_Agent.pdf)

Open it on GitHub to preview the PDF inline.

### End‑to‑End Summary (At a Glance)

- Input: your natural‑language query + API keys (`apis.yaml`) and an LLM provider (Ollama by default).
- Plan: the Strategist LLM proposes subqueries and OA domains.
- Search/OA: we collect links (SerpAPI/Brave) and enrich DOIs via Unpaywall to prefer OA.
- Crawl: robots‑aware httpx (or Playwright for JS) saves HTML to `pages/`.
- Extract: we clean HTML → `{file,title,text}` JSONL.
- Summarize (LLM): PageSummary JSONL (facts + narratives) under a strict schema.
- Validate: deterministic checks + async LLM‑RAG confirmation against retrieved paragraphs.
- Store: import pages and summaries into SQLite (typed tables).
- Compose (LLM): produce a Markdown report with inline [n] citations.

What you provide
- Query/topic; keys in `apis.yaml` (SerpAPI or Brave) + Unpaywall email; LLM model/provider (`llm.yaml` optional).
- Optional knobs: `max_results` (search), summarization `limit`, `max_chars`, model.

What you get
- `artifacts/*.jsonl`: PageSummary JSONL
- `artifacts/*.db`: SQLite with pages/summaries/metrics/enzymes/organisms
- `work/<run>/report.md`: consolidated, citation‑rich report
- `work/<run>/checkpoints/*.json`: audit trail for every stage

---

## End‑to‑End Task (What Happens, Inputs, Outputs)

What the pipeline does
- Converts a natural‑language goal (e.g., “3‑HP biosynthesis from acetate”) into a concrete plan, finds OA sources, crawls and cleans the pages, extracts structured facts with an LLM, validates numbers against retrieved evidence, stores everything into SQLite, and writes a citation‑rich report.

Inputs (user‑provided)
- Query: natural‑language topic or question.
- Keys: SerpAPI/Brave key, Unpaywall email (in `apis.yaml`), and an LLM provider (Ollama default).
- Optional knobs: `max_results` (search), `limit` (pages to summarize), `max_chars`, model selection.

Outputs (artifacts)
- `artifacts/*.jsonl`: Structured PageSummary JSONL (one per source).
- `artifacts/*.db`: SQLite KB with pages/summaries/metrics/enzymes/organisms.
- `work/<run>/report.md`: Merged, chronological report with [n] citations.
- `work/<run>/checkpoints/*.json`: Audit trail for each stage.

Contract (per stage)
- Plan → `01_plan.json` (LLM strategist output; subqueries + domain hints).
- Search/OA → `02_urls.json` (deduped URLs + OA links).
- Crawl → HTML files under `pages/<run>` + `03_crawl_report.json` summary.
- Extract → `artifacts/<run>.jsonl` (clean text records) + `04_extract_report.json`.
- Summarize → `artifacts/<run>_summaries.jsonl` + `05_summaries_quality.json`.
- Validate → `06_validation_report.json` with deterministic and LLM‑RAG warnings.
- Store → `artifacts/<run>.db` (SQLite).
- Compose → `work/<run>/report.md` final report.

Success criteria
- PageSummary JSONL lines exist and parse; DB contains rows in `summaries` and `metrics` for the topic; report is generated with [n] citations; validation checkpoint lists either empty warns or actionable notes.

---

## Prerequisites

- Keys: SerpAPI (or Brave) and an Unpaywall contact email.
- LLM: local Ollama (default) or online provider (OpenAI‑compatible).
- Python 3.11+ locally, or Docker/Compose for containerized runs.

---

## Tools & Packages (By Purpose)

- LLMs
  - Runtime: Ollama (local) — default provider
  - Models: gpt-oss:20b (tested), others supported via Ollama; OpenAI‑compatible providers optional
  - Client: `httpx` (custom clients in `src/biosyn_kb/llm/`)

- Search & OA
  - Web search: SerpAPI (Google) or Brave Search (`httpx` clients)
  - OA enrichment: Unpaywall (`httpx`), DOI detection via regex
  - URL/domain helpers: `tldextract`, `PyYAML` for config

- Crawl
  - HTTP crawler: `httpx.AsyncClient`, robots via `urllib.robotparser`
  - Headless (JS sites): Playwright Chromium (optional)

- Extract
  - HTML cleaner: `trafilatura` → fallback to `beautifulsoup4`
  - PDF: `pdfminer.six` (and `pdfplumber` if available)

- Summarize & Schema
  - Prompting: custom prompts under `src/biosyn_kb/summarize/`
  - Schema/validation: `pydantic v2`
  - Async orchestration: `asyncio`

- Storage (KB)
  - ORM: `SQLAlchemy 2`
  - DB: SQLite (file under `artifacts/*.db`)

- RAG & Embeddings
  - Vector store: `chromadb` (PersistentClient)
  - Embeddings: `sentence-transformers` (all‑MiniLM‑L6‑v2), downloads on first run

- Web/API & Queue
  - API/UI: `fastapi`, `uvicorn`, `Jinja2`
  - Queue: SQLite‑backed lightweight worker (custom)

- Orchestration
  - CLI: `biosyn-kb` entrypoints
  - Nextflow: workflows under `nextflow/` and `workflows/nextflow/`
  - Docker/Compose: containerized app + Ollama

- Testing & Docs
  - Tests: `pytest`
  - Docs/PDF helper: `reportlab` (for setup guide script)

---

## Tested Setup (Current Version)

- Python: 3.11
- OS: macOS (dev); Linux expected to work (Playwright adds platform deps)
- LLMs: Ollama local with `gpt-oss:20b` (primary test model)
- DB: SQLite files under `artifacts/`
- RAG: ChromaDB + SentenceTransformers (all‑MiniLM‑L6‑v2)
- Headless crawling: Playwright Chromium installed locally or via Dockerfile
- Orchestration: CLI verified; web/queue endpoints live; Nextflow maps to same CLI stages

Notes:
- OpenAI‑compatible providers can be used by setting `--provider openai --base-url ... --api-key ...` where applicable.
- The E2E validator uses an async LLM‑RAG path to confirm numeric metrics in retrieved contexts.

## Install (Local)

```
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

## Configuration

1) API keys (`apis.yaml` preferred):

```
serpapi:
  api_key: "<your_serpapi_api_key>"
  engine: "google"

brave:
  api_key: "<your_brave_api_key>"

unpaywall:
  email: "you@example.com"
```

2) LLM defaults (`llm.yaml`, optional):

```
llm:
  provider: ollama            # or openai
  base_url: http://localhost:11434
  model: gpt-oss:20b
  temperature: 0.0
  top_p: 1.0
  seed: 42
  # api_key: ...               # for online providers
```

3) Environment (optional overrides):
- `UNPAYWALL_EMAIL` (fallback for apis.yaml)
- `LLM_PROVIDER`, `LLM_BASE_URL` (container defaults)

---

## Quick Start (CLI)

- Search (paginated to reach max):
```
biosyn-kb search --query "biosynthesis of caffeine" --max-results 20 --provider serpapi --api-config apis.yaml
```

- Crawl a set of URLs (robots‑aware; headless optional):
```
biosyn-kb crawl --urls-file urls.txt --save-html-dir pages/demo --max-concurrency 4 --per-domain-delay 1.0 [--headless]
```

- Extract clean text from saved HTML (+PDFs):
```
biosyn-kb extract --html-dir pages/demo --out artifacts/pages_clean.jsonl --clean --include-pdfs
```

- Summarize to structured JSONL (large context, chunked):
```
biosyn-kb summarize --in artifacts/pages_clean.jsonl --out artifacts/summaries.jsonl \
  --provider ollama --base-url http://localhost:11434 --model "gpt-oss:20b" \
  --limit 10 --max-chars 12000 --chunked --chunk-chars 3500
```

- Compose a multi‑article report:
```
biosyn-kb compose-report --in artifacts/summaries.jsonl --out work/report.md \
  --provider ollama --base-url http://localhost:11434 --model "gpt-oss:20b"
```

---

## Web API (Docker/Compose)

Build + run (Docker):
```
docker build -t biosyn-kb:latest SynBioKB-Agent
docker run --rm -p 8000:8000 \
  -e UNPAYWALL_EMAIL="you@example.com" \
  -e LLM_PROVIDER=ollama -e LLM_BASE_URL="http://host.docker.internal:11434" \
-v "$(pwd)/SynBioKB-Agent/apis.yaml:/app/apis.yaml:ro" \
-v "$(pwd)/SynBioKB-Agent/pages:/app/pages" -v "$(pwd)/SynBioKB-Agent/work:/app/work" -v "$(pwd)/SynBioKB-Agent/artifacts:/app/artifacts" \
  biosyn-kb:latest
```

Compose (Ollama + app):
```
cd biosyn-kb
UNPAYWALL_EMAIL=you@example.com docker compose up -d --build
```

API:
- Swagger: `http://localhost:8000/docs`
- Queue e2e run:
```
curl -X POST http://localhost:8000/api/e2e -H "Content-Type: application/json" -d '{
  "query":"3-HP biosynthesis",
  "api_config":"/app/apis.yaml",
  "provider":"ollama", "base_url":"http://ollama:11434", "model":"gpt-oss:20b",
  "limit":3, "max_results":20, "max_chars":12000
}'
```
- Monitor: `GET /api/jobs`, `GET /api/job/{id}` or visit `http://localhost:8000/status`.
- Outputs (mounted to host): `pages/api_e2e/`, `artifacts/api_e2e*.jsonl|.db`, `work/api_e2e/report.md`.

---

## Agents & Multi‑Agent Framework

Roles (CrewAI style; all agent code lives under `src/biosyn_kb/agents/`):
- Strategist (LLM): Expands your goal into subqueries, synonym variants, and OA‑friendly domain hints (e2e flow).
- Extractor (code): Parses HTML into clean, relevant text (trafilatura→bs4 fallback), optional paragraph filtering.
- Summarizer (LLM): Converts page text into structured PageSummary JSON under a strict schema, with narrative fields.
- Validator (code + LLM):
  - Deterministic checks: units, EC formats, plausible ranges, evidence presence.
  - LLM‑RAG checks: retrieves top paragraphs from the DB and asks the LLM to confirm numeric metrics (OK/WARN).
- Store (code): Imports pages and summaries into SQLite with typed tables.
- Composer (LLM): Merges multiple PageSummary records into a single report (chronological sections, compact comparison table, inline [n] citations mapping to URLs).
- Experimentalist (LLM): Extracts practical experimental insights (host choice, pathway logic, fermentation parameters, bottlenecks).
- Compute (LLM): Normalizes units, summarizes metric ranges, finds best‑in‑class results, and discusses trends.

Execution modes
- CLI single‑stage commands (search, crawl, extract, summarize, compose, qa).
- crewai‑run (local crew) and crewai‑e2e (LLM strategist → full pipeline).
- Web API `/api/e2e` with background queue: queues the e2e pipeline; view progress at `/status`.
- Nextflow E2E or modular: run big batches or integrate into schedulers.

### Deep Dive: Multi‑Agent Workflow (Who does what, and how it connects)

- Strategist (LLM)
  - Input: user query
  - Output: 4–6 subqueries + OA‑friendly domain hints (checkpoint: `01_plan.json`)
  - Why: widen recall with synonyms/variants and bias to open‑access sources.

- Searcher + OA Enricher (code)
  - Input: subqueries and/or original query; provider keys (`apis.yaml`)
  - Output: URL list (after OA enrichment via Unpaywall) (checkpoint: `02_urls.json`)
  - Why: collect candidates, prefer OA HTML/PMC over paywalled landing pages.

- Crawler (code; httpx or optional Playwright)
  - Input: URLs
  - Output: saved HTML files under `pages/<run>/` (checkpoint: `03_crawl_report.json`)
  - Why: durable capture and reproducibility.

- Extractor (code)
  - Input: saved HTML (and PDFs when enabled)
  - Output: clean `{file,title,text}` JSONL (checkpoint: `04_extract_report.json`)
  - Why: remove boilerplate; keep signal paragraphs for the LLM.

- Summarizer (LLM)
  - Input: clean text JSONL
  - Output: PageSummary JSONL under strict schema (checkpoint: `05_summaries_quality.json`)
  - Why: convert unstructured text into structured facts + narrative sections.

- Validator (code + LLM‑RAG)
  - Input: PageSummary JSONL (+ paragraphs from DB)
  - Output: warnings (deterministic + LLM‑RAG) (checkpoint: `06_validation_report.json`)
  - Why: sanity‑check numbers and confirm they appear in retrieved contexts.

- Store (code)
  - Input: pages + summaries JSONL
  - Output: SQLite DB with typed tables (pages, summaries, metrics, enzymes, organisms…)
  - Why: enable queries, RAG, and downstream analytics.

- Composer (LLM)
  - Input: multiple PageSummary records
  - Output: consolidated Markdown report with inline [n] citations
  - Why: synthesize across studies (chronology, tables, comparisons).

Data flow summary: query → (Strategist) plan → (Search+OA) URLs → (Crawler) HTML → (Extractor) text → (Summarizer) PageSummary → (Validator) warnings → (Store) DB → (Composer) report.

Inputs you provide
- Query/topic; keys in `apis.yaml` (SerpAPI or Brave) + Unpaywall email; LLM config/provider.
- Optional knobs: `max_results`, summarization `limit`, `max_chars`, model selection.

Outputs you can expect
- JSONL of PageSummary records under `artifacts/`
- SQLite DB with pages/summaries/metrics/etc. under `artifacts/`
- Markdown report under `work/<run>/report.md`
- Checkpoints for every stage under `work/<run>/checkpoints/`

---

## Nextflow Orchestration

- E2E:
```
nextflow run nextflow/main.nf -c nextflow/nextflow.config \
  --query "3-HP biosynthesis" --api_config apis.yaml \
  --llm_provider ollama --llm_base_url http://localhost:11434 \
  --model "gpt-oss:20b" --limit 5 --max_results 50
```
- Modular: set `--e2e false` to run search/crawl → extract → summarize → import → compose in steps.

---

## Validation & Checkpoints

- Deterministic checks: year plausibility, chemical presence, metric sanity (yield 0..100, titer/productivity >0), EC format, evidence present for numeric claims.
- LLM‑RAG checks: retrieve top paragraphs from DB; the LLM judges whether each metric is supported (OK/WARN).
- Checkpoints written to `work/<run>/checkpoints/` for auditing and tuning.

---

## Crawling Strategies

- HTTPX crawler: robots‑aware, per‑domain delay, concurrency; default for speed & politeness.
- Headless crawler (Playwright Chromium): `biosyn-kb crawl --headless` for JS‑rendered pages; heavier and slower; use selectively.

---

## Example End‑to‑End (CLI, small batch)

```
# 1) Search + crawl (2 pages)
biosyn-kb run-all --query "3-HP biosynthesis" --max-results 2 \
  --provider serpapi --api-config apis.yaml \
  --save-html-dir pages/smoke --per-domain-delay 1.0 --max-concurrency 4

# 2) Extract clean text
biosyn-kb extract --html-dir pages/smoke --out artifacts/smoke.jsonl --clean --max-chars 6000

# 3) Summarize one page
biosyn-kb summarize --in artifacts/smoke.jsonl --out artifacts/smoke_summ.jsonl \
  --model "gpt-oss:20b" --limit 1 --max-chars 6000 --chunked --chunk-chars 3000

# 4) Compose a report
mkdir -p work/smoke
biosyn-kb compose-report --in artifacts/smoke_summ.jsonl --out work/smoke/report.md \
  --provider ollama --base-url http://localhost:11434 --model "gpt-oss:20b"
```

Expected: JSONL summaries in `artifacts/`, final Markdown under `work/smoke/report.md`. For blocked pages (403/JS), switch to headless crawl or bias the strategist to OA domains.

---

## RAG & Embeddings

- Vector store: ChromaDB with SentenceTransformers (`all‑MiniLM‑L6‑v2`) required. The hashing fallback is disabled to ensure quality retrieval.

---

## Makefile Targets

- `install`, `test`, `docs`
- `clean`, `clean-artifacts`
- `compose-up`, `compose-down`
- `api-e2e` — queue a small end‑to‑end job to the local API
- `demo-fast` — extract‑only sanity run (no LLM)

---

## Scripts (A/B)

- `scripts/ab_crawl.sh "QUERY" <tag>` — compare httpx vs headless capture counts on the same URLs.
- `scripts/ab_compare_extract.py <html_dir> <out_dir>` — compare cleaner vs bs4 average text length.

---

## Troubleshooting

- 404 on `/api/e2e`: rebuild the image with `--no-cache` and restart; ensure `/docs` shows the endpoint.
- "Unknown job type": clear old queue DB and rebuild; new handlers support `agent`, `crew`, and `e2e`.
- 403/"Just a moment…": prefer OA links (Unpaywall enrichment); use `--headless` for JS‑guarded pages.
- Ollama connectivity: in Docker use `host.docker.internal:11434` (Mac) or add `--add-host=host.docker.internal:host-gateway` on Linux; in Compose use `http://ollama:11434`.

---

## Repository Layout

- `src/biosyn_kb/` — core modules: `crawl/`, `search/`, `extract/`, `summarize/`, `validate/`, `rag/`, `agents/`, `report/`, `webapp/`
- `SynBioKB-Agent/` — Dockerfile, docker‑compose.yml, Makefile, nextflow, scripts
- `pages/`, `work/`, `artifacts/` — generated outputs (ignored by VCS)

---

## Roadmap

- Expose headless mode as an API flag in `/api/e2e`
- Crossref DOI discovery to improve OA coverage
- CI workflows (tests, image build, release)
- Richer validators and report templates

---

## License

TBD.

---

## Appendix — Orchestration Cheat‑Sheet, Examples, and Tips

### Orchestration Cheat‑Sheet (CLI • Nextflow • Web)

- CLI (interactive, small batches)
  - Search + OA + Save: `biosyn-kb run-all --query "..." --max-results 5 --provider serpapi --api-config apis.yaml --save-html-dir pages/demo`
  - Local pipeline (LLM): `biosyn-kb pipeline-run --html-dir pages/demo --work-dir work/demo --db artifacts/demo.db --out artifacts/demo.jsonl --model "gpt-oss:20b" --limit 3`
  - E2E CrewAI (strategy→compose): `biosyn-kb crewai-e2e --query "..." --api-config apis.yaml --html-dir pages/e2e --work-dir work/e2e --db artifacts/e2e.db --out artifacts/e2e.jsonl --provider ollama --base-url http://localhost:11434 --model "gpt-oss:20b" --limit 5 --max-results 50`

- Nextflow (repeatable workflows)
  - E2E: `nextflow run nextflow/main.nf -c nextflow/nextflow.config --query "..." --api_config apis.yaml --llm_provider ollama --llm_base_url http://localhost:11434 --model "gpt-oss:20b" --limit 5 --max_results 50`
  - Modular: add `--e2e false` to run stepwise (search/crawl → extract → summarize → import → compose) and inspect artifacts.

- Web + Queue (UI and JSON API)
  - Start: `uvicorn biosyn_kb.webapp.main:app --host 127.0.0.1 --port 8000`
  - UI: open `http://127.0.0.1:8000/` (or `/chat`) and submit queries
  - API (queue E2E): `POST /api/e2e` with payload:
```
{
  "query": "3-HP biosynthesis",
  "api_config": "apis.yaml",
  "provider": "ollama",
  "base_url": "http://localhost:11434",
  "model": "gpt-oss:20b",
  "limit": 3, "max_results": 20, "max_chars": 12000
}
```
  - Monitor: `GET /api/jobs`, `GET /api/job/{id}`, or visit `/status`
  - Outputs: `pages/api_e2e/`, `artifacts/api_e2e*.jsonl|.db`, `work/api_e2e/report.md`

Tips:
- After code changes, restart the web app so the background worker picks up new job types and validator logic.
- The queue understands `agent`, `crew`, and `e2e` jobs.

### Worked Example — “Isobutanol from acetate”

1) Gather and save a page:
```
biosyn-kb run-all --query "isobutanol acetate" --max-results 1 \
  --provider serpapi --api-config apis.yaml \
  --save-html-dir pages/local_smoke --per-domain-delay 0.5 --max-concurrency 2
```

2) LLM summarization → DB import:
```
biosyn-kb pipeline-run \
  --html-dir pages/local_smoke \
  --work-dir work/iso_demo \
  --db artifacts/iso_demo.db \
  --out artifacts/iso_demo.jsonl \
  --model "gpt-oss:20b" --limit 1 --max-chars 6000
```

3) RAG QA (embeddings + LLM):
```
biosyn-kb qa --db artifacts/iso_demo.db \
  --question "What titers for isobutanol are reported in the captured pages?" \
  --k 3 --model "gpt-oss:20b"
```

Expected:
- `artifacts/iso_demo.jsonl` — 1 PageSummary record
- `artifacts/iso_demo.db` — entries in `pages`, `summaries`, `metrics`
- `artifacts/rag/` — Chroma index after the QA step

### Queue Worker (manual control)

- Enqueue agent job:
```
biosyn-kb queue-add-agent \
  --queue-db artifacts/queue.db \
  --query "biosynthesis of isobutanol" \
  --provider serpapi --api-config apis.yaml \
  --max-results 3 --exclude-domain wikipedia.org \
  --save-html-dir pages/agent
```

- Process one job and exit:
```
biosyn-kb queue-worker-once --queue-db artifacts/queue.db
```

- Enqueue a summarize job:
```
biosyn-kb queue-add-summarize \
  --queue-db artifacts/queue.db \
  --in artifacts/pages.jsonl \
  --out artifacts/summaries.jsonl \
  --model "gpt-oss:20b"
```

### Additional Tips

- Playwright headless (JS pages):
  - Local: `python -m playwright install --with-deps chromium`
  - Dockerfile already installs Chromium for headless crawling.

- PDF text extraction:
  - `--include-pdfs` also parses `*.pdf` in the same directory (or `--pdf-dir`). PDF parsing uses `pdfminer.six` (installed) with a `pdfplumber` first try when available.

- RAG embeddings:
  - First QA can download `all‑MiniLM‑L6‑v2` (SentenceTransformers). Allow time/bandwidth, especially in Docker.

- Secrets & hygiene:
  - Do not commit real keys. `apis.yaml`, `llm.yaml`, `.env` are ignored; only `*.example` files are tracked.
  - Consider adding CI to run `pytest -q` and a tiny smoke `pipeline-run` on a fixed HTML fixture.

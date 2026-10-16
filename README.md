# Financial RAG

Enterprise-grade Retrieval-Augmented Generation platform for multi-ticker SEC 10-K
financial analysis with strict numerical hallucination guardrails, table-aware
retrieval, conversational memory/coreference, RBAC, and a production-oriented
containerized stack.

> Deep architectural detail: **[PROJECT_MAP.md](PROJECT_MAP.md)** (30+ sections,
> verified against the working tree).

---

## Executive Summary

Financial RAG ingests SEC 10-K filings (HTML/SGML/TXT plus API-uploaded
PDF/DOCX/HTML/SGML/TXT), indexes them into a **dual store** (MongoDB document
store + Qdrant vector store), and answers grounded financial questions using
dense+sparse hybrid retrieval, cross-encoder reranking, table-aware context
construction, Groq LLM generation, and an async numerical-claim guardrail.

**What it solves.** Financial filings are long and form-heavy: numbers live in
embedded tables, the answer depends on *which fiscal year* you mean, and
cross-company comparison needs balanced evidence from *each* ticker. A fluent but
wrong revenue figure is worse than a refusal — so the platform treats a refusal
and a verified number as first-class, equal outcomes.

---

## System Architecture & Containerized Stack

The stack runs on **Docker Compose** (project `financial-rag`) — six active
services plus a background worker:

| Service | Role |
|---|---|
| **FastAPI backend** | Auth, RBAC, rate limiting, audit logging, SSE streaming, session memory, analytics. `app/api/main.py` |
| **arq worker (Redis-backed)** | Background task queue. Standard API requests are served **synchronously**; the `arq` worker is integrated for **high-concurrency ingestion, heavy background processing, and scheduled tasks** under high-traffic load, so the primary API stays responsive and non-blocking. `app/api/worker.py` |
| **Next.js 15 frontend** | Dashboard: auth, chat, upload, analytics (`frontend/`) |
| **MongoDB 7** | Document store: `raw_chunks`, `users`, `rag_audit_logs` |
| **Qdrant** | Vector store: `financial_vectors` (768-dim, Cosine, HNSW) |
| **Redis 7** | Two-tier semantic cache, rate-limit buckets, auth-token blacklist, **and ARQ broker** for the worker |
| **MLflow** | Experiment tracking + registry (artifacts + file store) |

**Request lifecycle.** `POST /api/v1/chat` / `/chat/stream` restores session
memory → fiscal-year normalization and multi-ticker detection → semantic-cache
lookup → hybrid retrieval (Qdrant dense ANN + BM25, fused by RRF) → cross-encoder
rerank (40 → 8) → table shield + cylinder reorder → `_augment_context`
(query-grounded table injection under a shared budget) → Groq generation →
**async guardrail** → cache write-back (pass only) → response/SSE → audit log.
The pipeline is **stateless**; session memory lives at the API layer.

```mermaid
flowchart TD
    U[User] --> FE[Next.js 15 Dashboard]
    FE -->|POST /api/v1/chat/stream| API[FastAPI app/api/main.py]
    API --> AUTH[JWT + RBAC + RateLimit]
    API --> PIPE[FinancialRAGPipeline]
    PIPE --> CACHE[Redis semantic cache (2-tier)]
    PIPE --> RET[Hybrid dense + BM25 + RRF]
    RET --> RERANK[CrossEncoder FP16 reranker]
    RERANK --> CONTEXT[_augment_context + table injection]
    CONTEXT --> GEN[Groq generator structured JSON]
    GEN --> GRD[AsyncGuardrail D1-D8]
    GRD -->|PASS only| CACHE
    API --> AUDIT[MongoDB rag_audit_logs]
    API --> BG[arq worker (Redis-backed)]
    BG --> PIPE
    RET -.Qdrant.- QD[(Qdrant financial_vectors)]
    CONTEXT -.Mongo.- MO[(MongoDB raw_chunks)]
```

---

## Key Innovations & Production Features

### D1–D8 Guardrails (numerical hallucination control)

| ID | Defect / capability | Resolution | Code |
|----|---------------------|-----------|------|
| D1 | Guardrail verdict computed then discarded | Real verdict propagated end-to-end (`passed`/`failed`/`error`/`not_run`) into API response, audit log, and cache gate | `src/pipeline.py`, `app/api/main.py` |
| D1b | Cache served entries without enforcing `guardrail_passed` | Cache hit ignored unless `guardrail_passed is True` | `src/pipeline.py` |
| D2 | `guardrail_pass_rate` divided by ineligible events | Rate over eligible events only; `eligible_queries` exposed | `app/api/analytics.py` |
| D3 | Latency measured with non-monotonic wall clock | `time.perf_counter()`; latency outliers dropped and surfaced | `app/api/analytics.py`, `app/api/main.py` |
| D4 | `context_quality` buckets degenerate | Score normalization + tertile thresholds (0.66 / 0.33) | `app/api/analytics.py` |
| D5 | Refusals validated only via structured-output failure | **First-class refusal schema** via `model_validator`, separate `RefusalAnswer`, `classify_generation_output()` (`structured` / `refusal` / `malformed`) | `src/5_generation/schemas.py` |
| D6 | MongoDB ↔ Qdrant drift with no visibility or safe tooling | `/health.mongo_qdrant_parity` + scoped `safe_replacement.py`, identity reconcilers | `app/api/main.py`, `src/1_ingestion/*` |
| D7 | Demo-script correctness | Corrected company labels + step numbering | `docs/DEMO_SCRIPT.md` |
| D8 | **Year-token false-positive suppression** | Standalone year tokens (`"2025."`, `"2025,"`) were captured as claims because `_NUM_EXTRACT` absorbs trailing sentence punctuation and `_normalize_number` never stripped it → spurious `failed` verdicts on correct answers. Fixed by `s.rstrip(".,")` in `_normalize_number`, keeping internal decimals intact | `src/5_generation/async_guardrail.py` |

### Other production features

- **Multi-ticker balanced retrieval** — `_balanced_ticker_subretrievals` runs
  per-ticker sub-retrieval merged by `chunk_id` so no company dominates the
  evidence set (`src/pipeline.py`).
- **Memory & coreference resolution** — session-scoped memory at the API layer
  resolves cross-turn references ("the second company" → Microsoft) with the
  fiscal year carried forward (`app/api/main.py`).
- **D5 first-class refusal schema** — out-of-domain / out-of-scope questions
  return an explicit refusal with a real `guardrail_status` instead of
  hallucinated figures.
- **Admin-gated RBAC** — `require_roles(["admin"])` blocks non-admin
  `POST /api/v1/documents/upload` and `POST /api/v1/ingest` with **HTTP 403**
  (verified live in the E2E pass); JWT + bcrypt, rate limiting, token revocation.
- **Hybrid retrieval** — dense (Qdrant ANN) + sparse (BM25) fused with RRF
  (`k=60`, top 40 → rerank top 8), FP16 cross-encoder with FP32 CPU fallback.
- **Table-aware retrieval** — atomic Markdown tables, placeholder isolation,
  query-grounded re-injection under a shared context budget.

---

## Evaluation Metrics & Test Evidence

All figures below are **verified, code-backed or persisted-evidence backed**.
Do not extend this table with numbers that are not reproducible from the cited
artifacts.

### Evaluation scores (persisted)

| Metric | Score |
|---|---|
| **Faithfulness (F)** | **0.990** |
| **Recall (R)** | **0.920** |
| **Precision (P)** | **0.845** |
| **Context Quality (C)** | **0.847** |

Recorded in `artifacts/EVALUATION_SUMMARY.md` / `artifacts/evaluation_scores.json`
and rendered by the System Analytics tab.

### Test suite evidence

Persisted in **`test_artifacts/TEST_RUN_REPORT.md`** (with raw logs):

- **Live stack run (in-container, isolated test DB): 662 passed, 1 failed**
  (`pytest_live3.log`, 9:48). The single failure is an image-only dependency
  mismatch (`ragas` → `langchain_community`), passes on host. See
  `test_artifacts/TEST_RUN_REPORT.md` for the full attribution.
- **31 test files / 517 test functions** in the suite.
- **D8 regression test** `test_guardrail_trailing_punctuation_not_spurious_failure`
  passes along with `test_generation_stage.py` + `test_guardrail_verdict_propagation.py`
  (**50 passed**) and `test_stream_guardrail_enforcement.py` (**19 passed**).

---

## Quickstart

1. **Prepare env** — `cp .env.example .env`, then fill in `GROQ_API_KEY`,
   `LLAMA_CLOUD_API_KEY`, `JWT_SECRET_KEY`, `SUPERADMIN_EMAIL`,
   `SUPERADMIN_PASSWORD`.
2. **Launch the stack** — from the repository root:
   ```bash
   docker compose up -d --build
   ```
3. **Open the services:**
   - Frontend: **`http://localhost:3000`**
   - Backend API / OpenAPI: **`http://localhost:8000/docs`**
   - Health: **`http://localhost:8000/health`**
4. **Verify live** — run the scenarios in
   **[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)** (Tests 1–5 + admin
   ingestion/Test 6). Test evidence: [`test_artifacts/TEST_RUN_REPORT.md`](test_artifacts/TEST_RUN_REPORT.md).

---

## Project Structure

```
Financial_RAG/
├── app/api/            FastAPI: main.py (endpoints, session memory, SSE, upload,
│                       audit, analytics), schemas.py, worker.py (arq jobs),
│                       rate_limiter.py, db_logger.py, analytics.py, parsers.py,
│                       auth/ (router, jwt, service, dependencies)
├── src/
│   ├── pipeline.py     RAG orchestrator: fiscal-year, multi-ticker, context
│   ├── model_loader.py 3-tier pipeline loader (cache → MLflow → direct)
│   ├── 1_ingestion/    parse → clean → metadata → chunk → index (+ D6 reconcilers)
│   ├── 2_caching/      redis client → two-tier semantic cache
│   ├── 3_pre_retrieval/ intent router → query expansion (DISABLED by default)
│   ├── 4_retrieval/    hybrid search → reranker → post-retrieval → table shield
│   └── 5_generation/   generator → async_guardrail (D1–D8) → schemas (D5) → stream
├── config/             settings.py (env-driven) → logging_config.py (JSON logger)
├── evaluation/         synthetic generator → batch runner → judge → mlflow tracker
├── frontend/           Next.js 15 dashboard
├── test/               31 files / 517 functions
├── test_artifacts/     TEST_RUN_REPORT.md (live evidence) + raw logs
├── docs/               DEMO_SCRIPT.md, CUTOVER_REPORT.md, audit write-ups
├── artifacts/          evaluation_scores.json, EVALUATION_SUMMARY.md, benchmarks
└── docker-compose.yml  containerized stack (backend, frontend, mongo, redis,
                        qdrant, mlflow) + arq worker
```

---

## Security & Production Engineering

- **Authentication:** JWT (HS256) short-lived access + HttpOnly refresh cookie;
  bcrypt; Redis blacklist revocation; refresh rotation.
- **Authorization (RBAC):** `require_roles(["admin"])` gates admin user
  management, `POST /api/v1/documents/upload`, `POST /api/v1/ingest`,
  `DELETE /api/v1/db/clear`, `DELETE /api/v1/cache`.
- **Rate limiting:** slowapi, dynamic key (`user:<id>` vs `ip:<host>`), structured 429.
- **Audit & observability:** structured JSON logs; per-request Mongo
  `rag_audit_logs` (query, chunks, prompts, response, latency, guardrail, cache,
  model); request-id echo; analytics over audit/`raw_chunks`.
- **Resilience:** guarded fallbacks across LLM, reranker, cache, MLflow, and
  store outages.

---

## Known Limitations (verified)

Full list in **[PROJECT_MAP §26](PROJECT_MAP.md#26-known-limitations-verified)**:

1. Guardrail is **self-consistency only** — it verifies answer numbers against
   the model's own `extracted_raw_data`; it does not re-query raw document text
   and performs no arithmetic.
2. Streaming is word-granular (provider JSON coalescing), not true token-level.
3. Session memory is RAM-only and not concurrency-safe.
4. Re-ingestion is idempotent per `chunk_id` but not convergent (stale ids never
   deleted).
5. Pre-retrieval intent routing is disabled (`ENABLE_PRE_RETRIEVAL=False`).
6. One test fails in the container image due to a `ragas`/`langchain-community`
   dependency mismatch (passes on host) — the documented image limitation.
7. MongoDB/Qdrant parity reports drift (see `/health.mongo_qdrant_parity`).

---

## Documentation

- **[PROJECT_MAP.md](PROJECT_MAP.md)** — authoritative, code-verified architecture reference.
- **[docs/CUTOVER_REPORT.md](docs/CUTOVER_REPORT.md)** — cutover / final acceptance report (D1–D7, C1–C7).
- **[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)** — live demo scenarios (Tests 1–6).
- **[test_artifacts/TEST_RUN_REPORT.md](test_artifacts/TEST_RUN_REPORT.md)** — live test evidence (662 passed / 1 failed).
- **[artifacts/EVALUATION_SUMMARY.md](artifacts/EVALUATION_SUMMARY.md)** — evaluation scores (F 0.990 / R 0.920 / P 0.845 / C 0.847).
- **[artifacts/ab2_benchmark/fp16_impl_report.md](artifacts/ab2_benchmark/fp16_impl_report.md)** — FP16 reranker report.

---

## Author

**Youssef** — AI Engineering — RAG Systems — Data Science

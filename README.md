# Financial Report Analysis RAG System

> **Production-oriented Retrieval-Augmented Generation (RAG) system for financial report analysis, built around SEC filings with structured ingestion, hybrid chunking, vector retrieval, dual storage, LLM generation, numerical guardrails, caching, observability, FastAPI, and Streamlit.**

---

## Overview

**Financial_RAG** is a production-oriented RAG system designed to answer questions over financial reports while preserving the structure and numerical integrity of financial data.

The system was built as an end-to-end pipeline rather than as a simple "retrieve chunks and ask an LLM" implementation. It separates the offline ingestion lifecycle from the online query lifecycle and introduces dedicated components for:

- SEC filing parsing and HTML table handling
- Financial-text cleaning with preservation of financial notation
- Metadata extraction and contextual tagging
- Three-tier token-bounded chunking
- Embedding generation with `nomic-ai/nomic-embed-text-v1.5`
- Qdrant vector retrieval with metadata pre-filtering
- MongoDB document storage and text enrichment
- Groq-hosted LLM generation with model fallback
- Strict Pydantic output validation
- Numerical post-generation verification
- Redis exact-match query caching
- Structured JSON logging and MLflow tracking
- FastAPI production backend
- Multi-format document upload
- Streamlit financial dashboard
- Unit, integration, and end-to-end testing

The result is a complete RAG application architecture with ingestion, retrieval, generation, validation, API serving, UI, testing, and observability.

---

## System Goals

The system is designed around several practical requirements of financial-document QA:

1. **Preserve financial structure**
   - Financial tables are isolated and converted to Markdown.
   - Financial notation such as `(150)`, `$`, `%`, and `M/B/K` is preserved during cleaning.

2. **Retrieve with contextual constraints**
   - Vector retrieval is performed through Qdrant.
   - Metadata filters such as `ticker` and `fiscal_year` are applied before retrieval.

3. **Separate retrieval storage from source-of-truth storage**
   - Qdrant stores vectors and retrieval metadata.
   - MongoDB stores raw text and complete metadata used for downstream enrichment and verification.

4. **Constrain LLM output**
   - Responses are validated against strict Pydantic schemas.
   - Generation uses structured JSON output.

5. **Verify financial claims**
   - A post-generation guardrail checks numerical claims against raw MongoDB data before a response is accepted.

6. **Avoid unsafe cache pollution**
   - Only validated successful responses are cached.
   - Invalid, fallback, and `"not available"` responses are not written to Redis.

7. **Provide production-style interfaces**
   - FastAPI exposes the RAG backend.
   - Streamlit provides the interactive financial dashboard.
   - Health checks, structured logging, metrics, and background tasks are included.

---

# Architecture

The system is divided into two primary execution phases:

```text
                    FINANCIAL_RAG
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
     OFFLINE INGESTION          ONLINE RAG PIPELINE
            │                         │
            ▼                         ▼
      Parse & Clean              Query Processing
            │                         │
            ▼                         ▼
      Metadata Tagging             Redis Cache
            │                         │
            ▼                         ▼
       Hybrid Chunking          Qdrant Retrieval
            │                         │
            ▼                         ▼
       Dual Indexing             Mongo Enrichment
            │                         │
      ┌─────┴─────┐                   ▼
      │           │              LLM Generation
   MongoDB     Qdrant                 │
      │           │                   ▼
      │           │              Guardrail
      │           │                   │
      └───────────┴───────────────────┘
                                      │
                                      ▼
                          Validated Financial Answer
```

---

## 1. Offline Ingestion Pipeline

The ingestion pipeline transforms raw financial filings into searchable and verifiable representations.

```text
Raw SEC Filing (HTML/TXT)
        │
        ▼
┌──────────────────────────────┐
│ html_table_parser.py         │
│ Table Detection              │
│ HTML → Markdown              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ metadata_extractor.py        │
│ Contextual Metadata          │
│ ticker / year / section      │
│ table flag / chunk UUID      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ hybrid_chunker.py            │
│ 3-Tier Chunking              │
│ Section → Table → Recursive  │
│ 512–768 tokens / 10–15%      │
│ overlap                      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ database_indexer.py          │
│ Dual Storage                 │
│                              │
│ MongoDB     +     Qdrant     │
│ raw/meta          vectors    │
└──────────────────────────────┘
```

### Parsing and table handling

SEC filings can contain complex HTML structures, particularly financial tables.

The ingestion layer uses:

- **BeautifulSoup4**
- **lxml**
- **Pandas**
- **Tabulate**

Tables are detected and converted into a clean Markdown representation using `to_markdown()`.

This allows table content to remain available as structured context instead of treating financial tables as ordinary unstructured text.

### Financial text cleaning

The cleaning stage intentionally preserves financial notation, including:

- Parenthetical negative values such as `(150)`
- Currency symbols such as `$`
- Percentages such as `%`
- Financial magnitude notation such as `M`, `B`, and `K`

This is important because altering these representations can change the meaning of financial information.

### Metadata extraction

Each chunk receives contextual metadata including:

- `ticker`
- `fiscal_year`
- `section`
- `doc_type`
- `contains_table`
- `chunk_id`

This metadata is later used during retrieval and downstream processing.

---

# 2. Hybrid Chunking Strategy

The system uses a **three-tier chunking strategy** instead of applying a single recursive splitter to the entire document.

```text
Document
   │
   ▼
1. Section-Level Splitting
   │
   ▼
2. Table Isolation
   │
   ├── Text
   │
   └── Complete Table
   │
   ▼
3. Recursive Token-Bounded Chunking
```

### Tier 1 — Section-level splitting

The document is first divided according to its structural sections.

This preserves higher-level financial context.

### Tier 2 — Table isolation

Financial tables are isolated and kept whole.

This is a deliberate design decision because splitting a financial table into arbitrary chunks can separate headers, rows, and numerical values.

### Tier 3 — Recursive token-bounded splitting

Text is recursively divided using token-aware boundaries:

- **512–768 tokens**
- **10–15% overlap**
- Nomic-compatible tokenization

The implementation also accounts for the fact that financial tables can exceed the normal token constraint when they are treated as atomic table chunks.

---

# 3. Dual Storage Architecture

The system intentionally separates vector retrieval storage from source-of-truth document storage.

```text
                     Indexed Chunk
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          MongoDB                 Qdrant
              │                     │
       Raw text + metadata     768-dim vectors
              │                     │
              │                HNSW retrieval
              │                     │
              └──────────┬──────────┘
                         ▼
                 Retrieved Context
```

### MongoDB

MongoDB stores:

- Raw text
- Markdown table representations
- Full metadata
- Chunk information

MongoDB is also used to retrieve the actual text after vector retrieval.

### Qdrant

Qdrant stores:

- 768-dimensional embeddings
- Retrieval metadata
- HNSW vector index

The vector store is optimized for approximate nearest-neighbor retrieval.

### Why two stores?

The architecture separates two responsibilities:

| Responsibility | Storage |
|---|---|
| Raw document text | MongoDB |
| Full metadata | MongoDB |
| Source-of-truth verification | MongoDB |
| Vector similarity search | Qdrant |
| HNSW ANN index | Qdrant |
| Metadata pre-filtering | Qdrant |

An important implementation detail is that Qdrant does **not** contain `raw_text`. After retrieval, chunk IDs are used to fetch the corresponding text from MongoDB.

---

# 4. Online Retrieval & Generation Pipeline

```text
User Query
    │
    ▼
Query Normalization
    │
    ▼
Redis Exact-Match Cache
    │
    ├── HIT ───────────────► Cached Response
    │
    └── MISS
          │
          ▼
    Qdrant Vector Search
          │
          ▼
    Metadata Pre-Filtering
    ticker / fiscal_year / section
          │
          ▼
    MongoDB Text Enrichment
          │
          ▼
    Groq LLM Generation
          │
          ▼
    Pydantic Validation
          │
          ▼
    Async Numerical Guardrail
          │
          ├── PASS ────────► Redis Cache Write
          │
          └── FAIL ────────► Safe Fallback
          │
          ▼
    ConsolidatedFinancialAnswer
```

---

## Retrieval

The retrieval layer uses **Qdrant HNSW approximate nearest-neighbor search**.

Metadata can be applied before retrieval, including:

- `ticker`
- `fiscal_year`
- `section`

This provides a way to constrain retrieval to the relevant financial context.

### Retrieval + enrichment

Retrieval is implemented as a two-phase process:

1. Retrieve relevant vector matches from Qdrant.
2. Fetch the corresponding text from MongoDB using the retrieved chunk IDs.

This design exists because Qdrant stores the vector representation and metadata, while MongoDB remains the source for the actual raw text.

---

# 5. LLM Generation Layer

The generation engine is built around Groq-hosted LLMs.

### Primary model

```text
llama-3.3-70b-versatile
```

Configuration:

- Temperature: `0.0`
- Seed: `42`

### Fallback model

```text
qwen/qwen3.6-27b
```

If the primary model fails, the system automatically falls back to the secondary model.

The project also includes robust handling for model output variations, including stripping Qwen thinking tags before JSON extraction.

### Generation features

The generation engine includes:

- Primary/fallback model strategy
- Exponential-backoff retry
- Up to 3 attempts
- XML `<CONTEXT>` enclosure
- Sliding conversation memory
- `K=6` history window
- True asynchronous token streaming
- Strict JSON output
- Raw-text fallback when JSON parsing fails
- MLflow generation metrics

---

# 6. Structured Output with Pydantic

The generation layer does not simply return unstructured LLM text.

Responses are validated through Pydantic v2 schemas.

Core schemas include:

- `ConsolidatedFinancialAnswer`
- `GuardrailVerdict`
- `CacheEntry`

The consolidated answer contains structured fields such as:

- Internal thought
- Extracted raw data
- Final answer
- Sources

Field validation is applied before the result proceeds through the pipeline.

---

# 7. Financial Guardrails

Financial QA requires special attention to numerical claims.

The project therefore implements an asynchronous post-generation verification loop.

```text
LLM Response
     │
     ▼
Extract Numerical Claims
     │
     ▼
Compare Against MongoDB Raw Data
     │
     ├── Match ─────► PASS
     │
     └── Mismatch ──► Safe Fallback
```

The guardrail performs numerical verification using a regex-based mechanism rather than making another LLM call.

This creates an additional verification layer between generation and the final user-facing response.

### Guardrail behavior

The guardrail can:

- Validate numerical claims
- Reject inconsistent generated answers
- Trigger safe fallback behavior
- Control whether a response is written to Redis
- Log guardrail metrics to MLflow

---

# 8. Redis Caching

The project includes Redis-backed query caching.

Despite the internal `SemanticCache` class name, the implemented cache is an **exact-match cache**, not a vector-semantic similarity cache.

### Cache key

The query is normalized and hashed with SHA-256:

```text
rag_cache:{sha256_hash}
```

### Cache lifecycle

```text
Query
  │
  ▼
Normalize
  │
  ▼
SHA-256
  │
  ▼
Redis Lookup
  │
  ├── HIT ─────► Return Cached Response
  │
  └── MISS
        │
        ▼
     RAG Pipeline
        │
        ▼
     Guardrail
        │
        ├── PASS ──► Write to Redis
        │
        └── FAIL ──► Do Not Cache
```

### Cache pollution prevention

The system does **not** cache:

- Invalid responses
- Fallback responses
- `"not available"` responses
- Responses that fail guardrail verification

The cache can also be flushed through:

```http
DELETE /api/v1/cache
```

---

# 9. Conversation Memory

The generation engine maintains a sliding conversation history.

Configuration:

```text
K = 6 messages
```

This provides short-term conversational context without allowing the history to grow indefinitely.

---

# 10. Observability & MLflow

The project uses structured JSON logging and MLflow tracking.

### Structured logging

All major modules emit JSON logs to:

```text
logs/rag_events.log
```

The logging format is designed to remain compatible with external log ingestion systems such as ELK or Datadog.

### MLflow

MLflow tracks generation and pipeline metrics including:

- Model parameters
- Temperature
- Maximum tokens
- Seed
- Conversation history size
- Time to first token
- Total generation tokens
- Guardrail result
- Fallback activation
- End-to-end latency
- Retrieval count
- Cache hit/miss

---

# 11. Production FastAPI Backend

The system exposes the RAG pipeline through FastAPI.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Main RAG query |
| `POST` | `/api/v1/chat/stream` | Streaming RAG response |
| `POST` | `/api/v1/documents/upload` | Multi-format document upload |
| `DELETE` | `/api/v1/cache` | Flush Redis cache |
| `GET` | `/health` | Service health check |

### `/api/v1/chat`

Returns a structured `ChatQueryResponse` containing the answer, sources, and metrics.

### `/api/v1/chat/stream`

Provides SSE-based streaming using the asynchronous generation path.

### `/api/v1/documents/upload`

Accepts supported document formats and queues ingestion through FastAPI background tasks.

### `/health`

Checks connectivity to:

- MongoDB
- Qdrant
- Redis

---

# 12. Multi-Format Document Ingestion

The API layer extends the original SEC ingestion pipeline with multiple file formats.

Supported formats:

```text
HTML / HTM / TXT / SGML
PDF
DOCX
```

### Parser routing

| Extension | Primary Parser | Fallback |
|---|---|---|
| `.html` / `.htm` / `.txt` / `.sgml` | Existing SEC parser | — |
| `.pdf` | LlamaParse | pypdf |
| `.docx` | LlamaParse | python-docx |

The parser layer is implemented through `APIFileParser`.

For PDF and DOCX documents, LlamaParse is used as the primary parser with local extraction fallbacks.

---

# 13. Streamlit Financial Dashboard

The project includes an interactive Streamlit dashboard connected to the FastAPI backend.

```text
                 Streamlit UI
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   Health Monitor   Upload       Chat
                                      │
                              Real-Time Streaming
                                      │
                                      ▼
                                  Sources
                                      │
                                      ▼
                              Performance Metrics
```

## Dashboard capabilities

### Health Monitor

Displays service status for:

- MongoDB
- Qdrant
- Redis

### Document Upload

Allows users to:

- Upload supported files
- Provide ticker
- Provide fiscal year
- Trigger document ingestion
- View resulting chunk counts

### Financial Chat

The dashboard provides:

- Conversational history
- Ticker filtering
- Fiscal-year filtering
- Real-time token streaming
- Source inspection
- Performance information

### Sources

Each assistant response can expose:

- Ticker
- Fiscal year
- Section
- Retrieval score
- Text snippet

### Performance footer

Each response displays:

- Execution time
- Model used
- Cache HIT/MISS

---

# 14. Project Structure

```text
Financial_RAG/
│
├── data/
│   ├── AAPL/10-K/
│   ├── MSFT/10-K/
│   └── NVDA/10-K/
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging_config.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── 1_ingestion/
│   │   ├── __init__.py
│   │   ├── html_table_parser.py
│   │   ├── cleaning.py
│   │   ├── metadata_extractor.py
│   │   ├── hybrid_chunker.py
│   │   └── database_indexer.py
│   │
│   ├── 2_generation/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── generator.py
│   │   └── async_guardrail.py
│   │
│   └── pipeline.py
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── main.py
│   │   ├── worker.py
│   │   └── parsers.py
│   │
│   └── ui/
│       └── streamlit_app.py
│
├── test/
│   ├── __init__.py
│   ├── test_ingestion_stage1.py
│   ├── test_ingestion_stage2.py
│   ├── test_ingestion_e2e_integration.py
│   ├── test_generation_stage.py
│   ├── test_generation_e2e_integration.py
│   ├── test_pipeline_orchestrator.py
│   ├── test_api_endpoints.py
│   ├── test_api_upload_parser.py
│   └── test_streamlit_ui.py
│
├── logs/
│   └── rag_events.log
│
├── .env
├── .gitignore
├── requirements.txt
├── PROJECT_MAP.md
└── README.md
```

---

# 15. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| HTML Parsing | BeautifulSoup4 + lxml |
| Table Processing | Pandas + Tabulate |
| Tokenization | tiktoken |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` |
| Vector Store | Qdrant |
| Document Store | MongoDB |
| Cache / Memory | Redis |
| Primary LLM | `llama-3.3-70b-versatile` |
| Fallback LLM | `qwen/qwen3.6-27b` |
| LLM Provider | Groq |
| Orchestration | LangChain + LangChain-Groq |
| Output Validation | Pydantic v2 |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Observability | Structured JSON Logging + MLflow |
| Secrets | python-dotenv |

---

# 16. Configuration

The system uses environment-based configuration through `.env`.

The configuration layer centralizes:

- API keys
- Database URIs
- Redis configuration
- Paths
- System constants
- LlamaParse configuration

The project requires the relevant credentials for the configured services, including the Groq API and, where applicable, LlamaParse.

---

# 17. Testing & Verification

Testing was treated as a first-class part of the project.

The project includes:

- Unit tests
- Module integration tests
- End-to-end ingestion tests
- Live LLM integration tests
- API tests
- Upload parser tests
- Streamlit UI client tests

## Test status

| Component | Tests | Status |
|---|---:|---|
| Ingestion Stage 1 | 12/12 | ✅ |
| Ingestion Stage 2 | 12/12 | ✅ |
| Ingestion E2E | 8/8 | ✅ |
| Generation Engine | 29/29 | ✅ |
| Generation E2E | 15/15 | ✅ |
| Pipeline Orchestrator | 24/24 | ✅ |
| FastAPI Backend | 31/31 | ✅ |
| Upload Parser | 23/23 | ✅ |
| Streamlit UI | 17/17 | ✅ |

### Overall result

```text
148/148 unit tests passing
+ E2E integration verification
+ 17/17 Streamlit UI tests
```

The project map records the system as fully implemented, with deployment remaining as the next stage.

---

# 18. End-to-End Verification

A real AAPL 10-K filing was processed through the complete ingestion pipeline.

### Ingestion results

- Filing size: **8.96 MB**
- Total chunks: **110**
- Text chunks: **56**
- Table chunks: **54**
- Average text chunk size: **732 tokens**
- Maximum text chunk size: **768 tokens**
- Embeddings: **110 vectors**
- Embedding runtime: approximately **42 seconds on CUDA**
- GPU: **NVIDIA GeForce RTX 4050 Laptop GPU**
- MongoDB: **110/110 chunks stored**
- Qdrant: **110/110 vectors stored**
- Best filtered retrieval score: **0.7913**
- Structured log entries: **1144**

### Generation E2E

The complete generation integration was also verified using a real AAPL 10-K and live Groq API.

The integration covered:

1. Full ingestion
2. Answerable financial questions
3. Unanswerable questions
4. Guardrail execution
5. MLflow tracking

The system correctly returned `"not available"` for an unanswerable exact-revenue query rather than inventing a figure.

---

# 19. Important Engineering Decisions

## Dual storage

MongoDB is treated as the raw-data source while Qdrant is responsible for vector retrieval.

This allows the retrieval layer and verification layer to remain separated.

## Table atomicity

Tables are isolated and preserved as complete chunks.

This avoids breaking financial rows and headers across arbitrary chunk boundaries.

## Metadata-aware retrieval

Ticker and fiscal year can be used as retrieval filters.

This helps constrain vector search to the intended financial context.

## Model fallback

The generation engine does not depend on a single LLM endpoint.

A fallback model is activated when the primary model fails.

## Structured output

LLM responses are constrained to validated Pydantic schemas rather than being passed directly as arbitrary strings.

## Numerical guardrails

Financial numbers are checked after generation against the stored source data.

## Exact-match caching

Redis caching is intentionally based on normalized-query hashing rather than vector similarity.

## Cache safety

Only guardrail-approved responses are cached, preventing invalid responses from becoming persistent cached results.

## Graceful degradation

The pipeline includes safe fallback behavior for:

- Empty retrieval
- LLM failures
- JSON parsing failures
- Guardrail failures

---

# 20. Engineering Issues Solved During Development

Several implementation issues were discovered and fixed during end-to-end integration.

### Chunk ID overwrite

`upsert_chunks` originally allowed metadata expansion to overwrite `chunk_id`.

The order of the metadata expansion was corrected.

### CUDA execution

The embedding engine was changed from automatic device detection to CUDA-forced execution with a CUDA availability assertion.

### GPU memory

Embedding batch size was reduced from `64` to `32` for the 6 GB VRAM environment, with explicit CUDA cache cleanup.

### Table token limits

Tables were exempted from the normal 768-token constraint so that financial tables could remain atomic.

### Qdrant cleanup

The Qdrant client is closed before test-directory cleanup to avoid `.lock` permission errors.

### Missing guardrail helper

`async_guardrail.py` depended on `_hash_query()` from another module.

A local implementation was added to prevent runtime failure when Redis was available.

### Qdrant text assumption

The implementation identified that Qdrant payloads did not contain `raw_text`.

The retrieval pipeline therefore explicitly fetches the corresponding text from MongoDB.

### Table placeholders

Text chunks can contain `%%TABLE_N%%` placeholders while the actual financial figures exist in separate table chunks.

The integration verified that the system does not invent exact values when the required table context is absent.

### Model availability

The primary Groq model was updated after the previous `qwen-2.5-72b-instruct` model became unavailable.

The current primary is:

```text
llama-3.3-70b-versatile
```

with:

```text
qwen/qwen3.6-27b
```

as fallback.

---

# 21. Module Status

| Module | Status |
|---|---|
| Module 1 — Ingestion Pipeline | ✅ COMPLETE |
| Module 2 — Generation Engine | ✅ COMPLETE |
| Module 1+2 E2E Integration | ✅ VERIFIED |
| Module 3a — Pipeline Orchestrator | ✅ COMPLETE |
| Module 3b — FastAPI Backend | ✅ COMPLETE |
| Module 3c — Streamlit UI | ✅ COMPLETE |
| Module 3d — Deployment | ⏳ PENDING |

---

# 22. Current System Flow

The complete production-oriented flow can be summarized as:

```text
                         ┌─────────────────────┐
                         │     SEC Filing      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Parse & Clean       │
                         │ HTML / Tables       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Metadata Extraction │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Hybrid Chunking     │
                         │ Section + Tables    │
                         │ + Token Boundaries  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  ┌─────────────┐       ┌─────────────┐
                  │  MongoDB    │       │   Qdrant    │
                  │ Raw + Meta  │       │  Vectors    │
                  └──────┬──────┘       └──────┬──────┘
                         │                     │
                         │              ┌──────▼──────┐
                         │              │ Vector      │
                         │              │ Retrieval   │
                         │              └──────┬──────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Context Assembly    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Groq LLM Generation │
                         │ Primary + Fallback  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Pydantic Validation │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Numerical Guardrail │
                         └──────────┬──────────┘
                                    │
                           ┌────────┴────────┐
                           ▼                 ▼
                         PASS              FAIL
                           │                 │
                           ▼                 ▼
                    Redis Cache       Safe Fallback
                           │
                           ▼
                 ConsolidatedFinancialAnswer
```

---

# 23. What Makes This More Than a Basic RAG

This project intentionally goes beyond the minimal RAG pattern:

```text
Basic RAG
---------
Documents
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Search
   ↓
LLM
```

The implemented system adds multiple engineering layers around that core:

```text
Financial RAG
──────────────────────────────────────────────
Document Parsing
      ↓
Financial Cleaning
      ↓
Metadata Extraction
      ↓
Structure-Aware Hybrid Chunking
      ↓
Dual Storage
      ↓
Metadata-Aware Vector Retrieval
      ↓
MongoDB Context Enrichment
      ↓
Conversation Memory
      ↓
LLM Fallback + Retry
      ↓
Strict Structured Output
      ↓
Numerical Guardrail
      ↓
Cache Validation
      ↓
Redis Exact-Match Cache
      ↓
MLflow + Structured Observability
      ↓
FastAPI
      ↓
Streaming
      ↓
Streamlit Dashboard
      ↓
Automated Testing
```

The project therefore focuses not only on retrieval quality, but also on **data integrity, failure handling, observability, validation, API serving, caching, testing, and operational behavior**.

---

# 24. Project Maturity

The project currently reaches a complete implemented state across:

- Ingestion
- Chunking
- Embedding
- Indexing
- Retrieval
- Generation
- Guardrails
- Caching
- Observability
- API
- UI
- Testing

The remaining project-map stage is:

```text
Deployment → PENDING
```

---

## Status

```text
Project: Financial_RAG
Implementation: COMPLETE
Unit Tests: 148/148 PASSING
Streamlit UI Tests: 17/17 PASSING
E2E Ingestion: VERIFIED
E2E Generation: VERIFIED
Deployment: NEXT
```

---

## License

No license information is defined in the project map.

# Multi-Hop / Complex Context-Fusion Retrieval — Feasibility Study & Roadmap

**Status:** Proposed / Not implemented (frozen for live-demo stability)
**Owner:** RAG & LLM Systems Architecture
**Date:** 2026-08-26
**Scope:** Extend FinSight AI beyond single-pass retrieval to support iterative,
multi-hop queries whose later sub-goals depend on intermediates extracted from
earlier sub-goals (e.g. cross-section / cross-filing reasoning).

---

## 0. Verdict

**Feasible, low-risk, additive.** Roughly 80% of the required machinery already
exists in the pipeline:

- `_retrieve_documents_multi` (`src/pipeline.py:438`) — multi-query retrieval
  merged per `chunk_id`, keeping the best score.
- `_balanced_ticker_subretrievals_sync` (`src/pipeline.py:776`) — per-scope
  sub-retrievals (cross-entity parallelism).
- `_augment_context` (`src/pipeline.py:921`) — MongoDB table fusion by detected
  ticker / fiscal year.
- `CrossEncoderReranker.rerank(query, chunks)` (`src/4_retrieval/reranker.py:95`)
  — re-ranks (query, chunk) pairs; already per-call, so it re-runs cleanly per hop.
- `_mongo_indexer.get_chunks_by_filter({"ticker", "fiscal_year"})`
  (`src/pipeline.py:1030`) — pulls statement/risk chunks directly.

Multi-hop is therefore a **new orchestration layer** that loops those primitives.
Ship it behind an opt-in `enable_multi_hop=False` flag (same pattern as
`enable_pre_retrieval` / `enable_post_retrieval` at `src/pipeline.py:259-261`) so
the current single-pass path is byte-for-byte unchanged.

---

## 1. Current Architecture vs. Multi-Hop Bottlenecks

### 1.1 The single-pass shape today

`FinancialRAGPipeline.query()` (`src/pipeline.py:1281`) executes strictly:

```
retrieve  ->  _resolve_table_placeholders + _augment_context   (pipeline.py:1571-1572)
         ->  format_context_xml(documents)                      (pipeline.py:1579)
         ->  self._generator.generate(...)                      (pipeline.py:1582)
         ->  AsyncGuardrail + SemanticCache + MLflow            (pipeline.py:1591-1671)
```

The FastAPI layer (`app/api/main.py`) never sees hops: `/chat` (`main.py:490`)
calls `pipe.query(...)` and reads `result["parsed"]` / `retrieved_chunks`.

### 1.2 Why a single pass misses / degrades on the target example

> *"Find the segment that generated the highest revenue growth for Microsoft in
> FY2025, and then check what risk factors are listed in Item 1A for that specific
> segment."*

- **The dependent variable is unknown at retrieval time.** Which segment wins is
  only learned *after* Hop 1 generation. A single pass must retrieve Item 1A risk
  context *before* knowing the segment, so it either (a) pulls generic Item 1A
  narrative (not segment-scoped) and the LLM infers segment-specific risk →
  confabulation, or (b) ignores it → weak / missing citation.
- `_augment_context` injects Segment Results tables (`_SEGMENT_QUERY_TERMS`,
  `pipeline.py:1165`) but **never** Item 1A risk text, and injects only by
  detected ticker / fiscal year — not by an intermediate segment name.
- **Reranker scores against one fixed query** (`reranker.py:95`). The dependent
  sub-goal *"Item 1A risks for Intelligent Cloud"* never receives its own
  (query, chunk) cross-attention pass.
- **Context budget is allocated once.** `MAX_TOTAL_CONTEXT_CHARS = 12000`
  (`pipeline.py:99`) is consumed by Hop 1 tables; no second allocation exists for
  the dependent evidence.

### 1.3 What the Qdrant / MongoDB setup needs (and lacks)

| Capability | Today | Gap for multi-hop |
|---|---|---|
| Metadata pre-filter `{ticker, fiscal_year, section}` | ✅ Qdrant | Reuse as per-hop **guardrail** |
| MongoDB table/risk pull by ticker+fy | ✅ (`pipeline.py:1030`) | Reuse to pull Item 1A risk for a segment |
| `segment` metadata field on chunks | ❌ | No chunk is tagged by segment → Hop 2 must target via an **LLM/keyword-extracted intermediate** + BM25 (HybridSearchEngine makes this viable) |
| Qdrant payload carries `raw_text` | ❌ (Mongo only) | Every hop = extra Mongo round-trip → needs per-request hop cache |
| Per-query rerank | ✅ | Reuse per hop |

**Conclusion:** the immediate design uses an LLM-extracted intermediate segment
name fed into a BM25 + dense sub-query. A `segment` ingestion tag is a deferred
Phase-2 enhancement (see §4.4), not a blocker.

---

## 2. Proposed Strategy — Sub-Goal Decomposition + Iterative Hopping

A **hybrid** design (not free-form ReAct — too slow / latency-risky for a live
demo):

1. **Planner** decomposes the query into an ordered list of `Hop`s (hard cap = 3).
   Decomposition is **rule-based first, LLM-assisted fallback** — mirroring the
   `QueryExpander` philosophy (`src/3_pre_retrieval/query_expansion.py:15`):
   deterministic, zero hallucination, unit-testable.
2. Each `Hop = (sub_query, metadata_filter, depends_on?)`.
3. **Execute hops sequentially** (Hop 2 depends on Hop 1's extracted intermediate).
   Independent hops *could* run in parallel, but the target scenario is inherently
   sequential, so keep it simple.

### 2.1 Iterative flow (worked example)

```
TURN 1  Retrieve: "MSFT FY2025 segment revenue growth"
         filter = {ticker: MSFT, fiscal_year: 2025, section: segment}
         Rerank -> fuse (income + segment tables) -> GENERATE(extract-mode)
         => intermediate: segment = "Intelligent Cloud", d_growth = +X%

TURN 2  sub_query  = "Item 1A risk factors Intelligent Cloud Microsoft FY2025"
         filter     = {ticker: MSFT, fiscal_year: 2025, section: "Item 1A"}
         Retrieve (hybrid BM25 + dense) -> Rerank(sub_query, chunks)
         => Evidence: Item 1A passages mentioning Intelligent Cloud

SYNTHESIS  Merge Hop1 + Hop2 evidence (dedup by chunk_id, global token budget)
           format_context_xml(merged) -> GENERATE(answer-mode) -> guardrail -> cache
```

This reuses `_retrieve_documents_multi` (merge-by-`chunk_id`) and
`CrossEncoderReranker.rerank` per hop.

---

## 3. Performance & Latency Mitigation + Guardrails

| Concern | Mitigation (grounded in existing code) |
|---|---|
| Hop-count blow-up | Hard cap `MAX_HOPS = 3`. Planner stops when no `depends_on` remains or output is "answerable". |
| Token / TPM (Groq primary = 8000) | Reuse `MAX_TOTAL_CONTEXT_CHARS = 12000` **globally** across hops; per-hop slice = `budget / MAX_HOPS`. Hop 1 uses *extract-mode* (small prompt returning only the intermediate). |
| Re-embedding cost | Batch-embed all hop sub-queries in **one GPU pass** via `self._embedding_engine.embed(queries)` (as `QueryExpander` does, `query_expansion.py:120`). |
| Mongo round-trips | Per-request `_hop_cache: dict[(sub_query, guardrail)] -> chunks`; Redis `SemanticCache` keyed on sub-query catches repeats. |
| Latency estimate | ~1 embed-batch + 2–3 retrievals + 1 cheap extract + 1 full gen ≈ **1.8–2.5×** single-pass. Surface a "thinking: hop N" indicator in the UI. |
| Streaming | Multi-hop cannot token-stream the final answer live (all hops precede synthesis). Emit a lightweight SSE event `{"hop": N, "status": "retrieving"}` per hop, then stream the final synthesis via the existing `EventSourceResponse` (`main.py:577-731`). |
| Metadata bleed | **Immutable guardrail** locked from the request: `base = {ticker(s), fiscal_year}` (or `ALL` scope). Each hop filter is `intersect(base, hop_filter)` — a hop may *narrow* (add `section="Item 1A"`) but never *widen* to another ticker / year. Reuse `_apply_cross_ticker_bypass` (`pipeline.py:732`) so comparison intent still expands safely. |
| Non-breaking | `enable_multi_hop = False` default; `query()` dispatches only when planner detects multi-hop intent. |

---

## 4. Implementation Plan & Pseudocode

### 4.1 New module — `src/4_retrieval/multi_hop.py`

```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class Hop:
    index: int
    sub_query: str
    metadata_filter: dict                 # {ticker, fiscal_year, section?}
    depends_on: Optional[int] = None      # prior hop that yields the intermediate
    mode: str = "evidence"                # "evidence" | "extract" | "synthesize"

class MultiHopPlanner:
    MAX_HOPS = 3
    # deterministic triggers: "then check", "for that segment", "based on",
    # "after finding", "what about <X> for that …"
    _MH_RE = re.compile(
        r"\b(then|after (?:finding|that)|for that|based on (?:the|that)|"
        r"subsequently|next,? check)\b", re.I
    )

    def detect(self, q: str) -> bool:
        return bool(self._MH_RE.search(q)) or self._has_dependency_pronoun(q)

    def decompose(self, q, guardrail) -> list[Hop]:
        # Rule-based: split on connectors; first clause = hop0 (extract),
        # later clause(s) = hopN (evidence) scoped by guardrail.
        # LLM fallback only if clause count > MAX_HOPS or ambiguous.
        ...
        return hops  # each hop.metadata_filter already intersected with guardrail
```

### 4.2 Pipeline integration — `src/pipeline.py`

Add to `__init__`:

```python
self._multi_hop = MultiHopPlanner() if enable_multi_hop else None
```

New dispatcher inside `query()` (after the pre-retrieval block, ~`pipeline.py:1451`):

```python
if self._enable_multi_hop and self._multi_hop is not None \
        and self._multi_hop.detect(user_query):
    return self._run_multi_hop(user_query, ticker, fiscal_year, tickers, top_k)
# ... existing single-pass flow unchanged below ...
```

New method (reuses existing primitives verbatim):

```python
def _run_multi_hop(self, query, ticker, fiscal_year, tickers, top_k):
    guardrail = self._build_guardrail(query, ticker, fiscal_year, tickers)  # immutable {ticker(s), fy}
    hops = self._multi_hop.decompose(query, guardrail)
    evidence: list[dict] = []
    seen: set[str] = set()
    intermediate = None

    for hop in hops:
        filt = self._intersect_guardrail(guardrail, hop.metadata_filter)
        sub_q = hop.sub_query
        if hop.depends_on is not None and intermediate:
            sub_q = f"{intermediate} {sub_q}"          # "Intelligent Cloud Item 1A risk …"
        if self._enable_hybrid_retrieval:
            docs = self._run_hybrid_retrieval(sub_q, [sub_q], filt)
        else:
            docs = self._retrieve_documents_multi([sub_q], [], filt, top_k)
        docs = self._reranker.rerank(sub_q, docs, top_n=min(8, top_k * 2))
        docs = self._augment_context(docs, sub_q, all_companies=False)
        for d in docs:                                 # dedup by chunk_id (cf. _balanced_ticker_subretrievals)
            cid = str(d.get("chunk_id") or "")
            if cid and cid not in seen:
                seen.add(cid); evidence.append(d)
        if hop.mode == "extract":
            intermediate = self._generator.generate(
                query=sub_q, retrieved_docs=docs,
                stream=False, extract_only=True).get("intermediate")

    evidence = self._enforce_global_budget(evidence)   # cap at MAX_TOTAL_CONTEXT_CHARS
    self._last_contexts = list(evidence)
    context_xml = format_context_xml(evidence)
    result = self._generator.generate(query=query, retrieved_docs=evidence,
                                       stream=False, multi_ticker=guardrail.is_multi)
    # guardrail + cache + MLflow unchanged (pipeline.py:1591-1671)
    return result
```

### 4.3 Where it sits relative to `main.py` / `generator.py`

- **`main.py` `/chat` (`main.py:490`) & `/chat/stream` (`main.py:555`):** unchanged —
  still call `pipe.query(...)`. They read `result["parsed"]` / `retrieved_chunks`,
  which `_run_multi_hop` populates identically, so audit logging (`_build_sources`,
  `main.py:360`) and SSE need **zero changes**.
- **Generator (`src/5_generation/generator.py`):** add one keyword arg
  `extract_only: bool = False` to `generate()`. When true, prompt the LLM to return
  *only* the intermediate entity / metric as JSON (minimal context, cheap). Default
  `False` preserves today's behavior.
- **Reranker (`src/4_retrieval/reranker.py`):** no change — `rerank(query, chunks)`
  is already per-call; invoke once per hop with that hop's `sub_query`.

### 4.4 Optional Phase-2 (non-blocking, bigger win)

Add a `segment` tag during ingestion (`src/1_ingestion/hybrid_chunker.py` /
`metadata_extractor.py`) so Hop 2 becomes a clean Qdrant
`section="Item 1A", segment="Intelligent Cloud"` filter instead of BM25 keyword
matching. This is the only change that touches ingestion; defer until Phase 1
(LLM-extracted intermediate + BM25) proves out.

---

## 5. Recommended Next Step (post-demo)

1. Build Phase 1: `src/4_retrieval/multi_hop.py` + `query()` dispatcher +
   `generator.extract_only`, gated by `enable_multi_hop`.
2. Add one integration test using the exact example query against the live server,
   asserting the final answer cites **both** the segment-results chunk **and** an
   Item 1A risk chunk for the winning segment.
3. Benchmark latency vs. single-pass; tune `MAX_HOPS` / per-hop budget.

---

## Appendix A — Relevant existing code anchors

- `app/api/main.py:490` — `/chat` entry (calls `pipe.query`)
- `app/api/main.py:555` — `/chat/stream` (SSE)
- `app/api/main.py:360` — `_build_sources` (dedup by `chunk_id`)
- `src/pipeline.py:1281` — `query()` (single-pass orchestrator)
- `src/pipeline.py:438` — `_retrieve_documents_multi` (multi-query merge)
- `src/pipeline.py:776` — `_balanced_ticker_subretrievals_sync` (per-scope retrieve)
- `src/pipeline.py:921` — `_augment_context` (MongoDB table fusion)
- `src/pipeline.py:1030` — `get_chunks_by_filter` (direct Mongo pull)
- `src/pipeline.py:732` — `_apply_cross_ticker_bypass` (guardrail helper)
- `src/4_retrieval/reranker.py:95` — `CrossEncoderReranker.rerank`
- `src/3_pre_retrieval/query_expansion.py:120` — batch embed pattern
- `src/5_generation/generator.py` — `generate()` (add `extract_only`)

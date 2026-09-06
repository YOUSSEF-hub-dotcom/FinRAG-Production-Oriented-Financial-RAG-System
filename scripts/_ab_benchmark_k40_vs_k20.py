#!/usr/bin/env python3
"""
A/B Benchmark: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20

Controlled comparison of the production RAG pipeline with two candidate
pool sizes. Measures retrieval quality, reranker performance, answer
correctness, and end-to-end latency using the ACTUAL production code paths.

No production configuration was changed during this experiment.
"""

import gc
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Project setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/youssef/Financial_RAG")
os.chdir(PROJECT_ROOT)

# Ensure import paths
_root = str(PROJECT_ROOT)
_src = str(PROJECT_ROOT / "src")
for _p in (_root, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _sub in ("1_ingestion", "5_generation", "3_pre_retrieval", "4_retrieval",
             "2_caching"):
    _p = os.path.join(_src, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config.settings as settings
import pipeline as pipeline_mod
import hybrid_search as hs_mod


# ── Test Queries ───────────────────────────────────────────────────────
# Each query specifies the query text, expected tickers, expected numbers
# (substring matches), and the query category.

TEST_QUERIES = [
    # ── Category 1: Single-ticker factual ──
    {
        "id": "ST1",
        "query": "What was Apple total net sales in fiscal year 2025?",
        "ticker": "AAPL",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "single_ticker_factual",
        "expected_numbers": ["416161"],
        "expected_tickers": ["AAPL"],
        "ground_truth": "$416,161M",
    },
    {
        "id": "ST2",
        "query": "What was Microsoft total revenue in fiscal year 2025?",
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "single_ticker_factual",
        "expected_numbers": ["281724"],
        "expected_tickers": ["MSFT"],
        "ground_truth": "$281,724M",
    },
    {
        "id": "ST3",
        "query": "What was NVIDIA total revenue in fiscal year 2026?",
        "ticker": "NVDA",
        "fiscal_year": "2026",
        "tickers": None,
        "category": "single_ticker_factual",
        "expected_numbers": ["130497"],
        "expected_tickers": ["NVDA"],
        "ground_truth": "$130,497M",
    },
    # ── Category 2: Financial metrics ──
    {
        "id": "FM1",
        "query": "What was Apple operating income in fiscal year 2025?",
        "ticker": "AAPL",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "financial_metric",
        "expected_numbers": ["133050"],
        "expected_tickers": ["AAPL"],
        "ground_truth": "$133,050M",
    },
    {
        "id": "FM2",
        "query": "What was Apple gross margin percentage in fiscal year 2025?",
        "ticker": "AAPL",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "financial_metric",
        "expected_numbers": ["46.9"],
        "expected_tickers": ["AAPL"],
        "ground_truth": "46.9%",
    },
    {
        "id": "FM3",
        "query": "What was Microsoft net income in fiscal year 2025?",
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "financial_metric",
        "expected_numbers": ["101832"],
        "expected_tickers": ["MSFT"],
        "ground_truth": "$101,832M",
    },
    # ── Category 3: Multi-ticker comparison ──
    {
        "id": "MT1",
        "query": "Compare the revenue of Apple, Microsoft, and Nvidia.",
        "ticker": None,
        "fiscal_year": None,
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "category": "multi_ticker_comparison",
        "expected_numbers": ["416161", "281724", "130497"],
        "expected_tickers": ["AAPL", "MSFT", "NVDA"],
        "ground_truth": "AAPL $416,161M > MSFT $281,724M > NVDA $130,497M",
    },
    {
        "id": "MT2",
        "query": "Compare gross margins across Apple, Microsoft, and Nvidia.",
        "ticker": None,
        "fiscal_year": None,
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "category": "multi_ticker_comparison",
        "expected_numbers": [],
        "expected_tickers": ["AAPL", "MSFT", "NVDA"],
        "ground_truth": "Three gross margin percentages",
    },
    # ── Category 4: Table-heavy queries ──
    {
        "id": "TH1",
        "query": "What were Apple iPhone net sales and Services net sales in fiscal year 2025?",
        "ticker": "AAPL",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "table_heavy",
        "expected_numbers": ["209586", "109158"],
        "expected_tickers": ["AAPL"],
        "ground_truth": "iPhone $209,586M, Services $109,158M",
    },
    {
        "id": "TH2",
        "query": "What was Microsoft largest business segment by revenue in fiscal year 2025?",
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "table_heavy",
        "expected_numbers": ["120810"],
        "expected_tickers": ["MSFT"],
        "ground_truth": "Productivity & Business Processes $120,810M",
    },
    # ── Category 5: Negative / unsupported ──
    {
        "id": "NEG1",
        "query": "What was Tesla total revenue in fiscal year 2025?",
        "ticker": "TSLA",
        "fiscal_year": "2025",
        "tickers": None,
        "category": "negative",
        "expected_numbers": [],
        "expected_tickers": [],
        "ground_truth": "Not available",
    },
]


# ── Data Classes ───────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query_id: str
    category: str
    config_name: str
    # Retrieval
    num_candidates: int = 0
    num_candidates_per_ticker: dict = field(default_factory=dict)
    retrieval_ms: float = 0.0
    # Reranker
    reranker_latency_ms: float = 0.0
    top_8_scores: list = field(default_factory=list)
    top_8_tickers: list = field(default_factory=list)
    top_8_chunk_ids: list = field(default_factory=list)
    top_8_overlap_with_a: float = 0.0
    # Answer
    answer: str = ""
    numerical_correct: bool = False
    ticker_coverage: list = field(default_factory=list)
    ticker_coverage_complete: bool = False
    # Guardrail
    guardrail_passed: bool = False
    guardrail_detail: str = ""
    # E2E
    e2e_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    # Raw
    raw_result: dict = field(default_factory=dict)
    error: str = ""


# ── Helpers ────────────────────────────────────────────────────────────

def check_numerical_correctness(answer: str, expected_numbers: list[str]) -> bool:
    """Check if the answer contains the expected numerical values."""
    if not expected_numbers:
        return True  # No numbers to check
    answer_normalized = re.sub(r"[,\s]", "", answer.lower())
    for num in expected_numbers:
        num_normalized = re.sub(r"[,\s]", "", num)
        if num_normalized not in answer_normalized:
            return False
    return True


def check_ticker_coverage(answer: str, expected_tickers: list[str]) -> list[str]:
    """Check which expected tickers are mentioned in the answer."""
    covered = []
    answer_upper = answer.upper()
    for tk in expected_tickers:
        if tk in answer_upper:
            covered.append(tk)
    return covered


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def extract_answer_text(result: dict) -> str:
    """Extract the answer text from a pipeline result dict."""
    parsed = result.get("parsed")
    if parsed is not None:
        if hasattr(parsed, "answer"):
            return parsed.answer
    raw = result.get("raw_output", "")
    return str(raw)


def run_single_query(
    pipeline_instance,
    query_config: dict,
    config_name: str,
    run_index: int = 0,
) -> QueryResult:
    """Run a single query through the production pipeline and collect metrics."""
    qr = QueryResult(
        query_id=query_config["id"],
        category=query_config["category"],
        config_name=config_name,
    )

    try:
        t0 = time.time()

        # Run the production query
        result = pipeline_instance.query(
            user_query=query_config["query"],
            ticker=query_config.get("ticker"),
            fiscal_year=query_config.get("fiscal_year"),
            tickers=query_config.get("tickers"),
            top_k=3,
        )

        t_end = time.time()
        qr.e2e_latency_ms = (t_end - t0) * 1000
        qr.raw_result = result

        # Extract answer
        qr.answer = extract_answer_text(result)

        # Numerical correctness
        qr.numerical_correct = check_numerical_correctness(
            qr.answer, query_config.get("expected_numbers", [])
        )

        # Ticker coverage
        qr.ticker_coverage = check_ticker_coverage(
            qr.answer, query_config.get("expected_tickers", [])
        )
        qr.ticker_coverage_complete = (
            set(qr.ticker_coverage) == set(query_config.get("expected_tickers", []))
        )

        # Parse the answer for structured data
        parsed = result.get("parsed")
        if parsed is not None:
            # Sources / guardrail info
            guardrail = getattr(parsed, "guardrail_status", None)
            if guardrail is not None:
                qr.guardrail_passed = getattr(guardrail, "passed", False)
                qr.guardrail_detail = getattr(guardrail, "detail", "")

    except Exception as exc:
        qr.error = str(exc)

    return qr


def extract_hybrid_search_metrics(pipeline_instance, query_config: dict) -> dict:
    """
    Run retrieval only (no generation) to capture chunk-level metrics.
    Returns dict with candidate counts, ticker distribution, and latency.
    """
    metrics = {}
    try:
        t0 = time.time()

        # Use the pipeline's internal retrieval path
        query = query_config["query"]
        ticker = query_config.get("ticker")
        fiscal_year = query_config.get("fiscal_year")
        tickers = query_config.get("tickers")

        all_companies = (bool(ticker) and ticker.upper() == "ALL") or any(
            bool(t) and t.upper() == "ALL" for t in (tickers or [])
        )

        multi_ticker = pipeline_instance._has_multi_ticker_intent(query) or all_companies

        # Simulate the pipeline's retrieval path
        if multi_ticker:
            if all_companies:
                from config.settings import SUPPORTED_TICKERS
                compare_pool = list(SUPPORTED_TICKERS)
            else:
                compare_pool = pipeline_instance._comparison_ticker_pool(query, tickers)

            # Run balanced sub-retrievals
            raw_chunks = pipeline_instance._balanced_ticker_subretrievals_sync(
                query, compare_pool, fiscal_year=fiscal_year, top_k=3,
            )
        else:
            # Single-ticker path
            effective_ticker = ticker
            if tickers and len(tickers) == 1:
                effective_ticker = tickers[0]

            hybrid_filter = {"ticker": effective_ticker} if effective_ticker else {}
            if fiscal_year:
                hybrid_filter["fiscal_year"] = fiscal_year
            if not hybrid_filter.get("fiscal_year"):
                hybrid_filter["fiscal_year"] = pipeline_instance._query_fiscal_year(query)

            raw_chunks = pipeline_instance._run_hybrid_retrieval(
                query, [query], hybrid_filter,
            )

        t_end = time.time()
        metrics["retrieval_ms"] = (t_end - t0) * 1000
        metrics["num_candidates"] = len(raw_chunks)

        # Ticker distribution
        ticker_counts = {}
        for chunk in raw_chunks:
            meta = chunk.get("metadata", {})
            tk = meta.get("ticker", "UNKNOWN")
            ticker_counts[tk] = ticker_counts.get(tk, 0) + 1
        metrics["ticker_distribution"] = ticker_counts

        # Check if we can extract reranker scores
        # For multi-ticker path, reranker scores are embedded in the chunks
        scores = [c.get("rerank_score") for c in raw_chunks if "rerank_score" in c]
        if scores:
            metrics["rerank_scores"] = scores
            metrics["top_score"] = max(scores)
            metrics["min_score"] = min(scores)

        metrics["num_documents_after_augment"] = len(raw_chunks)

    except Exception as exc:
        metrics["error"] = str(exc)

    return metrics


# ── Benchmark Runner ───────────────────────────────────────────────────

def run_benchmark(
    pipeline_instance,
    config_name: str,
    queries: list[dict],
    num_warmup_runs: int = 1,
    num_measurement_runs: int = 3,
) -> list[QueryResult]:
    """Run the full benchmark for a given configuration."""
    all_results = []

    # Warmup run
    print(f"\n  [{config_name}] Warm-up run ({num_warmup_runs} iteration)...")
    for q in queries[:1]:  # Just warm up with first query
        for _ in range(num_warmup_runs):
            _ = run_single_query(pipeline_instance, q, config_name, run_index=0)
            gc.collect()

    # Measurement runs
    for run_idx in range(num_measurement_runs):
        print(f"  [{config_name}] Measurement run {run_idx + 1}/{num_measurement_runs}...")
        for q in queries:
            qr = run_single_query(pipeline_instance, q, config_name, run_index=run_idx)
            all_results.append(qr)
            status = "OK" if not qr.error else f"ERR: {qr.error[:60]}"
            print(f"    {qr.query_id}: {qr.e2e_latency_ms:.0f}ms | "
                  f"num_correct={qr.numerical_correct} | "
                  f"tickers={qr.ticker_coverage} | {status}")

    return all_results


def aggregate_results(results: list[QueryResult]) -> dict:
    """Aggregate benchmark results by query, computing mean/median/min/max/std."""
    # Group by query_id
    by_query = {}
    for r in results:
        if r.query_id not in by_query:
            by_query[r.query_id] = []
        by_query[r.query_id].append(r)

    aggregated = {}
    for qid, runs in by_query.items():
        latencies = [r.e2e_latency_ms for r in runs if not r.error]
        numerical = [r.numerical_correct for r in runs if not r.error]

        agg = {
            "query_id": qid,
            "category": runs[0].category,
            "config_name": runs[0].config_name,
            "num_runs": len([r for r in runs if not r.error]),
            "num_errors": len([r for r in runs if r.error]),
        }

        if latencies:
            agg["latency_mean_ms"] = statistics.mean(latencies)
            agg["latency_median_ms"] = statistics.median(latencies)
            agg["latency_min_ms"] = min(latencies)
            agg["latency_max_ms"] = max(latencies)
            agg["latency_std_ms"] = statistics.stdev(latencies) if len(latencies) > 1 else 0
            agg["latency_p95_ms"] = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[0]

        agg["numerical_correct_rate"] = sum(numerical) / len(numerical) if numerical else 0
        agg["ticker_coverage"] = runs[0].ticker_coverage
        agg["ticker_coverage_complete"] = all(r.ticker_coverage_complete for r in runs if not r.error)
        agg["answer_sample"] = runs[0].answer[:200] if runs else ""

        aggregated[qid] = agg

    return aggregated


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("A/B BENCHMARK: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20")
    print("No production configuration was changed during this experiment.")
    print("=" * 80)

    # ── Phase 1: Create Config A pipeline (HYBRID_TOP_K=40) ──
    print("\n[Phase 1] Creating Config A pipeline (HYBRID_TOP_K=40, production default)...")
    t0 = time.time()
    pipe_a = pipeline_mod.FinancialRAGPipeline(
        enable_hybrid_retrieval=True,
        enable_post_retrieval=True,
        enable_cache=False,
        enable_guardrail=False,
    )
    pipe_a.warm_reranker()
    print(f"  Config A pipeline ready in {time.time() - t0:.1f}s")
    print(f"  Hybrid search top_k = {pipe_a._hybrid_search._top_k}")

    # ── Phase 2: Create Config B pipeline (HYBRID_TOP_K=20) ──
    print("\n[Phase 2] Creating Config B pipeline (HYBRID_TOP_K=20)...")
    t0 = time.time()

    # Patch the settings before creating the second pipeline
    original_hybrid_top_k = settings.HYBRID_TOP_K
    original_pipeline_hybrid_top_k = pipeline_mod.HYBRID_TOP_K
    original_hs_hybrid_top_k = hs_mod.HYBRID_TOP_K

    settings.HYBRID_TOP_K = 20
    pipeline_mod.HYBRID_TOP_K = 20
    hs_mod.HYBRID_TOP_K = 20

    pipe_b = pipeline_mod.FinancialRAGPipeline(
        enable_hybrid_retrieval=True,
        enable_post_retrieval=True,
        enable_cache=False,
        enable_guardrail=False,
    )

    # Restore original values
    settings.HYBRID_TOP_K = original_hybrid_top_k
    pipeline_mod.HYBRID_TOP_K = original_pipeline_hybrid_top_k
    hs_mod.HYBRID_TOP_K = original_hs_hybrid_top_k

    pipe_b.warm_reranker()
    print(f"  Config B pipeline ready in {time.time() - t0:.1f}s")
    print(f"  Hybrid search top_k = {pipe_b._hybrid_search._top_k}")

    # Verify the configs are different
    assert pipe_a._hybrid_search._top_k == 40, f"Config A should be 40, got {pipe_a._hybrid_search._top_k}"
    assert pipe_b._hybrid_search._top_k == 20, f"Config B should be 20, got {pipe_b._hybrid_search._top_k}"
    print("\n  ✓ Configurations verified: A=40, B=20")

    # ── Phase 3: Run benchmarks ──
    print(f"\n[Phase 3] Running benchmark ({len(TEST_QUERIES)} queries x 3 runs each)...")

    print("\n--- Configuration A (HYBRID_TOP_K=40) ---")
    results_a = run_benchmark(pipe_a, "A_K40", TEST_QUERIES, num_warmup_runs=1, num_measurement_runs=3)

    print("\n--- Configuration B (HYBRID_TOP_K=20) ---")
    results_b = run_benchmark(pipe_b, "B_K20", TEST_QUERIES, num_warmup_runs=1, num_measurement_runs=3)

    # ── Phase 4: Aggregate results ──
    print("\n[Phase 4] Aggregating results...")
    agg_a = aggregate_results(results_a)
    agg_b = aggregate_results(results_b)

    # ── Phase 5: Compute comparison metrics ──
    print("\n[Phase 5] Computing comparison metrics...")

    comparison = {}
    for qid in agg_a:
        a = agg_a[qid]
        b = agg_b.get(qid, {})
        comp = {
            "query_id": qid,
            "category": a["category"],
            "latency_a_ms": a.get("latency_mean_ms", 0),
            "latency_b_ms": b.get("latency_mean_ms", 0),
            "latency_reduction_pct": (
                (1 - b.get("latency_mean_ms", 0) / a.get("latency_mean_ms", 1)) * 100
                if a.get("latency_mean_ms", 0) > 0 else 0
            ),
            "numerical_correct_a": a.get("numerical_correct_rate", 0),
            "numerical_correct_b": b.get("numerical_correct_rate", 0),
            "ticker_coverage_a": a.get("ticker_coverage_complete", False),
            "ticker_coverage_b": b.get("ticker_coverage_complete", False),
            "answer_changed": a.get("answer_sample", "") != b.get("answer_sample", ""),
            "answer_a": a.get("answer_sample", ""),
            "answer_b": b.get("answer_sample", ""),
        }
        comparison[qid] = comp

    # ── Phase 6: Retrieval-only metrics ──
    print("\n[Phase 6] Collecting retrieval-only metrics (candidate counts)...")
    retrieval_metrics_a = {}
    retrieval_metrics_b = {}

    for q in TEST_QUERIES:
        print(f"  Retrieval metrics for {q['id']}...")
        rm_a = extract_hybrid_search_metrics(pipe_a, q)
        rm_b = extract_hybrid_search_metrics(pipe_b, q)
        retrieval_metrics_a[q["id"]] = rm_a
        retrieval_metrics_b[q["id"]] = rm_b
        print(f"    A: {rm_a.get('num_candidates', '?')} candidates, "
              f"retrieval={rm_a.get('retrieval_ms', 0):.0f}ms")
        print(f"    B: {rm_b.get('num_candidates', '?')} candidates, "
              f"retrieval={rm_b.get('retrieval_ms', 0):.0f}ms")

    # ── Phase 7: Write results ──
    output_dir = PROJECT_ROOT / "artifacts" / "ab_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "A/B Benchmark: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "disclaimer": "No production configuration was changed during this experiment.",
        "config_a": {
            "HYBRID_TOP_K": 40,
            "POST_RETRIEVAL_RERANK_TOP_N": 8,
            "model": "BAAI/bge-reranker-large",
        },
        "config_b": {
            "HYBRID_TOP_K": 20,
            "POST_RETRIEVAL_RERANK_TOP_N": 8,
            "model": "BAAI/bge-reranker-large",
        },
        "queries": TEST_QUERIES,
        "retrieval_a": retrieval_metrics_a,
        "retrieval_b": retrieval_metrics_b,
        "aggregated_a": agg_a,
        "aggregated_b": agg_b,
        "comparison": comparison,
        "summary": {},
    }

    # Compute summary statistics
    latencies_a = [c["latency_a_ms"] for c in comparison.values() if c["latency_a_ms"] > 0]
    latencies_b = [c["latency_b_ms"] for c in comparison.values() if c["latency_b_ms"] > 0]
    reductions = [c["latency_reduction_pct"] for c in comparison.values()]
    correct_a = [c["numerical_correct_a"] for c in comparison.values()]
    correct_b = [c["numerical_correct_b"] for c in comparison.values()]

    report["summary"] = {
        "latency_a_mean_ms": statistics.mean(latencies_a) if latencies_a else 0,
        "latency_b_mean_ms": statistics.mean(latencies_b) if latencies_b else 0,
        "latency_reduction_mean_pct": statistics.mean(reductions) if reductions else 0,
        "latency_reduction_median_pct": statistics.median(reductions) if reductions else 0,
        "numerical_correct_rate_a": statistics.mean(correct_a) if correct_a else 0,
        "numerical_correct_rate_b": statistics.mean(correct_b) if correct_b else 0,
        "ticker_coverage_a": all(c["ticker_coverage_a"] for c in comparison.values()),
        "ticker_coverage_b": all(c["ticker_coverage_b"] for c in comparison.values()),
        "answer_change_rate": sum(1 for c in comparison.values() if c["answer_changed"]) / len(comparison) if comparison else 0,
    }

    # Write JSON report
    report_path = output_dir / "ab_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report written to: {report_path}")

    # Write human-readable summary
    summary_path = output_dir / "ab_benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("A/B BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("No production configuration was changed during this experiment.\n\n")

        s = report["summary"]
        f.write(f"Overall Results:\n")
        f.write(f"  Config A (K=40) mean latency: {s['latency_a_mean_ms']:.0f}ms\n")
        f.write(f"  Config B (K=20) mean latency: {s['latency_b_mean_ms']:.0f}ms\n")
        f.write(f"  Mean latency reduction: {s['latency_reduction_mean_pct']:.1f}%\n")
        f.write(f"  Median latency reduction: {s['latency_reduction_median_pct']:.1f}%\n")
        f.write(f"  Numerical correctness A: {s['numerical_correct_rate_a']:.1%}\n")
        f.write(f"  Numerical correctness B: {s['numerical_correct_rate_b']:.1%}\n")
        f.write(f"  Ticker coverage A complete: {s['ticker_coverage_a']}\n")
        f.write(f"  Ticker coverage B complete: {s['ticker_coverage_b']}\n")
        f.write(f"  Answer change rate: {s['answer_change_rate']:.1%}\n\n")

        f.write("Per-Query Details:\n")
        f.write("-" * 80 + "\n")
        for qid, comp in comparison.items():
            f.write(f"\n  {qid} ({comp['category']}):\n")
            f.write(f"    Latency: {comp['latency_a_ms']:.0f}ms -> {comp['latency_b_ms']:.0f}ms "
                    f"({comp['latency_reduction_pct']:+.1f}%)\n")
            f.write(f"    Numerical correct: A={comp['numerical_correct_a']:.0%} B={comp['numerical_correct_b']:.0%}\n")
            f.write(f"    Ticker coverage: A={comp['ticker_coverage_a']} B={comp['ticker_coverage_b']}\n")
            f.write(f"    Answer changed: {comp['answer_changed']}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"  Summary written to: {summary_path}")

    # ── Cleanup ──
    print("\n[Cleanup] Closing pipeline instances...")
    try:
        pipe_a.close()
    except Exception:
        pass
    try:
        pipe_b.close()
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

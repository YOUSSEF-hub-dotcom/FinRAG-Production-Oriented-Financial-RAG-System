#!/usr/bin/env python3
"""
A/B Benchmark v2: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20
- 2 measurement runs per config
- Per-query timeout (120s single / 180s multi-ticker)
- Checkpoint save after each config
- Skip retrieval-only phase
"""

import gc
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path("/home/youssef/Financial_RAG")
os.chdir(PROJECT_ROOT)

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

TEST_QUERIES = [
    {"id": "ST1", "query": "What was Apple total net sales in fiscal year 2025?", "ticker": "AAPL", "fiscal_year": "2025", "tickers": None, "category": "single_ticker_factual", "expected_numbers": ["416161"], "expected_tickers": ["AAPL"]},
    {"id": "ST2", "query": "What was Microsoft total revenue in fiscal year 2025?", "ticker": "MSFT", "fiscal_year": "2025", "tickers": None, "category": "single_ticker_factual", "expected_numbers": ["281724"], "expected_tickers": ["MSFT"]},
    {"id": "ST3", "query": "What was NVIDIA total revenue in fiscal year 2026?", "ticker": "NVDA", "fiscal_year": "2026", "tickers": None, "category": "single_ticker_factual", "expected_numbers": ["130497"], "expected_tickers": ["NVDA"]},
    {"id": "FM1", "query": "What was Apple operating income in fiscal year 2025?", "ticker": "AAPL", "fiscal_year": "2025", "tickers": None, "category": "financial_metric", "expected_numbers": ["133050"], "expected_tickers": ["AAPL"]},
    {"id": "FM2", "query": "What was Apple gross margin percentage in fiscal year 2025?", "ticker": "AAPL", "fiscal_year": "2025", "tickers": None, "category": "financial_metric", "expected_numbers": ["46.9"], "expected_tickers": ["AAPL"]},
    {"id": "FM3", "query": "What was Microsoft net income in fiscal year 2025?", "ticker": "MSFT", "fiscal_year": "2025", "tickers": None, "category": "financial_metric", "expected_numbers": ["101832"], "expected_tickers": ["MSFT"]},
    {"id": "MT1", "query": "Compare the revenue of Apple, Microsoft, and Nvidia.", "ticker": None, "fiscal_year": None, "tickers": ["AAPL", "MSFT", "NVDA"], "category": "multi_ticker_comparison", "expected_numbers": ["416161", "281724", "130497"], "expected_tickers": ["AAPL", "MSFT", "NVDA"]},
    {"id": "MT2", "query": "Compare gross margins across Apple, Microsoft, and Nvidia.", "ticker": None, "fiscal_year": None, "tickers": ["AAPL", "MSFT", "NVDA"], "category": "multi_ticker_comparison", "expected_numbers": [], "expected_tickers": ["AAPL", "MSFT", "NVDA"]},
    {"id": "TH1", "query": "What were Apple iPhone net sales and Services net sales in fiscal year 2025?", "ticker": "AAPL", "fiscal_year": "2025", "tickers": None, "category": "table_heavy", "expected_numbers": ["209586", "109158"], "expected_tickers": ["AAPL"]},
    {"id": "TH2", "query": "What was Microsoft largest business segment by revenue in fiscal year 2025?", "ticker": "MSFT", "fiscal_year": "2025", "tickers": None, "category": "table_heavy", "expected_numbers": ["120810"], "expected_tickers": ["MSFT"]},
    {"id": "NEG1", "query": "What was Tesla total revenue in fiscal year 2025?", "ticker": "TSLA", "fiscal_year": "2025", "tickers": None, "category": "negative", "expected_numbers": [], "expected_tickers": []},
]

TIMEOUT_SINGLE_S = 120
TIMEOUT_MULTI_S = 180
NUM_RUNS = 2


@dataclass
class QueryResult:
    query_id: str = ""
    category: str = ""
    config_name: str = ""
    num_candidates: int = 0
    e2e_latency_ms: float = 0.0
    answer: str = ""
    numerical_correct: bool = False
    ticker_coverage: list = field(default_factory=list)
    ticker_coverage_complete: bool = False
    error: str = ""
    timed_out: bool = False


def check_numerical_correctness(answer, expected_numbers):
    if not expected_numbers:
        return True
    ans = re.sub(r"[,\s]", "", answer.lower())
    for num in expected_numbers:
        if re.sub(r"[,\s]", "", num) not in ans:
            return False
    return True


def check_ticker_coverage(answer, expected_tickers):
    if not expected_tickers:
        return []
    up = answer.upper()
    return [tk for tk in expected_tickers if tk in up]


def extract_answer_text(result):
    parsed = result.get("parsed")
    if parsed is not None and hasattr(parsed, "answer"):
        return parsed.answer
    return str(result.get("raw_output", ""))


def _run_query_inner(pipe, qc):
    return pipe.query(
        user_query=qc["query"], ticker=qc.get("ticker"),
        fiscal_year=qc.get("fiscal_year"), tickers=qc.get("tickers"), top_k=3,
    )


def run_single_query(pipe, qc, config_name, run_idx):
    qr = QueryResult(query_id=qc["id"], category=qc["category"], config_name=config_name)
    is_multi = qc.get("tickers") and len(qc.get("tickers", [])) > 1
    timeout_s = TIMEOUT_MULTI_S if is_multi else TIMEOUT_SINGLE_S
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run_query_inner, pipe, qc)
            result = future.result(timeout=timeout_s)
        qr.e2e_latency_ms = (time.time() - t0) * 1000
        qr.answer = extract_answer_text(result)
        qr.numerical_correct = check_numerical_correctness(
            qr.answer, qc.get("expected_numbers", [])
        )
        qr.ticker_coverage = check_ticker_coverage(
            qr.answer, qc.get("expected_tickers", [])
        )
        qr.ticker_coverage_complete = (
            set(qr.ticker_coverage) == set(qc.get("expected_tickers", []))
        )
        parsed = result.get("parsed")
        if parsed:
            sources = getattr(parsed, "sources", [])
            qr.num_candidates = len(sources) if sources else 0
    except FuturesTimeout:
        qr.timed_out = True
        qr.error = f"TIMEOUT after {timeout_s}s"
        qr.e2e_latency_ms = timeout_s * 1000
    except Exception as exc:
        qr.error = str(exc)[:200]
    return qr


def run_benchmark(pipe, config_name, queries, num_runs):
    all_results = []
    print(f"\n  [{config_name}] Warm-up (1 query)...")
    _ = run_single_query(pipe, queries[0], config_name, 0)
    gc.collect()

    for run_idx in range(num_runs):
        print(f"  [{config_name}] Run {run_idx + 1}/{num_runs}...")
        for qc in queries:
            qr = run_single_query(pipe, qc, config_name, run_idx)
            all_results.append(qr)
            flag = "TIMEOUT" if qr.timed_out else ("ERR" if qr.error else "OK")
            print(
                f"    {qr.query_id}: {qr.e2e_latency_ms:.0f}ms | "
                f"correct={qr.numerical_correct} | tickers={qr.ticker_coverage} | {flag}"
            )
    return all_results


def aggregate(results):
    by_q = {}
    for r in results:
        by_q.setdefault(r.query_id, []).append(r)
    agg = {}
    for qid, runs in by_q.items():
        good = [r for r in runs if not r.error and not r.timed_out]
        lats = [r.e2e_latency_ms for r in good]
        correct = [r.numerical_correct for r in good]
        a = {
            "query_id": qid,
            "category": runs[0].category,
            "config_name": runs[0].config_name,
            "num_valid": len(good),
            "num_errors": len([r for r in runs if r.error]),
            "num_timeouts": len([r for r in runs if r.timed_out]),
        }
        if lats:
            a["latency_mean_ms"] = statistics.mean(lats)
            a["latency_median_ms"] = statistics.median(lats)
            a["latency_min_ms"] = min(lats)
            a["latency_max_ms"] = max(lats)
            a["latency_std_ms"] = statistics.stdev(lats) if len(lats) > 1 else 0
        else:
            a["latency_mean_ms"] = 0
        a["numerical_correct_rate"] = sum(correct) / len(correct) if correct else 0
        a["ticker_coverage"] = runs[0].ticker_coverage
        a["ticker_coverage_complete"] = all(r.ticker_coverage_complete for r in good)
        a["answer_sample"] = runs[0].answer[:300] if runs else ""
        agg[qid] = a
    return agg


def save_checkpoint(data, output_dir, tag=""):
    fname = f"ab_benchmark_report{tag}.json"
    with open(output_dir / fname, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Checkpoint saved: {output_dir / fname}")


def write_summary(report, output_dir):
    s = report["summary"]
    path = output_dir / "ab_benchmark_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("A/B BENCHMARK SUMMARY: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20\n")
        f.write("=" * 80 + "\n\n")
        f.write("No production configuration was changed.\n\n")
        f.write(f"Config A (K=40) mean latency: {s.get('latency_a_mean_ms', 0):.0f}ms\n")
        f.write(f"Config B (K=20) mean latency: {s.get('latency_b_mean_ms', 0):.0f}ms\n")
        f.write(f"Mean latency reduction: {s.get('latency_reduction_mean_pct', 0):.1f}%\n")
        f.write(f"Median latency reduction: {s.get('latency_reduction_median_pct', 0):.1f}%\n\n")
        f.write(f"Numerical correctness A: {s.get('numerical_correct_rate_a', 0):.1%}\n")
        f.write(f"Numerical correctness B: {s.get('numerical_correct_rate_b', 0):.1%}\n\n")
        f.write("Per-Query:\n")
        f.write("-" * 80 + "\n")
        for qid, c in report.get("comparison", {}).items():
            f.write(
                f"  {qid} ({c['category']}): "
                f"{c['latency_a_ms']:.0f}ms -> {c['latency_b_ms']:.0f}ms "
                f"({c['latency_reduction_pct']:+.1f}%) | "
                f"correct A={c['numerical_correct_a']:.0%} B={c['numerical_correct_b']:.0%}\n"
            )
        f.write("\n" + "=" * 80 + "\n")
    print(f"  Summary: {path}")


def main():
    t_global = time.time()
    output_dir = PROJECT_ROOT / "artifacts" / "ab_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("A/B BENCHMARK v3: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20")
    print(f"{len(TEST_QUERIES)} queries x {NUM_RUNS} runs x 2 configs")
    print("Configs run SEQUENTIALLY to avoid Qdrant lock contention.")
    print("=" * 80)

    # Phase 1: Config A (K=40) - run and close before B
    print("\n[Phase 1] Config A pipeline (K=40)...")
    t0 = time.time()
    pipe_a = pipeline_mod.FinancialRAGPipeline(
        enable_hybrid_retrieval=True, enable_post_retrieval=True,
        enable_cache=False, enable_guardrail=False,
    )
    pipe_a.warm_reranker()
    print(f"  Ready in {time.time() - t0:.1f}s | top_k={pipe_a._hybrid_search._top_k}")
    assert pipe_a._hybrid_search._top_k == 40

    print(f"\n[Phase 2] Config A benchmark ({NUM_RUNS} runs)...")
    results_a = run_benchmark(pipe_a, "A_K40", TEST_QUERIES, NUM_RUNS)
    agg_a = aggregate(results_a)
    save_checkpoint(
        {"config": "A_K40", "results": [r.__dict__ for r in results_a], "aggregated": agg_a},
        output_dir, "_configA",
    )

    # Close A before creating B to avoid Qdrant lock contention
    print("\n[Phase 3] Closing Config A pipeline...")
    try:
        pipe_a.close()
    except Exception:
        pass
    import gc as gc_mod
    gc_mod.collect()
    import shutil as _shutil
    # Remove Qdrant lock if any
    lock_file = PROJECT_ROOT / "data" / "qdrant_db" / ".lock"
    if lock_file.exists():
        lock_file.unlink()
        print("  Removed stale Qdrant lock file")

    # Phase 4: Config B (K=20) - fresh start with no contention
    print("\n[Phase 4] Config B pipeline (K=20)...")
    t0 = time.time()
    orig_sk = settings.HYBRID_TOP_K
    orig_pk = pipeline_mod.HYBRID_TOP_K
    orig_hk = hs_mod.HYBRID_TOP_K
    settings.HYBRID_TOP_K = 20
    pipeline_mod.HYBRID_TOP_K = 20
    hs_mod.HYBRID_TOP_K = 20
    pipe_b = pipeline_mod.FinancialRAGPipeline(
        enable_hybrid_retrieval=True, enable_post_retrieval=True,
        enable_cache=False, enable_guardrail=False,
    )
    settings.HYBRID_TOP_K = orig_sk
    pipeline_mod.HYBRID_TOP_K = orig_pk
    hs_mod.HYBRID_TOP_K = orig_hk
    pipe_b.warm_reranker()
    print(f"  Ready in {time.time() - t0:.1f}s | top_k={pipe_b._hybrid_search._top_k}")
    assert pipe_b._hybrid_search._top_k == 20

    print(f"\n[Phase 5] Config B benchmark ({NUM_RUNS} runs)...")
    results_b = run_benchmark(pipe_b, "B_K20", TEST_QUERIES, NUM_RUNS)
    agg_b = aggregate(results_b)
    save_checkpoint(
        {"config": "B_K20", "results": [r.__dict__ for r in results_b], "aggregated": agg_b},
        output_dir, "_configB",
    )

    try:
        pipe_b.close()
    except Exception:
        pass

    # Phase 6: Comparison (from saved checkpoints, no pipelines needed)
    print("\n[Phase 6] Computing comparison...")
    comparison = {}
    for qid in agg_a:
        a = agg_a[qid]
        b = agg_b.get(qid, {})
        la = a.get("latency_mean_ms", 0)
        lb = b.get("latency_mean_ms", 0)
        comp = {
            "query_id": qid, "category": a["category"],
            "latency_a_ms": la, "latency_b_ms": lb,
            "latency_reduction_pct": ((1 - lb / la) * 100) if la > 0 else 0,
            "numerical_correct_a": a.get("numerical_correct_rate", 0),
            "numerical_correct_b": b.get("numerical_correct_rate", 0),
            "ticker_coverage_a": a.get("ticker_coverage_complete", False),
            "ticker_coverage_b": b.get("ticker_coverage_complete", False),
            "answer_changed": a.get("answer_sample", "") != b.get("answer_sample", ""),
            "answer_a": a.get("answer_sample", ""),
            "answer_b": b.get("answer_sample", ""),
            "timeouts_a": a.get("num_timeouts", 0),
            "timeouts_b": b.get("num_timeouts", 0),
        }
        comparison[qid] = comp

    lat_a = [c["latency_a_ms"] for c in comparison.values() if c["latency_a_ms"] > 0]
    lat_b = [c["latency_b_ms"] for c in comparison.values() if c["latency_b_ms"] > 0]
    reds = [c["latency_reduction_pct"] for c in comparison.values()]
    ca = [c["numerical_correct_a"] for c in comparison.values()]
    cb = [c["numerical_correct_b"] for c in comparison.values()]

    summary = {
        "latency_a_mean_ms": statistics.mean(lat_a) if lat_a else 0,
        "latency_b_mean_ms": statistics.mean(lat_b) if lat_b else 0,
        "latency_reduction_mean_pct": statistics.mean(reds) if reds else 0,
        "latency_reduction_median_pct": statistics.median(reds) if reds else 0,
        "numerical_correct_rate_a": statistics.mean(ca) if ca else 0,
        "numerical_correct_rate_b": statistics.mean(cb) if cb else 0,
        "ticker_coverage_a": all(c["ticker_coverage_a"] for c in comparison.values()),
        "ticker_coverage_b": all(c["ticker_coverage_b"] for c in comparison.values()),
        "answer_change_rate": (
            sum(1 for c in comparison.values() if c["answer_changed"]) / len(comparison)
            if comparison else 0
        ),
        "total_time_s": time.time() - t_global,
    }

    report = {
        "experiment": "A/B Benchmark: HYBRID_TOP_K=40 vs HYBRID_TOP_K=20",
        "version": "v3-sequential",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_runs": NUM_RUNS,
        "timeout_single_s": TIMEOUT_SINGLE_S,
        "timeout_multi_s": TIMEOUT_MULTI_S,
        "config_a": {"HYBRID_TOP_K": 40, "POST_RETRIEVAL_RERANK_TOP_N": 8},
        "config_b": {"HYBRID_TOP_K": 20, "POST_RETRIEVAL_RERANK_TOP_N": 8},
        "queries": TEST_QUERIES,
        "aggregated_a": agg_a,
        "aggregated_b": agg_b,
        "comparison": comparison,
        "summary": summary,
    }

    save_checkpoint(report, output_dir, "")
    write_summary(report, output_dir)

    print(f"\nTotal time: {time.time() - t_global:.0f}s")

    print("\n" + "=" * 80)
    print("BENCHMARK v3 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

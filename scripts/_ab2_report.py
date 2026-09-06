import json, os

ART = "artifacts/ab2_benchmark"

def load(fn):
    with open(os.path.join(ART, fn)) as f:
        return json.load(f)

expA = load("expA_fp16.json")
expB = load("expB_v2m3.json")
expC = load("expC_prefilter.json")
expD = load("expD_topn.json")
e2e = load("e2e_correctness.json")

print("EXP A: comparison per-query")
for pq in expA["comparison"]["per_query"]:
    print(f"  {pq['query_id']:<4} fp32={pq['fp32_latency_mean_ms']:>7.0f}ms "
          f"fp16={pq['fp16_latency_mean_ms']:>6.0f}ms speedup={pq['speedup_pct']:>5.1f}% "
          f"overlap={pq['top8_overlap']}/8 ev_eq={pq['evidence_identical']}")
print("\nEXP A: size sweep (Q2 candidates)")
for n, v in expA["comparison"]["size_sweep"].items():
    print(f"  n={n:<4} fp32={v['fp32_ms']:.0f}ms fp16={v['fp16_ms']:.0f}ms "
          f"speedup={v['speedup_pct']:.1f}% overlap={v['top8_overlap']}/8")
print("\nvram:", expA["vram_reserved_mb"])

print("\nEXP B: v2-m3 (fp16, batch16)")
for qid, v in expB["per_query"].items():
    print(f"  {qid:<4} predict={v['predict_ms']:.0f}ms overlap={v['top8_overlap_vs_large']}/8 "
          f"ev_eq={v['evidence_identical_to_large']}")
print("  vram:", expB["vram_reserved_mb"])

print("\nEXP C: prefilter (latency + overlap)")
for qid, v in expC["per_query"].items():
    caps = ", ".join(f"cap{cap}:{c['candidates_in']}c/{c['predict_ms']:.0f}ms/ov{c['top8_overlap']}"
                     for cap, c in v["caps"].items())
    print(f"  {qid:<4} base={v['baseline_predict_ms']:.0f}ms | {caps}")

print("\nEXP D: topn 8 vs 6 (context + evidence)")
for qid, v in expD["per_query"].items():
    print(f"  {qid:<4} ctx n8={v['context_chunks_n8']} n6={v['context_chunks_n6']} "
          f"tickers_kept={v['ticker_coverage_kept']} ev_lost={v['evidence_lost_n6']} "
          f"tables n8={v['table_chunks_n8']} n6={v['table_chunks_n6']}")

print("\nE2E correctness")
for r in e2e:
    print(f"  {r['query_id']:<3} model={r.get('model_used')} fallback={r.get('fallback_triggered')} "
          f"err={bool(r.get('error'))}")
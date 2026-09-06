import json

d = json.load(open("artifacts/ab2_benchmark/debug_e2_refusal.json"))

print("=== PROBES ===")
for k, v in d["probes"].items():
    print(f"  {k}: {v}")

print("\n=== BALANCED CALL ===", d["stages"]["balanced_call"])
bal = d["stages"]["balanced_output"]
print(f"balanced n={len(bal)}; per-ticker distribution:")
from collections import Counter, defaultdict
div = defaultdict(int)
for doc in bal:
    m = doc.get("metadata") or {}
    div[(m.get("ticker"), m.get("fiscal_year"))] += 1
for k, v in sorted(div.items(), key=str):
    print(f"  {k}: {v}")

# top-8 per ticker with rerank scores
print("\n=== PER-TICKER BALANCED (8 each) ===")
seen_tk = defaultdict(int)
for doc in bal:
    m = doc.get("metadata") or {}
    tk = m.get("ticker")
    if seen_tk[tk] < 8:
        seen_tk[tk] += 1
        print(f"  {tk} | {doc.get('chunk_id')} | sec={m.get('section')} | "
              f"year={m.get('fiscal_year')} | type={m.get('doc_type')} | "
              f"ct={m.get('contains_table')} | rrk={doc.get('score')} | "
              f"rr={doc.get('rerank_score')} | text[:60]={ (doc.get('text') or '')[:60]!r}")

print("\n=== RESOLVE TABLE PLACEHOLDERS ===")
rin = d["stages"].get("resolve_input") or []
rout = d["stages"].get("resolve_output") or []
print(f"  in={len(rin)} out={len(rout)}")
for i, doc in enumerate(rin):
    if "%%TABLE_" in (doc.get("text") or ""):
        print(f"  in[{i}] HAS placeholder: {doc.get('chunk_id')}")
for i, doc in enumerate(rout):
    t = doc.get("text") or ""
    if "%%TABLE_" in t or "TABLE" in t[:80]:
        print(f"  out[{i}] STILL has markers: {doc.get('chunk_id')} ({t[:80]!r})")

print("\n=== AUGMENT CALL ===", d["stages"]["augment_call"])
ain = d["stages"].get("augment_input") or []
aout = d["stages"].get("augment_output") or []
print(f"  input={len(ain)} -> output={len(aout)}")
print("  OUTPUT DOCS:")
for i, doc in enumerate(aout):
    m = doc.get("metadata") or {}
    t = doc.get("text") or ""
    print(f"    [{i}] id={doc.get('chunk_id')} aux_rank={m.get('augment_rank')} | "
          f"sec={m.get('section')} | supp={doc.get('supplementary')} | "
          f"ct={m.get('contains_table')} | chars={len(t)}")
    print(f"        text[:180]={t[:180]!r}")
print(f"  total_chars={sum(len((x.get('text')) or '') for x in aout)}")

print("\n=== DOCS SENT TO GENERATOR ===")
gd = d["llm"].get("docs_sent_to_generator") or []
print(f"  n={len(gd)}")
for x in gd:
    m = x.get("metadata") or {}
    print(f"    {m.get('ticker')} | {x.get('chunk_id')} | {m.get('section')} | {len(x.get('text') or '')} chars")
print("  gen_kwargs:", d["llm"].get("gen_kwargs"))

print("\n=== LLM MESSAGES (system prompt + human w/ context + question) ===")
for msg in d["llm"].get("messages", []):
    print(f"  --- {msg['role']} ---")
    print(msg["content"][:3000])
    print()

print("\n=== LLM CALL RETURNS ===")
for r in d["llm"].get("call_returns", []):
    print("  model:", r["model"], "ttft:", r["ttft"])
    print("  raw:", r["raw"])

print("\n=== GUARDRAIL CALLS/VERDICTS ===")
print(json.dumps(d.get("guardrail_verdicts"), indent=2, default=str))
for g in d["llm"].get("guardrail_calls", []):
    print("  answer:", (g.get("answer") or "")[:200])
    print("  extracted_raw:", (g.get("extracted_raw_data") or "")[:200])

print("\n=== RESULT ===")
print(json.dumps(d["result"], indent=2, default=str))
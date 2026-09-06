import json
d = json.load(open("artifacts/ab2_benchmark/debug_e2_fixsim.json"))
print("=== AUGMENT OUTPUT (with fiscal_year=2025 sim) ===")
print("n_docs:", len(d["augment"]["docs"]))
for i, doc in enumerate(d["augment"]["docs"]):
    m = doc.get("metadata") or {}
    t = doc.get("text") or ""
    print(f'[{i}] id={doc.get("chunk_id")} | aux_rank={m.get("augment_rank")} | '
          f'supp={doc.get("supplementary")} | ticker={m.get("ticker")} | '
          f'sec={(m.get("section") or "")[:45]} | chars={len(t)}')
tot = sum(len((x.get("text")) or "") for x in d["augment"]["docs"])
print("total_chars:", tot)
all_text = " ".join((x.get("text")) or "" for x in d["augment"]["docs"])
low = all_text.lower()
for tok in ["416161", "281724", "130497", "60616", "operating income", "operating margin", "net sales", "net revenue"]:
    print(f"  context contains '{tok}':", tok in all_text or tok in low)
print()
print("=== LLM RAW (simulated fix) ===")
print((d["llm"].get("raw") or "")[:1600])
print()
print("model:", d.get("model_used"), "fallback:", d.get("fallback"))
import json

d = json.load(open("artifacts/evaluation_scores.json"))
for m in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
    print(m, round(d[m], 3))
print("judge_model:", d["judge_model"])
print("samples:", len(d["samples"]))
print("\nper-sample:")
for s in d["samples"]:
    print(round(s["faithfulness"], 2), round(s["answer_relevance"], 2),
          round(s["context_precision"], 2), round(s["context_recall"], 2),
          "|", s["question"][:55])

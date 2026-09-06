import json

p = json.load(open("artifacts/evaluation_scores_progress.json"))
print("samples done:", len(p["samples"]))
for s in p["samples"]:
    print(
        f"  {round(s['faithfulness'],2)} {round(s['answer_relevance'],2)} "
        f"{round(s['context_precision'],2)} {round(s['context_recall'],2)} | {s['question'][:55]}"
    )

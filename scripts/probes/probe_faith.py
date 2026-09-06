import json

d = json.load(open("artifacts/evaluation_scores.json"))
s = d["samples"][0]
print("Q:", s["question"][:70])
print("A:", s["answer"][:80])
print("faith reasoning:", s.get("faithfulness_reasoning", ""))
print("keys:", list(s.keys()))

# where does the figure sit in the first context?
import pandas as pd
df = pd.read_csv("artifacts/evaluation_results.csv")
ctxs = json.loads(df["contexts"].iloc[0])
for c in ctxs:
    t = c.get("text", "")
    i = t.find("416,161")
    j = t.find("416161")
    print("ctx len:", len(t), "| '416,161' at", i, "| '416161' at", j, "| first 200:", t[:200].replace(chr(10), " "))

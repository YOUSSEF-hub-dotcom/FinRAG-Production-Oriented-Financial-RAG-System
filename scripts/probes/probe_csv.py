import pandas as pd
import json

df = pd.read_csv("artifacts/evaluation_results.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("model_used:", df["model_used"].value_counts().to_dict())
print("\nrow 0 contexts len:", len(df["contexts"].iloc[0]))
try:
    ctxs = json.loads(df["contexts"].iloc[0])
    for c in ctxs:
        print("  -", c.get("metadata", {}).get("chunk_id"), "textlen:", len(c.get("text", "")))
except Exception as e:
    print("parse err:", e)

# numeric target check: does each answer contain its expected GT number?
import re
def numbers(s):
    return set(re.sub(r"[^0-9.]", " ", str(s)).split())
df["gt_nums"] = df["ground_truth"].apply(lambda s: numbers(s))
df["ans_nums"] = df["answer"].apply(lambda s: numbers(s))
# crude overlap of the big figures only
big = df["gt_nums"].apply(lambda n: {x for x in n if len(x) >= 4})
def has_any(row):
    return bool(big[row.name] & df["ans_nums"][row.name])
df["covers_gt_figure"] = df.apply(has_any, axis=1)
print("\nrows whose answer misses every big GT figure:", int((~df["covers_gt_figure"]).sum()))
for i, row in df.iterrows():
    if not row["covers_gt_figure"]:
        print("  MISS:", row["question"][:60], "| ans:", str(row["answer"])[:60])

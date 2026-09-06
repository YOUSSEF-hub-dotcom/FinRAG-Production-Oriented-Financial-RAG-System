import json, re
import pandas as pd

df = pd.read_csv("artifacts/evaluation_results.csv")

def figures(gt):
    return [m for m in re.findall(r"\d[\d,]*", gt)]

for i, row in df.iterrows():
    gt = str(row["ground_truth"])
    ctxs = json.loads(row["contexts"])
    texts = [c.get("text", "") if isinstance(c, dict) else c for c in ctxs]
    fs = figures(gt)
    print(f"--- [{i:02d}] nctx={len(texts)}")
    print(f"    GT: {gt[:90]}")
    for j, t in enumerate(texts):
        hits = []
        for f in fs:
            ff = f.replace(",", "")
            pos = t.find(ff)
            if pos >= 0:
                hits.append(f"{f}@{pos}")
        print(f"    ctx{j} len={len(t)} hits={hits or 'NONE'}")

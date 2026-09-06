import json
import pandas as pd

df = pd.read_csv("artifacts/evaluation_results.csv")

total = 0
tables = 0
placeholder_in_text = 0
for c in df["contexts"]:
    for x in json.loads(c):
        total += 1
        meta = x.get("metadata", {})
        if meta.get("contains_table") is True:
            tables += 1
        if "%%TABLE_" in (x.get("text") or ""):
            placeholder_in_text += 1

print("total contexts:", total)
print("contains_table=True contexts:", tables)
print("texts containing %%TABLE_ placeholder:", placeholder_in_text)

# Show all distinct contains_table values
vals = set()
for c in df["contexts"]:
    for x in json.loads(c):
        vals.add(x.get("metadata", {}).get("contains_table"))
print("distinct contains_table values:", vals)

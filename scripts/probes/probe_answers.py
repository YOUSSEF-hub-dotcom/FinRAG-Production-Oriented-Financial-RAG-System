import pandas as pd

df = pd.read_csv("artifacts/evaluation_results.csv")
for i, row in df.iterrows():
    print(f"[{i:02d}]", str(row["question"])[:60])
    print("    A:", str(row["answer"])[:100])
    print("    GT:", str(row["ground_truth"])[:100])

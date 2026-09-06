import pandas as pd

df = pd.read_csv("artifacts/evaluation_results.csv")
print("rows:", len(df))
print(df["model_used"].value_counts())
emp = df["answer"].fillna("").str.len() < 5
print("empty/short answers:", int(emp.sum()))
print("answer len stats:")
print(df["answer"].fillna("").str.len().describe())
print("\nfirst answer sample:")
print(str(df["answer"].iloc[0])[:300])

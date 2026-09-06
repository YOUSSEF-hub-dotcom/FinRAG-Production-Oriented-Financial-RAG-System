import pandas as pd
df = pd.read_csv("artifacts/evaluation_results.csv")
print("SHAPE:", df.shape)
print("COLS:", list(df.columns))
print(df[["question", "model_used", "fallback_triggered"]].head(3).to_string())

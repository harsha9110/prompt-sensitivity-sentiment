import pandas as pd

df = pd.read_csv("results/smollm_predictions_raw.csv")

prompts = ["direct", "formal", "concise", "analyst", "definition"]

df["disagreement"] = df[prompts].nunique(axis=1) > 1

print("Disagreement:", df["disagreement"].mean())
print("Count:", df["disagreement"].sum())
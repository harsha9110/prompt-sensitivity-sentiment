import pandas as pd

df = pd.read_csv("results/smollm_predictions_raw.csv")

prompts = ["direct", "formal", "concise", "analyst", "definition"]

df["sensitivity"] = df[prompts].apply(
    lambda x: x[x != -1].nunique(),
    axis=1
)

print(df["sensitivity"].value_counts().sort_index())
print("Sensitive:", (df["sensitivity"] > 1).sum())
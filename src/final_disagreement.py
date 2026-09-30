import pandas as pd

files = {
    "Qwen": "results/qwen_predictions_raw.csv",
    "SmolLM2": "results/smollm_predictions_raw.csv"
}

prompts = ["direct", "formal", "concise", "analyst", "definition"]

for model, file in files.items():
    df = pd.read_csv(file)

    def check(row):
        x = row[prompts]
        x = x[x != -1]
        return x.nunique() > 1

    df["disagreement"] = df.apply(check, axis=1)

    print(model)
    print("Disagreement:", df["disagreement"].mean())
    print("Count:", df["disagreement"].sum())
    print()
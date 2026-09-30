import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

files = {
    "Qwen": "results/qwen_predictions_raw.csv",
    "SmolLM2": "results/smollm_predictions_raw.csv"
}

prompts = ["direct", "formal", "concise", "analyst", "definition"]

rows = []

for model, file in files.items():
    df = pd.read_csv(file)

    for prompt in prompts:
        x = df[df[prompt] != -1]

        rows.append({
            "model": model,
            "prompt": prompt,
            "accuracy": accuracy_score(x["label"], x[prompt]),
            "macro_f1": f1_score(x["label"], x[prompt], average="macro"),
            "valid": len(x),
            "invalid": len(df) - len(x)
        })

result = pd.DataFrame(rows)

print(result)
result.to_csv("results/final_metrics.csv", index=False)
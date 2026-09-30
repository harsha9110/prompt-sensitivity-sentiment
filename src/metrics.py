import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("results/smollm_predictions_raw.csv")

for prompt in ["direct", "formal", "concise", "analyst", "definition"]:
    x = df[df[prompt] != -1]

    acc = accuracy_score(x["label"], x[prompt])
    f1 = f1_score(x["label"], x[prompt], average="macro")

    print(prompt)
    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print()
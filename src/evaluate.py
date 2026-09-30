import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(df):
    rows = []
    for (model, prompt_id), g in df.groupby(["model", "prompt_id"]):
        rows.append({
            "model": model,
            "prompt_id": prompt_id,
            "accuracy": accuracy_score(g["gold"], g["prediction"]),
            "macro_f1": f1_score(g["gold"], g["prediction"], average="macro")
        })
    return pd.DataFrame(rows)

def instance_sensitivity(df):
    p = df.pivot_table(index=["model", "example_id"], columns="prompt_id",
                       values="prediction", aggfunc="first")
    p["changed"] = p.nunique(axis=1) > 1
    return p.reset_index()

from datasets import load_dataset
import pandas as pd

data = load_dataset("stanfordnlp/sst2")
df = pd.DataFrame(data["validation"])

df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})

print(df.head())
print(df.shape)
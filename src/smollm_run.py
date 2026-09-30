import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from prompts import prompts
from extract import extract_prediction

name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

data = load_dataset("stanfordnlp/sst2", split="validation")
os.makedirs("results", exist_ok=True)

results = []

for i, item in enumerate(data):
    texts = [p.format(sentence=item["sentence"]) for p in prompts.values()]
    messages = [[{"role": "user", "content": t}] for t in texts]

    texts = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=40)

    row = {"label": item["label"]}

    for j, prompt_name in enumerate(prompts):
        result = tokenizer.decode(
            output[j][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        row[prompt_name] = extract_prediction(result)
        row[prompt_name + "_raw"] = result

    results.append(row)

    if (i + 1) % 50 == 0:
        print(i + 1)

pd.DataFrame(results).to_csv(
    "results/smollm_predictions_raw.csv",
    index=False
)

print("Done")
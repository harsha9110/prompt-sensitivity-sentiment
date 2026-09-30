# Client 02 — Prompt Sensitivity & Sentiment Analysis

**Working title:** How Sensitive Are Instruction-Tuned LLMs to Prompt Formulation? An Empirical Study of Zero-Shot Sentiment Classification

**RQ:** How much can semantically equivalent prompt formulations change zero-shot sentiment classification performance?

**H1:** Semantically equivalent prompt formulations can produce different sentiment predictions and therefore different measured performance.

**Dataset:** SST-2 validation subset.

**Models:** Qwen/Qwen2.5-3B-Instruct and mistralai/Mistral-7B-Instruct-v0.3.

**Prompt conditions:** five semantically equivalent formulations: direct, formal, concise, analyst, and definition-based.

**Metrics:** Accuracy, Macro-F1, per-prompt performance, prediction disagreement, and instance-level sensitivity.

For reproducibility, record model, prompt ID, example ID, gold label, raw output, parsed label, decoding parameters and seed.

Run:
```bash
pip install -r requirements.txt
python src/run_experiment.py
python src/evaluate.py
```

Do not place expected or fabricated numerical results in the poster; report measured results only.

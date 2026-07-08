"""
One-time script: extract training pairs from BenHalluEval QA dataset.
Produces benhallueval_training.json in the same format as dataset samples.json
so it can be concatenated during BanglaBERT training.

Faithful pairs:   (question, correct_answer, label=1)
Hallucinated:     (question, model_answer, label=0) where judge score == 0
"""
import json
import pandas as pd

SRC = "curious_insight/BanglaHalluEval Datasets/banglahallueval_qa_dataset.csv"
OUT = "benhallueval_training.json"

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} rows")

pairs = []

# 1. Faithful pairs — question → correct_answer
for _, r in df.iterrows():
    ca = str(r.get("correct_answer", "")).strip()
    if ca and ca.lower() not in ("nan", "null", ""):
        pairs.append({
            "context": "[NULL]",
            "prompt_bn": str(r["question"]),
            "response_bn": ca,
            "label": 1,
        })

# 2. Hallucinated pairs — question → model_answer where score == 0
MODELS = [
    ("deepseek_answer", "deepseek_score"),
    ("gemma_answer", "gemma_score"),
    ("qwen_answer", "qwen_score"),
]
for ans_col, score_col in MODELS:
    for _, r in df.iterrows():
        score = r.get(score_col, 1)
        ans = str(r.get(ans_col, "")).strip()
        if score == 0 and ans and ans.lower() not in ("nan", "null", ""):
            pairs.append({
                "context": "[NULL]",
                "prompt_bn": str(r["question"]),
                "response_bn": ans,
                "label": 0,
            })

faithful = sum(1 for p in pairs if p["label"] == 1)
hallu = sum(1 for p in pairs if p["label"] == 0)
print(f"Extracted: {faithful} faithful + {hallu} hallucinated = {len(pairs)} total")

# Balance: keep at most 2x hallucinated per faithful
max_hallu = faithful * 2
if hallu > max_hallu:
    import random
    random.seed(42)
    hallu_pairs = [p for p in pairs if p["label"] == 0]
    faith_pairs = [p for p in pairs if p["label"] == 1]
    random.shuffle(hallu_pairs)
    hallu_pairs = hallu_pairs[:max_hallu]
    pairs = faith_pairs + hallu_pairs
    print(f"Balanced: {faithful} faithful + {len(hallu_pairs)} hallucinated = {len(pairs)} total")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
print(f"Saved → {OUT}")
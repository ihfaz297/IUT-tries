#!/usr/bin/env python3
"""Evaluate ground truth reasoning chains using four Ollama models.

Reads: Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv
Saves separate result files per model to: Reasoning/Results/

Usage:
    python "Reasoning/Evaluation Script/Evaluation Script for hallucinated 1000/evaluate_reasoning_groundtruth_mistral_llama.py"
"""

import pandas as pd
import json
import re
import requests
import time
import os
import argparse
from tqdm import tqdm

INPUT_FILE = "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv"
RESULTS_DIR = "Reasoning/Results"

MODELS = [
    ("qwen2.5:32b-instruct", "reasoning_evaluation_scored_groundtruth_qwen2_5_32b.csv"),
    ("gemma2:27b",           "reasoning_evaluation_scored_groundtruth_gemma2_27b.csv"),
    ("mistral-nemo:latest",  "reasoning_evaluation_scored_groundtruth_mistral_nemo.csv"),
    ("llama3.1:8b",          "reasoning_evaluation_scored_groundtruth_llama3_1_8b.csv"),
]

PROMPT_TEMPLATE = """You are an expert evaluator for Bengali mathematical reasoning tasks.
Your task is to determine whether the given hallucinated_chain is hallucinated (i.e., incorrect or fabricated).

Question: {question}
Reasoning Chain: {hallucinated_chain}
Answer: {hallucinated_answer}

Is this hallucinated_chain hallucinated? Respond ONLY with a JSON object like this:
{{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}
Do not explain your reasoning or output anything else.
"""


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_label(raw: str) -> str:
    raw = strip_think_tags(raw).strip()

    try:
        obj = json.loads(raw)
        val = obj.get("is_hallucinated", "")
        if val.strip().lower() in ("yes", "no"):
            return val.strip().capitalize()
    except Exception:
        pass

    match = re.search(r'\{.*?"is_hallucinated"\s*:\s*"(Yes|No)".*?\}', raw, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).capitalize()

    lower = raw.lower()
    if "yes" in lower:
        return "Yes"
    if "no" in lower:
        return "No"

    return ""


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 32,
            "temperature": 0,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def run_model(model: str, output_file: str, base_url: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nLoading dataset...")
    df = pd.read_csv(INPUT_FILE)

    # map ground truth columns to expected names
    df = df.rename(columns={"question_id": "id"})
    df["hallucinated_chain"]  = df["answer"]
    df["hallucinated_answer"] = ""

    if os.path.exists(output_file):
        print(f"Resuming from {output_file}...")
        checkpoint_df = pd.read_csv(output_file)
        df = checkpoint_df
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    print(f"Total samples: {len(df)}")
    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].shape[0]
    print(f"Pending: {pending}")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row.get("is_hallucinated") in ("Yes", "No"):
            continue

        prompt = PROMPT_TEMPLATE.format(
            question=row["question"],
            hallucinated_chain=row["hallucinated_chain"],
            hallucinated_answer=row["hallucinated_answer"],
        )

        raw = call_ollama(prompt, model=model, base_url=base_url)
        label = extract_label(raw)

        if not label:
            time.sleep(1)
            raw = call_ollama(prompt, model=model, base_url=base_url)
            label = extract_label(raw)

        df.at[index, "is_hallucinated"] = label
        df.to_csv(output_file, index=False)

    print(f"Saved to {output_file}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL. Falls back to OLLAMA_BASE_URL env var, then http://localhost:11434",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run only specific models e.g. --models mistral-nemo:latest",
    )
    args = p.parse_args()

    base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    print(f"Ollama at: {base_url}")

    models_to_run = MODELS
    if args.models:
        models_to_run = [(m, f) for m, f in MODELS if m in args.models]

    for model, filename in models_to_run:
        output_file = os.path.join(RESULTS_DIR, filename)
        print(f"\n{'='*60}")
        print(f"Model : {model}")
        print(f"Output: {output_file}")
        print(f"{'='*60}")
        run_model(model=model, output_file=output_file, base_url=base_url)

    print("\nAll models done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate QA, Summarization, and Reasoning hallucinations using TigerLLM-9B-IT.

Saves results to:
  QA         → QA/Results/qa_4000_labeled_tigerllm.csv
  Summarization (3000) → Summarization/Evaluation_Results/summarization_3000_corrected_labeled_tigerllm.csv
  Summarization (1000) → Summarization/Evaluation_Results/summarization_1000_labeled_tigerllm.csv
  Reasoning (hallucinated) → Reasoning/Results/reasoning_evaluation_scored_tigerllm.csv
  Reasoning (ground truth) → Reasoning/Results/reasoning_evaluation_scored_groundtruth_tigerllm.csv

Usage:
    python scripts/evaluate_tigerllm.py --task all
    python scripts/evaluate_tigerllm.py --task qa
    python scripts/evaluate_tigerllm.py --task summ3000
    python scripts/evaluate_tigerllm.py --task summ1000
    python scripts/evaluate_tigerllm.py --task reasoning
    python scripts/evaluate_tigerllm.py --task reasoning_gt
"""

import csv
import os
import re
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise SystemExit("Run: pip install transformers torch accelerate")

MODEL_ID = "/workspace/TigerLLM"

# ── Prompts ────────────────────────────────────────────────────────────────────

QA_PROMPT = (
    "You are an evaluator.\n"
    "Decide whether the provided model answer is hallucinated relative to the question.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the question/context "
    "or is factually incorrect (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:"
)

SUMM_PROMPT = (
    "You are an evaluator.\n"
    "Decide whether the provided model summary is hallucinated relative to the document.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the summary contains information not supported by the document "
    "or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Answer now:"
)

REASONING_PROMPT = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n"
    "Your task is to determine whether the given hallucinated_chain is hallucinated "
    "(i.e., incorrect or fabricated).\n\n"
    "Question: {question}\n"
    "Reasoning Chain: {hallucinated_chain}\n"
    "Answer: {hallucinated_answer}\n\n"
    "Is this hallucinated_chain hallucinated? Respond ONLY with a JSON object like this:\n"
    '{"is_hallucinated": "Yes"} or {"is_hallucinated": "No"}\n'
    "Do not explain your reasoning or output anything else."
)

# ── Model loading ──────────────────────────────────────────────────────────────

BATCH_SIZE = 8  # process 8 samples at once


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def generate_batch(prompts: list, tokenizer, model, max_new_tokens: int = 16) -> list:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for out in outputs:
        decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
        results.append(decoded.strip())
    return results


def generate(prompt: str, tokenizer, model, max_new_tokens: int = 16) -> str:
    return generate_batch([prompt], tokenizer, model, max_new_tokens)[0]

# ── Label normalizers ──────────────────────────────────────────────────────────

def normalize_yesno(s: str) -> str:
    t = s.strip().lower()
    if t.startswith("y") or "yes" in t or "হ্যাঁ" in t:
        return "yes"
    if t.startswith("n") or "no" in t or "না" in t:
        return "no"
    return "unknown"


def normalize_reasoning(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    try:
        import json
        obj = json.loads(s)
        val = obj.get("is_hallucinated", "")
        if val.strip().lower() in ("yes", "no"):
            return val.capitalize()
    except Exception:
        pass
    match = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', s, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    if "yes" in s.lower():
        return "Yes"
    if "no" in s.lower():
        return "No"
    return ""

# ── Task runners ───────────────────────────────────────────────────────────────

def run_qa(tokenizer, model, resume: bool) -> None:
    input_file = "Hallucination Generated Answers/qa_4000.csv"
    output_file = "QA/Results/qa_4000_labeled_tigerllm.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed = set()
    if resume and Path(output_file).exists():
        with open(output_file, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id")
                if sid:
                    completed.add(sid)
        print(f"  Resuming — {len(completed)} rows done.")

    pending = [(i, r) for i, r in enumerate(rows)
               if not (resume and (r.get("id") or r.get("source_id") or str(i)) in completed)]

    write_header = not Path(output_file).exists() or Path(output_file).stat().st_size == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            prompts = [QA_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("hallucinated_answer", "") or r.get("model_answer", ""),
            ) for _, r in batch]
            raws = generate_batch(prompts, tokenizer, model)
            for (i, r), raw in zip(batch, raws):
                sid = r.get("id") or r.get("source_id") or str(i)
                label = normalize_yesno(raw)
                r_out = dict(r)
                r_out["is_hallucinated"] = label
                writer.writerow(r_out)
                print(f"  {i}: {sid} -> {label}")

    print(f"  Saved to {output_file}")


def run_summ(input_file: str, output_file: str, tokenizer, model, resume: bool) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed = set()
    if resume and Path(output_file).exists():
        with open(output_file, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id")
                if sid:
                    completed.add(sid)
        print(f"  Resuming — {len(completed)} rows done.")

    pending = [(i, r) for i, r in enumerate(rows)
               if not (resume and (r.get("id") or r.get("source_id") or str(i)) in completed)]

    write_header = not Path(output_file).exists() or Path(output_file).stat().st_size == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            prompts = [SUMM_PROMPT.format(
                document=r.get("document", "") or r.get("question", ""),
                summary=r.get("hallucinated_summary", "") or r.get("summary", "") or r.get("model_summary", ""),
            ) for _, r in batch]
            raws = generate_batch(prompts, tokenizer, model)
            for (i, r), raw in zip(batch, raws):
                sid = r.get("id") or r.get("source_id") or str(i)
                label = normalize_yesno(raw)
                r_out = dict(r)
                r_out["is_hallucinated"] = label
                writer.writerow(r_out)
                print(f"  {i}: {sid} -> {label}")

    print(f"  Saved to {output_file}")


def run_reasoning(input_file: str, output_file: str, tokenizer, model, resume: bool, is_groundtruth: bool = False) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["hallucinated_chain"] = df["answer"]
        df["hallucinated_answer"] = ""

    if os.path.exists(output_file) and resume:
        print(f"  Resuming from {output_file}...")
        checkpoint_df = pd.read_csv(output_file)
        df = checkpoint_df
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    print(f"  Total: {len(df)} | Pending: {(~df['is_hallucinated'].isin(['Yes','No'])).sum()}")

    for index, row in df.iterrows():
        if row.get("is_hallucinated") in ("Yes", "No"):
            continue
        prompt = REASONING_PROMPT.format(
            question=row["question"],
            hallucinated_chain=row["hallucinated_chain"],
            hallucinated_answer=row["hallucinated_answer"],
        )
        raw = generate(prompt, tokenizer, model, max_new_tokens=32)
        label = normalize_reasoning(raw)
        df.at[index, "is_hallucinated"] = label
        df.to_csv(output_file, index=False)
        print(f"  {index}: {row.get('id', index)} -> {label}")

    print(f"  Saved to {output_file}")

# ── Main ───────────────────────────────────────────────────────────────────────

TASKS = {
    "qa": lambda tok, mod, resume: run_qa(tok, mod, resume),
    "summ3000": lambda tok, mod, resume: run_summ(
        "Hallucination Generated Answers/summarization_3000_corrected.csv",
        "Summarization/Evaluation_Results/summarization_3000_corrected_labeled_tigerllm.csv",
        tok, mod, resume,
    ),
    "summ1000": lambda tok, mod, resume: run_summ(
        "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
        "Summarization/Evaluation_Results/summarization_1000_labeled_tigerllm.csv",
        tok, mod, resume,
    ),
    "reasoning": lambda tok, mod, resume: run_reasoning(
        "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv",
        "Reasoning/Results/reasoning_evaluation_scored_tigerllm.csv",
        tok, mod, resume, is_groundtruth=False,
    ),
    "reasoning_gt": lambda tok, mod, resume: run_reasoning(
        "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
        "Reasoning/Results/reasoning_evaluation_scored_groundtruth_tigerllm.csv",
        tok, mod, resume, is_groundtruth=True,
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["all", "qa", "summ3000", "summ1000", "reasoning", "reasoning_gt"],
        default="all",
        help="Which task to run",
    )
    p.add_argument("--resume", action="store_true", default=True, help="Resume from existing output")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    tokenizer, model = load_model()

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]

    for task in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        TASKS[task](tokenizer, model, args.resume)

    print("\nAll tasks done.")


if __name__ == "__main__":
    main()

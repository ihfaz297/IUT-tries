#!/usr/bin/env python3
"""CoT (Chain-of-Thought) hallucination evaluation for TigerLLM-9B-it.

TigerLLM is a Gemma3-architecture HuggingFace model (md-nishat-008/TigerLLM-9B-it),
not an Ollama model, so it can't go through evaluate_cot_ollama.py. This script
loads it once via transformers and reuses the exact same CoT prompts and label
parsers as the Ollama version, writing to matching _cot_ files with the
`tigerllm_9b` slug so results line up with the other models.

Requires an env with transformers>=4.50 (Gemma3 support), e.g. the `attention`
conda env. The model is loaded in bfloat16 (~18 GB) and is meant to run alone on
the GPU — wait for the Ollama CoT run to finish first to avoid VRAM contention.

Tasks (default: all):
  qa_hallu     → QA/Results/qa_cot_hallu_tigerllm_9b.csv
  qa_gt        → QA/Results/qa_cot_gt_tigerllm_9b.csv
  summ_hallu   → Summarization/Evaluation_Results/summ_3000_cot_tigerllm_9b.csv
  summ_gt      → Summarization/Evaluation_Results/summ_1000_cot_tigerllm_9b.csv
  reason_hallu → Reasoning/Results/reasoning_cot_tigerllm_9b.csv
  reason_gt    → Reasoning/Results/reasoning_gt_cot_tigerllm_9b.csv

Usage:
    python scripts/evaluate_cot_tigerllm.py --task all
    python scripts/evaluate_cot_tigerllm.py --task reason_hallu
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the exact CoT prompts and parsers from the Ollama script so the methodology
# is identical across models.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_cot_ollama import (  # noqa: E402
    QA_COT_PROMPT,
    SUMM_COT_PROMPT,
    REASONING_COT_PROMPT,
    parse_summ_label,
    parse_reasoning_label,
)

MODEL_ID = "md-nishat-008/TigerLLM-9B-it"
SLUG = "tigerllm_9b"

# ── Model wrapper ────────────────────────────────────────────────────────────

class TigerLLM:
    def __init__(self, model_id: str = MODEL_ID):
        print(f"Loading {model_id} (bfloat16)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print("Model loaded.\n")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Only decode the newly generated continuation.
        gen = out[0][inputs.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

# ── CSV task runner (mirrors evaluate_cot_ollama.run_csv_task) ────────────────

def run_csv_task(input_file, output_file, build_prompt_fn, parse_fn, llm,
                 max_new_tokens=512):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed = set()
    if Path(output_file).exists():
        with open(output_file, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id")
                lbl = r.get("is_hallucinated", "")
                if sid and lbl in ("yes", "no", "Yes", "No"):
                    completed.add(sid)
        print(f"  Resuming — {len(completed)} rows already labeled.")

    pending = [
        (i, r) for i, r in enumerate(rows)
        if (r.get("id") or r.get("source_id") or str(i)) not in completed
    ]
    print(f"  Pending: {len(pending)}")

    write_header = not Path(output_file).exists() or Path(output_file).stat().st_size == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, r in pending:
            sid = r.get("id") or r.get("source_id") or str(i)
            raw = llm.generate(build_prompt_fn(r), max_new_tokens)
            label = parse_fn(raw)
            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            f.flush()
            print(f"  {i}: {sid} -> {label}")

    print(f"  Saved → {output_file}\n")

# ── Reasoning task runner (mirrors evaluate_cot_ollama.run_reasoning_task) ────

def run_reasoning_task(input_file, output_file, is_groundtruth, llm,
                       max_new_tokens=512):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["chain"] = df["answer"]
        df["ans_col"] = ""
    else:
        df["chain"] = df["hallucinated_chain"]
        df["ans_col"] = df["hallucinated_answer"]

    if os.path.exists(output_file):
        print(f"  Resuming from {output_file}...")
        df = pd.read_csv(output_file)
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].index.tolist()
    print(f"  Total: {len(df)} | Pending: {len(pending)}")

    chain_col = "chain" if "chain" in df.columns else "hallucinated_chain"
    ans_col = "ans_col" if "ans_col" in df.columns else "hallucinated_answer"

    for idx in pending:
        prompt = REASONING_COT_PROMPT.format(
            question=df.at[idx, "question"],
            chain=df.at[idx, chain_col],
            answer=df.at[idx, ans_col],
        )
        raw = llm.generate(prompt, max_new_tokens)
        label = parse_reasoning_label(raw)
        df.at[idx, "is_hallucinated"] = label
        print(f"  {idx}: {df.at[idx, 'id']} -> {label}")
        df.to_csv(output_file, index=False)

    print(f"  Saved → {output_file}\n")

# ── Task definitions ──────────────────────────────────────────────────────────

def get_tasks(llm) -> dict:
    return {
        "qa_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            f"QA/Results/qa_cot_hallu_{SLUG}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("hallucinated_answer", ""),
            ),
            parse_summ_label, llm,
        ),
        # Ground truth is one row per question (1000), matching the other models.
        "qa_gt": lambda: run_csv_task(
            "QA/qa_gt_1000.csv",
            f"QA/Results/qa_cot_gt_{SLUG}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("correct_answer", ""),
            ),
            parse_summ_label, llm,
        ),
        "summ_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/summarization_3000_corrected.csv",
            f"Summarization/Evaluation_Results/summ_3000_cot_{SLUG}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("document", "") or r.get("question", ""),
                summary=r.get("hallucinated_summary", "") or r.get("summary", ""),
            ),
            parse_summ_label, llm,
        ),
        "summ_gt": lambda: run_csv_task(
            "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
            f"Summarization/Evaluation_Results/summ_1000_cot_{SLUG}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("question", ""),
                summary=r.get("summary", ""),
            ),
            parse_summ_label, llm,
        ),
        "reason_hallu": lambda: run_reasoning_task(
            "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv",
            f"Reasoning/Results/reasoning_cot_{SLUG}.csv",
            is_groundtruth=False, llm=llm,
        ),
        "reason_gt": lambda: run_reasoning_task(
            "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
            f"Reasoning/Results/reasoning_gt_cot_{SLUG}.csv",
            is_groundtruth=True, llm=llm,
        ),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["all", "qa_hallu", "qa_gt", "summ_hallu", "summ_gt",
                 "reason_hallu", "reason_gt"],
        default="all",
    )
    args = p.parse_args()

    llm = TigerLLM()
    tasks = get_tasks(llm)
    to_run = list(tasks.keys()) if args.task == "all" else [args.task]

    for task in to_run:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_ID} | Task: {task}")
        print(f"{'='*60}")
        tasks[task]()

    print("All done.")


if __name__ == "__main__":
    main()

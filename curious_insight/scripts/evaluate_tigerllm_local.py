#!/usr/bin/env python3
"""Evaluate hallucinations using TigerLLM-9B-IT — runs locally or on RunPod.

Tasks:
  qa_hallu     → QA/Results/qa_4000_tigerllm_hallu.csv
  qa_gt        → QA/Results/qa_4000_tigerllm_gt.csv
  summ_hallu   → Summarization/Evaluation_Results/summ_3000_tigerllm_hallu.csv
  summ_gt      → Summarization/Evaluation_Results/summ_1000_tigerllm_gt.csv
  reason_hallu → Reasoning/Results/reasoning_tigerllm_hallu.csv
  reason_gt    → Reasoning/Results/reasoning_tigerllm_gt.csv

Usage:
    # Run all tasks
    python scripts/evaluate_tigerllm_local.py --model-path ./TigerLLM --task all

    # Run one task
    python scripts/evaluate_tigerllm_local.py --model-path ./TigerLLM --task qa_hallu

    # Use 4-bit quantization (for GPUs with less than 18GB VRAM)
    python scripts/evaluate_tigerllm_local.py --model-path ./TigerLLM --task all --quantize
"""

import csv
import os
import re
import json
import argparse
import pandas as pd
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    raise SystemExit("Run: pip install transformers torch accelerate bitsandbytes")

BATCH_SIZE = 8

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
    "Your task is to determine whether the given reasoning chain is hallucinated "
    "(i.e., incorrect or fabricated).\n\n"
    "Question: {question}\n"
    "Reasoning Chain: {chain}\n"
    "Answer: {answer}\n\n"
    "Is this reasoning chain hallucinated? Respond ONLY with a JSON object like this:\n"
    '{"is_hallucinated": "Yes"} or {"is_hallucinated": "No"}\n'
    "Do not explain your reasoning or output anything else."
)

# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(model_path: str, quantize: bool):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        print("Using 4-bit quantization (requires bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return tokenizer, model


def generate_batch(prompts: list, tokenizer, model, max_new_tokens: int = 16) -> list:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]

# ── Label parsers ──────────────────────────────────────────────────────────────

def parse_yesno(s: str) -> str:
    t = s.strip().lower()
    if t.startswith("y") or "yes" in t or "হ্যাঁ" in t:
        return "yes"
    if t.startswith("n") or "no" in t or "না" in t:
        return "no"
    return "unknown"


def parse_reasoning(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    try:
        obj = json.loads(s)
        val = obj.get("is_hallucinated", "")
        if val.strip().lower() in ("yes", "no"):
            return val.capitalize()
    except Exception:
        pass
    m = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', s, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    if "yes" in s.lower():
        return "Yes"
    if "no" in s.lower():
        return "No"
    return ""

# ── Generic CSV runner ─────────────────────────────────────────────────────────

def run_csv_task(
    input_file: str,
    output_file: str,
    build_prompt_fn,
    parse_fn,
    tokenizer,
    model,
    resume: bool,
    max_new_tokens: int = 16,
) -> None:
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

        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start: batch_start + BATCH_SIZE]
            prompts = [build_prompt_fn(r) for _, r in batch]
            raws = generate_batch(prompts, tokenizer, model, max_new_tokens)
            for (i, r), raw in zip(batch, raws):
                sid = r.get("id") or r.get("source_id") or str(i)
                label = parse_fn(raw)
                r_out = dict(r)
                r_out["is_hallucinated"] = label
                writer.writerow(r_out)
                print(f"  {i}: {sid} -> {label}")

    print(f"  Saved → {output_file}\n")


def run_reasoning_task(
    input_file: str,
    output_file: str,
    is_groundtruth: bool,
    tokenizer,
    model,
    resume: bool,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["chain"] = df["answer"]
        df["hallucinated_answer"] = ""
    else:
        df["chain"] = df["hallucinated_chain"]

    if resume and os.path.exists(output_file):
        print(f"  Resuming from {output_file}...")
        ckpt = pd.read_csv(output_file)
        df = ckpt
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")

    else:
        df["is_hallucinated"] = ""

    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].index.tolist()
    print(f"  Total: {len(df)} | Pending: {len(pending)}")

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch_idx = pending[batch_start: batch_start + BATCH_SIZE]
        prompts = [
            REASONING_PROMPT.format(
                question=df.at[idx, "question"],
                chain=df.at[idx, "chain"],
                answer=df.at[idx, "hallucinated_answer"],
            )
            for idx in batch_idx
        ]
        raws = generate_batch(prompts, tokenizer, model, max_new_tokens=32)
        for idx, raw in zip(batch_idx, raws):
            label = parse_reasoning(raw)
            df.at[idx, "is_hallucinated"] = label
            print(f"  {idx}: {df.at[idx, 'id']} -> {label}")
        df.to_csv(output_file, index=False)

    print(f"  Saved → {output_file}\n")

# ── Task definitions ───────────────────────────────────────────────────────────

def get_tasks(args):
    return {
        "qa_hallu": lambda tok, mod: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            "QA/Results/qa_4000_tigerllm_hallu.csv",
            lambda r: QA_PROMPT.format(question=r.get("question",""), answer=r.get("hallucinated_answer","")),
            parse_yesno, tok, mod, resume=True,
        ),
        "qa_gt": lambda tok, mod: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            "QA/Results/qa_4000_tigerllm_gt.csv",
            lambda r: QA_PROMPT.format(question=r.get("question",""), answer=r.get("right_answer","")),
            parse_yesno, tok, mod, resume=True,
        ),
        "summ_hallu": lambda tok, mod: run_csv_task(
            "Hallucination Generated Answers/summarization_3000_corrected.csv",
            "Summarization/Evaluation_Results/summ_3000_tigerllm_hallu.csv",
            lambda r: SUMM_PROMPT.format(
                document=r.get("document","") or r.get("question",""),
                summary=r.get("hallucinated_summary","") or r.get("summary",""),
            ),
            parse_yesno, tok, mod, resume=True,
        ),
        "summ_gt": lambda tok, mod: run_csv_task(
            "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
            "Summarization/Evaluation_Results/summ_1000_tigerllm_gt.csv",
            lambda r: SUMM_PROMPT.format(
                document=r.get("question",""),
                summary=r.get("summary",""),
            ),
            parse_yesno, tok, mod, resume=True,
        ),
        "reason_hallu": lambda tok, mod: run_reasoning_task(
            "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv",
            "Reasoning/Results/reasoning_tigerllm_hallu.csv",
            is_groundtruth=False, tokenizer=tok, model=mod, resume=True,
        ),
        "reason_gt": lambda tok, mod: run_reasoning_task(
            "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
            "Reasoning/Results/reasoning_tigerllm_gt.csv",
            is_groundtruth=True, tokenizer=tok, model=mod, resume=True,
        ),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default="./TigerLLM",
        help="Path to downloaded TigerLLM model folder (default: ./TigerLLM)",
    )
    p.add_argument(
        "--task",
        choices=["all", "qa_hallu", "qa_gt", "summ_hallu", "summ_gt", "reason_hallu", "reason_gt"],
        default="all",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization — required if GPU has less than 18GB VRAM",
    )
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    tokenizer, model = load_model(args.model_path, args.quantize)
    tasks = get_tasks(args)
    to_run = list(tasks.keys()) if args.task == "all" else [args.task]

    for task in to_run:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        tasks[task](tokenizer, model)

    print("All done.")


if __name__ == "__main__":
    main()

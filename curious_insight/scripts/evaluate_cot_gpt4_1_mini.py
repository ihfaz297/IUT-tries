#!/usr/bin/env python3
"""CoT (Chain-of-Thought) hallucination evaluation using OpenAI GPT-4.1 mini.

Mirrors evaluate_cot_ollama.py but calls the OpenAI Responses API instead of Ollama.
API key is read from the .env file in the project root (OPENAI_API_KEY).

Tasks:
  qa_hallu    → QA/Results/qa_cot_hallu_gpt4_1_mini.csv          (4 000 rows)
  qa_gt       → QA/Results/qa_cot_gt_gpt4_1_mini.csv             (1 000 rows)
  summ_hallu  → Summarization/Evaluation_Results/summ_3000_cot_gpt4_1_mini.csv
  summ_gt     → Summarization/Evaluation_Results/summ_1000_cot_gpt4_1_mini.csv
  reason_hallu→ Reasoning/Results/reasoning_cot_gpt4_1_mini.csv  (1 000 rows)
  reason_gt   → Reasoning/Results/reasoning_gt_cot_gpt4_1_mini.csv

Resume / checkpoint behaviour
  Every row is flushed to disk immediately after being labeled.  If the script is
  stopped and re-run it detects the already-written rows and skips them, so no
  work is ever duplicated.

Usage:
    python scripts/evaluate_cot_gpt4_1_mini.py --task all
    python scripts/evaluate_cot_gpt4_1_mini.py --task qa_hallu
    python scripts/evaluate_cot_gpt4_1_mini.py --task summ_hallu
    python scripts/evaluate_cot_gpt4_1_mini.py --task reason_hallu
"""

import csv
import os
import re
import json
import argparse
import time
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError
except ImportError as exc:
    raise SystemExit("Missing dependency: openai.  Run: pip install openai") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("Missing dependency: pandas.  Run: pip install pandas") from exc

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL = "gpt-4.1-mini"
SLUG  = "gpt4_1_mini"

# ── CoT Prompts (identical to the Ollama script) ───────────────────────────────

QA_COT_PROMPT = (
    "You are an evaluator checking whether a model answer is hallucinated.\n\n"
    "Question: {question}\n"
    "Model Answer: {answer}\n\n"
    "Analyze step by step:\n"
    "Step 1: What factual claims does the answer make?\n"
    "Step 2: Are these claims supported by or inferable from the question context?\n"
    "Step 3: Based on steps 1-2, is the answer hallucinated?\n\n"
    "Final answer (write only this word on the last line): yes or no\n"
    "(yes = answer is hallucinated, no = answer is not hallucinated)"
)

SUMM_COT_PROMPT = (
    "You are an evaluator checking whether a summary is hallucinated relative to a document.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Analyze step by step:\n"
    "Step 1: List the key claims made in the summary.\n"
    "Step 2: For each claim, check whether it is directly supported by the document.\n"
    "Step 3: Based on steps 1-2, decide your final answer.\n\n"
    "Final answer (write only this word on the last line): yes or no\n"
    "(yes = summary is hallucinated, no = summary is not hallucinated)"
)

REASONING_COT_PROMPT = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n\n"
    "Question: {question}\n"
    "Reasoning Chain: {chain}\n"
    "Answer: {answer}\n\n"
    "Analyze step by step:\n"
    "Step 1: Is each calculation or logical step in the reasoning chain mathematically correct?\n"
    "Step 2: Does the final answer follow logically from the reasoning chain?\n"
    "Step 3: Based on steps 1-2, is this reasoning chain hallucinated (incorrect or fabricated)?\n\n"
    'Respond ONLY with a JSON object on the last line: {{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}'
)

# ── OpenAI call with retry / rate-limit handling ───────────────────────────────

def call_openai(prompt: str, client: OpenAI, max_tokens: int = 1024) -> str:
    """Call the OpenAI Responses API with exponential back-off on errors.

    Unlike the Ollama version we use the *Responses* API (client.responses.create)
    to stay consistent with the other GPT-4.1 mini scripts in this project.
    """
    backoff = 5
    attempts = 0
    while True:
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=max_tokens,
                temperature=0,
            )
            return (resp.output_text or "").strip()
        except RateLimitError as e:
            wait = backoff * (2 ** min(attempts, 4))
            print(f"  [RATE LIMIT] {e} - sleeping {wait}s ...")
            time.sleep(wait)
            attempts += 1
        except (APIConnectionError, APIError) as e:
            attempts += 1
            wait = backoff * min(attempts, 6)
            print(f"  [API ERROR] {e} (attempt {attempts}) - retrying in {wait}s ...")
            if attempts >= 10:
                print("  [GIVE UP] Too many failures, returning empty string.")
                return ""
            time.sleep(wait)
        except Exception as e:
            attempts += 1
            print(f"  [ERROR] Unexpected: {e} (attempt {attempts})")
            if attempts >= 5:
                return ""
            time.sleep(backoff)

# ── Label parsers (identical logic to the Ollama script) ──────────────────────

def parse_summ_label(raw: str) -> str:
    """Extract yes/no from a CoT response — look at the LAST occurrence."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    for line in reversed(text.splitlines()):
        line = line.strip().lower()
        if line in ("yes", "no"):
            return line
        if line.startswith("yes"):
            return "yes"
        if line.startswith("no"):
            return "no"
        if "final answer" in line or "step 3" in line:
            if "yes" in line:
                return "yes"
            if "no" in line:
                return "no"

    matches = list(re.finditer(r'\b(yes|no)\b', text.lower()))
    if matches:
        return matches[-1].group(1)

    print("  [WARN] Could not parse label, defaulting to yes")
    return "yes"


def parse_reasoning_label(raw: str) -> str:
    """Extract Yes/No from a CoT reasoning response."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    json_matches = list(re.finditer(r'\{[^{}]*"is_hallucinated"[^{}]*\}', text, re.IGNORECASE))
    if json_matches:
        last_json = json_matches[-1].group(0)
        try:
            obj = json.loads(last_json)
            val = obj.get("is_hallucinated", "").strip().lower()
            if val in ("yes", "no"):
                return val.capitalize()
        except Exception:
            pass
        m = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', last_json, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    try:
        obj = json.loads(text)
        val = obj.get("is_hallucinated", "").strip().lower()
        if val in ("yes", "no"):
            return val.capitalize()
    except Exception:
        pass

    for line in reversed(text.splitlines()):
        line_l = line.strip().lower()
        if "step 3" in line_l or "final" in line_l or "hallucinated" in line_l:
            if "yes" in line_l and "not" not in line_l:
                return "Yes"
            if "no" in line_l or "not hallucinated" in line_l:
                return "No"

    matches = list(re.finditer(r'\b(yes|no)\b', text.lower()))
    if matches:
        return matches[-1].group(1).capitalize()

    print("  [WARN] Could not parse label, defaulting to Yes")
    return "Yes"

# ── Generic CSV task runner ────────────────────────────────────────────────────

def run_csv_task(
    input_file: str,
    output_file: str,
    build_prompt_fn,
    parse_fn,
    client: OpenAI,
    max_tokens: int = 1024,
) -> None:
    """Process a CSV row-by-row, flushing results immediately for safe resumption."""
    root = Path(__file__).resolve().parent.parent
    inp  = root / input_file
    out  = root / output_file
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    # ── Resume: collect already-labeled IDs ───────────────────────────────────
    completed: set = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8", errors="replace") as f:
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
    print(f"  Total rows: {len(rows)} | Pending: {len(pending)}")

    write_header = not out.exists() or out.stat().st_size == 0

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for i, r in pending:
            sid = r.get("id") or r.get("source_id") or str(i)
            raw   = call_openai(build_prompt_fn(r), client, max_tokens=max_tokens)
            label = parse_fn(raw)
            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            f.flush()
            print(f"  {i}: {sid} -> {label}")

    print(f"  Saved -> {out}\n")


def run_reasoning_task(
    input_file: str,
    output_file: str,
    is_groundtruth: bool,
    client: OpenAI,
    max_tokens: int = 1024,
) -> None:
    """Process the reasoning CSV, saving after every row for safe resumption."""
    root = Path(__file__).resolve().parent.parent
    inp  = root / input_file
    out  = root / output_file
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["chain"]   = df["answer"]
        df["ans_col"] = ""
    else:
        df["chain"]   = df["hallucinated_chain"]
        df["ans_col"] = df["hallucinated_answer"]

    # ── Resume ────────────────────────────────────────────────────────────────
    if out.exists():
        print(f"  Resuming from {out} ...")
        ckpt = pd.read_csv(out)
        df = ckpt
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].index.tolist()
    print(f"  Total: {len(df)} | Pending: {len(pending)}")

    chain_col = "chain"   if "chain"   in df.columns else "hallucinated_chain"
    ans_col   = "ans_col" if "ans_col" in df.columns else "hallucinated_answer"

    for idx in pending:
        prompt = REASONING_COT_PROMPT.format(
            question=df.at[idx, "question"],
            chain=df.at[idx, chain_col],
            answer=df.at[idx, ans_col],
        )
        raw   = call_openai(prompt, client, max_tokens=max_tokens)
        label = parse_reasoning_label(raw)
        df.at[idx, "is_hallucinated"] = label
        print(f"  {idx}: {df.at[idx, 'id']} -> {label}")
        df.to_csv(out, index=False)   # full-flush after every row

    print(f"  Saved -> {out}\n")

# ── Task registry ──────────────────────────────────────────────────────────────

def get_tasks(client: OpenAI) -> dict:
    return {
        "qa_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            f"QA/Results/qa_cot_hallu_{SLUG}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("hallucinated_answer", ""),
            ),
            parse_summ_label,
            client,
        ),
        "qa_gt": lambda: run_csv_task(
            "QA/qa_gt_1000.csv",
            f"QA/Results/qa_cot_gt_{SLUG}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("correct_answer", ""),
            ),
            parse_summ_label,
            client,
        ),
        "summ_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/summarization_3000_corrected.csv",
            f"Summarization/Evaluation_Results/summ_3000_cot_{SLUG}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("document", "") or r.get("question", ""),
                summary=r.get("hallucinated_summary", "") or r.get("summary", ""),
            ),
            parse_summ_label,
            client,
        ),
        "summ_gt": lambda: run_csv_task(
            "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
            f"Summarization/Evaluation_Results/summ_1000_cot_{SLUG}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("question", ""),
                summary=r.get("summary", ""),
            ),
            parse_summ_label,
            client,
        ),
        "reason_hallu": lambda: run_reasoning_task(
            "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv",
            f"Reasoning/Results/reasoning_cot_{SLUG}.csv",
            is_groundtruth=False,
            client=client,
        ),
        "reason_gt": lambda: run_reasoning_task(
            "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
            f"Reasoning/Results/reasoning_gt_cot_{SLUG}.csv",
            is_groundtruth=True,
            client=client,
        ),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

TASK_ORDER = ["qa_hallu", "qa_gt", "summ_hallu", "summ_gt", "reason_hallu", "reason_gt"]

def main() -> None:
    p = argparse.ArgumentParser(
        description="CoT evaluation with GPT-4.1 mini (OpenAI API). "
                    "Fully resumable — re-run any task to continue from where it stopped."
    )
    p.add_argument(
        "--task",
        choices=["all"] + TASK_ORDER,
        default="all",
        help="Which task to run (default: all, in order)",
    )
    args = p.parse_args()

    # Load .env from the project root
    root = Path(__file__).resolve().parent.parent
    if load_dotenv is not None:
        load_dotenv(root / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set.  "
            "Add it to the .env file in the project root or export it."
        )

    client = OpenAI(api_key=api_key)
    tasks  = get_tasks(client)
    to_run = TASK_ORDER if args.task == "all" else [args.task]

    print(f"Model : {MODEL}  (slug={SLUG})")
    print(f"Tasks : {to_run}\n")

    for task in to_run:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        tasks[task]()

    print("All done.")


if __name__ == "__main__":
    main()

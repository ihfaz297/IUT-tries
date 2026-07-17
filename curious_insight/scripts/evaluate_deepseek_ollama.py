#!/usr/bin/env python3
"""Evaluate hallucinations using deepseek-r1:14b via Ollama.

Tasks:
  summ_hallu   → Summarization/Evaluation_Results/summ_3000_deepseek_hallu.csv
  reason_hallu → Reasoning/Results/reasoning_evaluation_scored_deepseek.csv
  reason_gt    → Reasoning/Results/reasoning_evaluation_scored_groundtruth_deepseek.csv

Parsing strategy (never returns blank):
  1. Strip <think>...</think>, try JSON then plain yes/no on remainder.
  2. If still empty, search inside <think> content for conclusion signals.
  3. Count yes/no signals across full response as tiebreaker.
  4. Hard default to "Yes" (conservative) if all else fails.

Usage:
    python scripts/evaluate_deepseek_ollama.py --task all
    python scripts/evaluate_deepseek_ollama.py --task summ_hallu
    python scripts/evaluate_deepseek_ollama.py --task reason_hallu
    python scripts/evaluate_deepseek_ollama.py --task reason_gt
"""

import csv
import os
import re
import json
import argparse
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

MODEL = "deepseek-r1:14b"

SUMM_PROMPT = (
    "You are an evaluator.\n"
    "Decide whether the provided model summary is hallucinated relative to the document.\n"
    "Only reply with a single word: yes or no.\n"
    "Interpretation: 'yes' means the summary contains information not supported by the document "
    "or contradicts it (hallucinated).\n"
    "Provide the answer in English only.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Answer now (yes or no):"
)

REASONING_PROMPT = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n"
    "Determine whether the given reasoning chain is hallucinated (incorrect or fabricated).\n\n"
    "Question: {question}\n"
    "Reasoning Chain: {chain}\n"
    "Answer: {answer}\n\n"
    "Is this reasoning chain hallucinated? "
    'Respond ONLY with a JSON object: {{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}\n'
    "Do not output anything else."
)


# ── Ollama call ────────────────────────────────────────────────────────────────

MAX_DOC_CHARS = 3500  # Bengali ≈ 2 tokens/char → 3500 chars ≈ 7000 tokens, safe under 8192 num_ctx


def call_ollama(prompt: str, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16384,  # Bengali tokenizes at ~2 tokens/char; 16384 covers doc+summary safely
            "num_predict": 1024,  # deepseek think chain can exceed 512 tokens before final yes/no
            "temperature": 0,
        },
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            if result:
                return result
            if attempt < 2:
                print(f"  [RETRY] Empty response, retrying ({attempt + 1}/3)...")
                time.sleep(5)
        except Exception as e:
            print(f"  [ERROR] Ollama request failed: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


# ── Label extractor (never returns blank) ─────────────────────────────────────

def _search_yes_no(text: str) -> Optional[str]:
    """Return 'yes'/'no' if found as a word boundary in text, else None."""
    t = text.lower()
    if re.search(r'\byes\b', t):
        return "yes"
    if re.search(r'\bno\b', t):
        return "no"
    return None


def extract_label(raw: str, json_mode: bool = False) -> str:
    """
    Parse Yes/No from a deepseek response.
    json_mode=True  → returns 'Yes'/'No'  (reasoning tasks)
    json_mode=False → returns 'yes'/'no'  (summarization task)
    Never returns empty string.
    """
    if not raw:
        result = "yes"
        print("  [WARN] Empty raw response, defaulting to yes/Yes")
        return "Yes" if json_mode else "yes"

    # 1. Strip think tags; work on the remainder first
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if json_mode:
        # Try full JSON parse
        try:
            obj = json.loads(cleaned)
            val = obj.get("is_hallucinated", "").strip().lower()
            if val in ("yes", "no"):
                return val.capitalize()
        except Exception:
            pass
        # Try JSON regex on cleaned
        m = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', cleaned, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        # Try JSON regex on full raw (in case JSON is inside think block)
        m = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', raw, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # 2. Plain yes/no word search on cleaned text
    found = _search_yes_no(cleaned)
    if found:
        return found.capitalize() if json_mode else found

    # 3. Search inside <think> content for conclusion signals
    think_blocks = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    think_text = " ".join(think_blocks).lower()

    conclusion_yes = re.search(
        r"(therefore|so|thus|hence|conclusion|final answer)[^\n]{0,60}(hallucinated|yes)\b",
        think_text,
    )
    conclusion_no = re.search(
        r"(therefore|so|thus|hence|conclusion|final answer)[^\n]{0,60}(not hallucinated|no)\b",
        think_text,
    )
    if conclusion_yes and not conclusion_no:
        print("  [FALLBACK] Inferred Yes from think conclusion")
        return "Yes" if json_mode else "yes"
    if conclusion_no and not conclusion_yes:
        print("  [FALLBACK] Inferred No from think conclusion")
        return "No" if json_mode else "no"

    # 4. Count signals across the full raw response
    r_lower = raw.lower()
    # "not hallucinated" counts as No signal, subtract from hallucinated count
    yes_count = (
        len(re.findall(r'\bhallucinated\b', r_lower))
        - len(re.findall(r'\bnot hallucinated\b', r_lower))
        + r_lower.count('"yes"')
    )
    no_count = (
        len(re.findall(r'\bnot hallucinated\b', r_lower))
        + r_lower.count('"no"')
    )

    if yes_count > no_count:
        print(f"  [FALLBACK] Signal count Yes={yes_count} No={no_count} → Yes")
        return "Yes" if json_mode else "yes"
    if no_count > yes_count:
        print(f"  [FALLBACK] Signal count Yes={yes_count} No={no_count} → No")
        return "No" if json_mode else "no"

    # 5. Hard default — conservative (flag uncertain as hallucinated)
    print(f"  [WARN] Could not determine label from response, defaulting to Yes")
    return "Yes" if json_mode else "yes"


# ── Summarization task ─────────────────────────────────────────────────────────

def run_summ_hallu(base_url: str) -> None:
    input_file = "Hallucination Generated Answers/summarization_3000_corrected.csv"
    output_file = "Summarization/Evaluation_Results/summ_3000_deepseek_hallu.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed: set = set()
    if Path(output_file).exists():
        with open(output_file, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id")
                lbl = r.get("is_hallucinated", "")
                if sid and lbl in ("yes", "no"):
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
            document = r.get("document", "") or r.get("question", "")
            if len(document) > MAX_DOC_CHARS:
                document = document[:MAX_DOC_CHARS] + "... [truncated]"
            summary = r.get("hallucinated_summary", "") or r.get("summary", "") or r.get("model_summary", "")
            sid = r.get("id") or r.get("source_id") or str(i)

            prompt = SUMM_PROMPT.format(document=document, summary=summary)
            raw = call_ollama(prompt, base_url)
            label = extract_label(raw, json_mode=False)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            print(f"  {i}: {sid} -> {label}")

    print(f"  Saved → {output_file}\n")


# ── Reasoning tasks ────────────────────────────────────────────────────────────

def run_reasoning(input_file: str, output_file: str, is_groundtruth: bool, base_url: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["chain"] = df["answer"]
        df["answer_field"] = ""
    else:
        df["chain"] = df["hallucinated_chain"]
        df["answer_field"] = df["hallucinated_answer"]

    if os.path.exists(output_file):
        print(f"  Resuming from {output_file}...")
        ckpt = pd.read_csv(output_file)
        df = ckpt
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].index.tolist()
    print(f"  Total: {len(df)} | Pending: {len(pending)}")

    for idx in pending:
        prompt = REASONING_PROMPT.format(
            question=df.at[idx, "question"],
            chain=df.at[idx, "chain"] if "chain" in df.columns else df.at[idx, "hallucinated_chain"],
            answer=df.at[idx, "answer_field"] if "answer_field" in df.columns else df.at[idx, "hallucinated_answer"],
        )
        raw = call_ollama(prompt, base_url)
        label = extract_label(raw, json_mode=True)
        df.at[idx, "is_hallucinated"] = label
        print(f"  {idx}: {df.at[idx, 'id']} -> {label}")
        df.to_csv(output_file, index=False)

    print(f"  Saved → {output_file}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

TASKS = {
    "summ_hallu": lambda url: run_summ_hallu(url),
    "reason_hallu": lambda url: run_reasoning(
        "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv",
        "Reasoning/Results/reasoning_evaluation_scored_deepseek.csv",
        is_groundtruth=False,
        base_url=url,
    ),
    "reason_gt": lambda url: run_reasoning(
        "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
        "Reasoning/Results/reasoning_evaluation_scored_groundtruth_deepseek.csv",
        is_groundtruth=True,
        base_url=url,
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["all", "summ_hallu", "reason_hallu", "reason_gt"],
        default="all",
    )
    p.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL. Falls back to OLLAMA_BASE_URL env var, then http://localhost:11434",
    )
    args = p.parse_args()

    base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    print(f"Ollama at: {base_url}")
    print(f"Model: {MODEL}\n")

    to_run = list(TASKS.keys()) if args.task == "all" else [args.task]

    for task in to_run:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        TASKS[task](base_url)

    print("All done.")


if __name__ == "__main__":
    main()

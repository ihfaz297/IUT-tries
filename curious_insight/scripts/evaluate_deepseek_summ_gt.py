#!/usr/bin/env python3
"""Evaluate ground truth summaries using deepseek-r1:14b via Ollama.

Input:  Sample Selection for Summ/banglahallueval_summarization_dataset_1000.csv
Output: Summarization/Evaluation_Results/summ_gt_1000_deepseek.csv

Columns: id, question (document), summary (ground truth)
Expected result: mostly "no" (ground truth summaries should not be hallucinated)

Usage:
    python scripts/evaluate_deepseek_summ_gt.py
    python scripts/evaluate_deepseek_summ_gt.py --ollama-url http://localhost:11434
"""

import csv
import os
import re
import argparse
import time
import requests
from pathlib import Path
from typing import Optional

MODEL = "deepseek-r1:14b"

MAX_DOC_CHARS = 3500  # Bengali ≈ 2 tokens/char → 3500 chars ≈ 7000 tokens, safe under 16384 num_ctx

INPUT_FILE  = "Sample Selection for Summ/banglahallueval_summarization_dataset_1000.csv"
OUTPUT_FILE = "Summarization/Evaluation_Results/summ_gt_1000_deepseek.csv"

SUMM_PROMPT = (
    "You are an evaluator.\n"
    "Decide whether the provided summary is hallucinated relative to the document.\n"
    "Only reply with a single word: yes or no.\n"
    "Interpretation: 'yes' means the summary contains information not supported by the document "
    "or contradicts it (hallucinated). 'no' means the summary is faithful to the document.\n"
    "Provide the answer in English only.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Answer now (yes or no):"
)


# ── Ollama call ────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16384,
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


# ── Label extractor ────────────────────────────────────────────────────────────

def _search_yes_no(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r'\byes\b', t):
        return "yes"
    if re.search(r'\bno\b', t):
        return "no"
    return None


def extract_label(raw: str) -> str:
    if not raw:
        print("  [WARN] Empty raw response, defaulting to yes")
        return "yes"

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    found = _search_yes_no(cleaned)
    if found:
        return found

    # Search inside <think> for conclusion signals
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
        print("  [FALLBACK] Inferred yes from think conclusion")
        return "yes"
    if conclusion_no and not conclusion_yes:
        print("  [FALLBACK] Inferred no from think conclusion")
        return "no"

    # Count signals
    r_lower = raw.lower()
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
        print(f"  [FALLBACK] Signal count Yes={yes_count} No={no_count} → yes")
        return "yes"
    if no_count > yes_count:
        print(f"  [FALLBACK] Signal count Yes={yes_count} No={no_count} → no")
        return "no"

    print("  [WARN] Could not determine label, defaulting to yes")
    return "yes"


# ── Main evaluation ────────────────────────────────────────────────────────────

def run(base_url: str) -> None:
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed: set = set()
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id")
                lbl = r.get("is_hallucinated", "")
                if sid and lbl in ("yes", "no"):
                    completed.add(sid)
        print(f"  Resuming — {len(completed)} rows already labeled.")

    pending = [(i, r) for i, r in enumerate(rows) if r.get("id") not in completed]
    print(f"  Pending: {len(pending)}")

    write_header = not Path(OUTPUT_FILE).exists() or Path(OUTPUT_FILE).stat().st_size == 0
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in pending:
            document = r.get("question", "")
            if len(document) > MAX_DOC_CHARS:
                document = document[:MAX_DOC_CHARS] + "... [truncated]"
            summary = r.get("summary", "")
            sid = r.get("id", str(i))

            prompt = SUMM_PROMPT.format(document=document, summary=summary)
            raw = call_ollama(prompt, base_url)
            label = extract_label(raw)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            print(f"  {i}: {sid} -> {label}")

    print(f"\n  Saved → {OUTPUT_FILE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL. Falls back to OLLAMA_BASE_URL env var, then http://localhost:11434",
    )
    args = p.parse_args()
    base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    print(f"Ollama at: {base_url}")
    print(f"Model:     {MODEL}")
    print(f"Input:     {INPUT_FILE}")
    print(f"Output:    {OUTPUT_FILE}\n")
    run(base_url)
    print("Done.")


if __name__ == "__main__":
    main()

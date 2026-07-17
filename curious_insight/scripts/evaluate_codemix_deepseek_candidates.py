#!/usr/bin/env python3
"""Label codemix hallucinated candidates using deepseek-r1:14b via Ollama.

Input:  Hallucination Generated Answers/codemix_4000.csv
Output: Codemix/Results/codemix_4000_candidates_deepseek.csv

Usage:
    python3 scripts/evaluate_codemix_deepseek_candidates.py --resume
"""

import csv
import time
import os
import re
from pathlib import Path
from typing import Optional

import requests

INPUT_FILE  = "Hallucination Generated Answers/codemix_4000.csv"
OUTPUT_FILE = "Codemix/Results/codemix_4000_candidates_deepseek.csv"
MODEL       = "deepseek-r1:14b"
OLLAMA_URL  = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = (
    "You are an evaluator.\n"
    "Decide whether the provided model answer is hallucinated relative to the context and question.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the context or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:"
)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(prompt: str) -> Optional[str]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 1024,
            "temperature": 0,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return strip_think(raw)
    except Exception:
        return None


def normalize_label(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    t = s.strip().lower()
    if t.startswith("y") or t == "yes":
        return "yes"
    if t.startswith("n") or t == "no":
        return "no"
    parts = t.split()
    if parts and parts[0] in ("yes", "no"):
        return parts[0]
    return "unknown"


def label_row(context: str, question: str, answer: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
    for attempt in range(3):
        resp = call_ollama(prompt)
        if resp:
            lab = normalize_label(resp)
            if lab != "unknown":
                return lab
        time.sleep(0.5 + attempt * 0.5)
    return "unknown"


def main() -> None:
    out = Path(OUTPUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed_ids: set = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                sid = row.get("id") or row.get("source_id")
                if sid and row.get("is_hallucinated") in ("yes", "no"):
                    completed_ids.add(sid)
        print(f"Resuming — {len(completed_ids)} rows already done.")

    pending = len(rows) - len(completed_ids)
    print(f"Total: {len(rows)} | Pending: {pending}")

    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in enumerate(rows):
            sid = r.get("id") or r.get("source_id") or str(i)
            if sid in completed_ids:
                continue

            context  = r.get("codemix_context", "")
            question = r.get("codemix_question", "")
            answer   = r.get("hallucinated_answer", "")

            label = label_row(context, question, answer)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            print(f"{i}: {sid} -> {label}")

    print(f"\nDone! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

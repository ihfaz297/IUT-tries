#!/usr/bin/env python3
"""Label hallucinations using a local Ollama Qwen model.

Reads a CSV with at least columns: id, question, hallucinated_answer
Sends each sample to the local Ollama instance (HTTP API at localhost:11434 by default)
or falls back to the `ollama` CLI if the HTTP API is unavailable.

Writes a new CSV with an added column `is_hallucinated` with values: yes / no / unknown

Usage (examples):
  python scripts/label_hallucinations_ollama.py \
    --input Results/hallucinated_answers_generation_qa.csv \
    --output Results/hallucinated_answers_generation_qa_with_labels.csv \
    --model qwen-2.5-32b --start 0 --end 400 --dry-run

This script is defensive: it retries calls, enforces the model to reply with a single token
("yes"/"no"), and maps some non-English answers as well.
"""

import csv
import argparse
import time
import subprocess
import shlex
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception:
    requests = None


PROMPT_TEMPLATE = (
    "You are an evaluator.\n"
    "Decide whether the provided model answer is hallucinated relative to the question.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the question/context or is factually incorrect (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:" 
)


def call_ollama_http(prompt: str, model: str = "mistral-nemo:latest", timeout: int = 30) -> Optional[str]:
    """Call local Ollama HTTP API at localhost:11434/api/generate.
    Returns the raw string response or None on failure.
    """
    if requests is None:
        return None

    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 8, "temperature": 0}}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        # Try to parse common response shapes
        j = r.json()
        # Ollama sometimes returns {'id':..., 'model':..., 'result':{'output': '...'}} or choices
        if isinstance(j, dict):
            # check common keys
            for key in ("response", "result", "output", "choices", "text"):
                if key in j:
                    v = j[key]
                    # nested
                    if isinstance(v, dict) and "output" in v:
                        return v["output"]
                    if isinstance(v, list) and v:
                        first = v[0]
                        if isinstance(first, dict) and "content" in first:
                            return first["content"]
                        if isinstance(first, str):
                            return first
                    if isinstance(v, str):
                        return v
            # as fallback, join textual values
            text = j.get("text") or j.get("output")
            if text:
                return text
        # fallback raw text
        return r.text
    except Exception:
        return None


def call_ollama_cli(prompt: str, model: str = "mistral-nemo:latest", timeout: int = 60) -> Optional[str]:
    """Fallback: call the `ollama` CLI if installed. Returns output or None.
    Uses the `ollama generate` form when available.
    """
    # Build a safe command
    cmd = f"ollama generate {shlex.quote(model)} --no-stream --prompt {shlex.quote(prompt)}"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or proc.stderr.strip()
    except Exception:
        return None


def normalize_label(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    t = s.strip().lower()
    # accept english yes/no
    if t.startswith("y") or t == "yes" or "হ্যাঁ" in t or "হ্যাঞ" in t:
        return "yes"
    if t.startswith("n") or t == "no" or "না" in t:
        return "no"
    # sometimes models return extra words; take first token
    first = t.split()[0]
    if first in ("yes", "no"):
        return first
    # accept bengali words
    if first in ("হ্যাঁ", "না"):
        return "yes" if first == "হ্যাঁ" else "no"
    return "unknown"


def label_row(question: str, answer: str, model: str, http_only: bool = False) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
    # Try HTTP API first
    for attempt in range(3):
        resp = call_ollama_http(prompt, model=model)
        if resp:
            lab = normalize_label(resp)
            if lab != "unknown":
                return lab
        time.sleep(0.5 + attempt * 0.5)

    if http_only:
        return "unknown"

    # Fallback to CLI
    for attempt in range(2):
        resp = call_ollama_cli(prompt, model=model)
        if resp:
            lab = normalize_label(resp)
            if lab != "unknown":
                return lab
        time.sleep(0.5)

    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--model", default="mistral-nemo:latest", help="Local Ollama model name")
    p.add_argument("--start", type=int, default=0, help="Start row (0-index)")
    p.add_argument("--end", type=int, default=None, help="End row (exclusive)")
    p.add_argument("--dry-run", action="store_true", help="Do not call model; only parse and write stub labels (unknown)")
    p.add_argument("--http-only", action="store_true", help="Only try HTTP API, don't fall back to CLI")
    p.add_argument("--retry-unknown", action="store_true", help="If input CSV already has labels, only retry 'unknown' rows")
    p.add_argument("--force-binary", action="store_true", help="Force output to only 'yes' or 'no' by retrying until a binary label is obtained")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    # Determine slice
    end = args.end if args.end is not None else len(rows)
    selected = rows[args.start:end]

    # Prepare output rows (copy original fields and add is_hallucinated)
    out_fieldnames = reader.fieldnames or []
    if "is_hallucinated" not in out_fieldnames:
        out_fieldnames.append("is_hallucinated")

    with out.open("w", newline="", encoding="utf-8") as ofh:
        writer = csv.DictWriter(ofh, fieldnames=out_fieldnames)
        writer.writeheader()

        for i, r in enumerate(selected, start=args.start):
            sample_id = r.get("id") or r.get("source_id") or str(i)
            question = r.get("question", "")
            # prefer explicit correct_answer field if present in this dataset
            answer = r.get("correct_answer") or r.get("hallucinated_answer") or r.get("model_answer") or r.get("answer") or ""
            current_label = r.get("is_hallucinated", "unknown")

            if args.dry_run:
                label = "unknown"
            elif args.retry_unknown and current_label not in ("unknown", "", None):
                # skip already labeled rows
                label = current_label
            else:
                # If force-binary is set, retry labeling until yes/no is returned (with a safety cap)
                if args.force_binary:
                    label = "unknown"
                    max_retries = 8
                    for attempt in range(max_retries):
                        label = label_row(question, answer, args.model, http_only=args.http_only)
                        if label in ("yes", "no"):
                            break
                        # slight backoff
                        time.sleep(0.5 + attempt * 0.2)
                    # as a last resort, if still unknown, map to 'no' conservatively
                    if label not in ("yes", "no"):
                        label = "no"
                else:
                    label = label_row(question, answer, args.model, http_only=args.http_only)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            if not (args.retry_unknown and current_label not in ("unknown", "", None)):
                print(f"{i}: {sample_id} -> {label}")

    print(f"Wrote labeled CSV to {out}")


if __name__ == "__main__":
    main()

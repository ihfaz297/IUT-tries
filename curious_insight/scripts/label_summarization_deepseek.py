#!/usr/bin/env python3
"""Label whether summarization outputs are hallucinated using deepseek-r1:14b via Ollama.

Reads a CSV with at least: id, document, hallucinated_summary.
Writes a new CSV with an added column `is_hallucinated`.

Usage:
    python scripts/label_summarization_deepseek.py \
        --input "Hallucination Generated Answers/summarization_3000_corrected.csv" \
        --output "Summarization/Evaluation_Results/summarization_3000_corrected_labeled_deepseek_r1_14b.csv" \
        --resume
"""

import csv
import argparse
import time
import os
import re
import requests
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROMPT_TEMPLATE = (
    "You are an evaluator.\n"
    "Decide whether the provided model summary is hallucinated relative to the document.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the summary contains information not supported by the document or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Answer now:"
)


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(prompt: str, model: str, base_url: str) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": -1,  # unlimited — deepseek needs long think chains
            "temperature": 0,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        cleaned = strip_think_tags(raw)
        print(f"  [DEBUG] raw={repr(raw[:120])}  cleaned={repr(cleaned)}")
        return cleaned
    except Exception as e:
        print(f"  [DEBUG] request error: {e}")
        return None


def normalize_label(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    t = s.strip().lower()
    if t.startswith("y") or t == "yes" or "হ্যাঁ" in t or "হ্যাঞ" in t:
        return "yes"
    if t.startswith("n") or t == "no" or "না" in t:
        return "no"
    parts = t.split()
    if parts:
        first = parts[0]
        if first in ("yes", "no"):
            return first
        if first in ("হ্যাঁ", "না"):
            return "yes" if first == "হ্যাঁ" else "no"
    return "unknown"


def label_row(document: str, summary: str, model: str, base_url: str) -> str:
    prompt = PROMPT_TEMPLATE.format(document=document, summary=summary)
    for attempt in range(3):
        resp = call_ollama(prompt, model=model, base_url=base_url)
        if resp:
            lab = normalize_label(resp)
            if lab != "unknown":
                return lab
        time.sleep(0.5 + attempt * 0.5)
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument(
        "--output",
        default="Summarization/Evaluation_Results/summarization_3000_corrected_labeled_deepseek_r1_14b.csv",
        help="Output CSV path",
    )
    p.add_argument("--model", default="deepseek-r1:14b", help="Ollama model name")
    p.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL. Falls back to OLLAMA_BASE_URL env var, then http://localhost:11434",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip already-labeled rows")
    p.add_argument("--force-binary", action="store_true", help="Force yes/no output")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    if load_dotenv is not None:
        load_dotenv(root / ".env")

    base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    print(f"Using Ollama at: {base_url}")
    print(f"Model: {args.model}")

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    end = args.end if args.end is not None else len(rows)
    selected = rows[args.start:end]

    out_fieldnames = list(fieldnames)
    if "is_hallucinated" not in out_fieldnames:
        out_fieldnames.append("is_hallucinated")

    completed_ids: set = set()
    if args.resume and out.exists():
        with out.open(newline="", encoding="utf-8", errors="replace") as existing_fh:
            for row in csv.DictReader(existing_fh):
                sid = row.get("id") or row.get("source_id")
                if sid:
                    completed_ids.add(sid)
        print(f"Resuming — {len(completed_ids)} rows already done.")

    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as ofh:
        writer = csv.DictWriter(ofh, fieldnames=out_fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in enumerate(selected, start=args.start):
            sample_id = r.get("id") or r.get("source_id") or str(i)
            if args.resume and sample_id in completed_ids:
                continue

            document = r.get("document", "")
            summary = (
                r.get("hallucinated_summary", "")
                or r.get("summary", "")
                or r.get("model_summary", "")
                or ""
            )

            if args.force_binary:
                label = "unknown"
                for attempt in range(8):
                    label = label_row(document, summary, args.model, base_url)
                    if label in ("yes", "no"):
                        break
                    time.sleep(0.5 + attempt * 0.2)
                if label not in ("yes", "no"):
                    label = "no"
            else:
                label = label_row(document, summary, args.model, base_url)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            print(f"{i}: {sample_id} -> {label}")

    print(f"Wrote labeled CSV to {out}")


if __name__ == "__main__":
    main()

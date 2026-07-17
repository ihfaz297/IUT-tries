#!/usr/bin/env python3
"""Label whether provided answers are hallucinated using OpenAI GPT-4.1 mini.

Reads a CSV with at least: id, question, correct_answer (or "correct answer").
Uses context if present. Writes a new CSV with an added column `is_hallucinated`.
"""

import csv
import argparse
import time
import os
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. Install with `pip install openai`.") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


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


def call_openai(prompt: str, model: str) -> Optional[str]:
    try:
        request_kwargs = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": 16,
            "temperature": 0,
        }
        response = CLIENT.responses.create(**request_kwargs)
        return response.output_text
    except Exception:
        return None


def normalize_label(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    t = s.strip().lower()
    if t.startswith("y") or t == "yes" or "হ্যাঁ" in t or "হ্যাঞ" in t:
        return "yes"
    if t.startswith("n") or t == "no" or "না" in t:
        return "no"
    first = t.split()[0]
    if first in ("yes", "no"):
        return first
    if first in ("হ্যাঁ", "না"):
        return "yes" if first == "হ্যাঁ" else "no"
    return "unknown"


def label_row(context: str, question: str, answer: str, model: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
    for attempt in range(3):
        resp = call_openai(prompt, model=model)
        if resp:
            lab = normalize_label(resp)
            if lab != "unknown":
                return lab
        time.sleep(0.5 + attempt * 0.5)
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    p.add_argument("--start", type=int, default=0, help="Start row (0-index)")
    p.add_argument("--end", type=int, default=None, help="End row (exclusive)")
    p.add_argument("--dry-run", action="store_true", help="Do not call model; write stub labels (unknown)")
    p.add_argument("--retry-unknown", action="store_true", help="Retry only rows with unknown labels")
    p.add_argument("--force-binary", action="store_true", help="Force output to only 'yes' or 'no'")
    p.add_argument("--resume", action="store_true", help="Resume from existing output by skipping labeled ids")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    if load_dotenv is not None:
        load_dotenv(root / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    global CLIENT
    CLIENT = OpenAI(api_key=api_key)

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    end = args.end if args.end is not None else len(rows)
    selected = rows[args.start:end]

    out_fieldnames = reader.fieldnames or []
    if "is_hallucinated" not in out_fieldnames:
        out_fieldnames.append("is_hallucinated")

    completed_ids = set()
    if args.resume and out.exists():
        with out.open(newline="", encoding="utf-8", errors="replace") as existing_fh:
            existing_reader = csv.DictReader(existing_fh)
            for row in existing_reader:
                existing_id = row.get("id") or row.get("source_id")
                if existing_id:
                    completed_ids.add(existing_id)

    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as ofh:
        writer = csv.DictWriter(ofh, fieldnames=out_fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in enumerate(selected, start=args.start):
            sample_id = r.get("id") or r.get("source_id") or str(i)
            if args.resume and sample_id in completed_ids:
                continue

            context = r.get("context", "")
            question = r.get("question", "")
            answer = (
                r.get("correct_answer")
                or r.get("correct answer")
                or r.get("model_answer")
                or r.get("answer")
                or ""
            )
            current_label = r.get("is_hallucinated", "unknown")

            if args.dry_run:
                label = "unknown"
            elif args.retry_unknown and current_label not in ("unknown", "", None):
                label = current_label
            else:
                if args.force_binary:
                    label = "unknown"
                    max_retries = 8
                    for attempt in range(max_retries):
                        label = label_row(context, question, answer, args.model)
                        if label in ("yes", "no"):
                            break
                        time.sleep(0.5 + attempt * 0.2)
                    if label not in ("yes", "no"):
                        label = "no"
                else:
                    label = label_row(context, question, answer, args.model)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            if not (args.retry_unknown and current_label not in ("unknown", "", None)):
                print(f"{i}: {sample_id} -> {label}")

    print(f"Wrote labeled CSV to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Label summarization hallucinations on the 1000-sample dataset using four Ollama models.

Input columns: id, question (document), summary
Models: qwen2.5:32b-instruct, gemma2:27b, llama3.1:8b, mistral-nemo:latest
Each model writes its own output file under Summarization/Evaluation_Results/.

Usage:
    python scripts/label_summarization_1000_four_models.py \
        --input "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv" \
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

MODELS = [
    ("qwen2.5:32b-instruct", "summarization_1000_labeled_qwen2_5_32b.csv"),
    ("gemma2:27b",           "summarization_1000_labeled_gemma2_27b.csv"),
    ("llama3.1:8b",          "summarization_1000_labeled_llama3_1_8b.csv"),
    ("mistral-nemo:latest",  "summarization_1000_labeled_mistral_nemo.csv"),
]


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(prompt: str, model: str, base_url: str) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": -1,
            "temperature": 0,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return strip_think_tags(raw)
    except Exception as e:
        print(f"  [ERROR] {e}")
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


def run_model(
    rows: list,
    fieldnames: list,
    model: str,
    out_path: Path,
    base_url: str,
    start: int,
    end: int,
    resume: bool,
    force_binary: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[start:end]

    out_fieldnames = list(fieldnames)
    if "is_hallucinated" not in out_fieldnames:
        out_fieldnames.append("is_hallucinated")

    completed_ids: set = set()
    if resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                sid = row.get("id")
                if sid:
                    completed_ids.add(sid)
        print(f"  Resuming — {len(completed_ids)} rows already done.")

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as ofh:
        writer = csv.DictWriter(ofh, fieldnames=out_fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in enumerate(selected, start=start):
            sample_id = r.get("id") or str(i)
            if resume and sample_id in completed_ids:
                continue

            document = r.get("question", "")   # this dataset uses 'question' as the document
            summary  = r.get("summary", "")

            if force_binary:
                label = "unknown"
                for attempt in range(8):
                    label = label_row(document, summary, model, base_url)
                    if label in ("yes", "no"):
                        break
                    time.sleep(0.5 + attempt * 0.2)
                if label not in ("yes", "no"):
                    label = "no"
            else:
                label = label_row(document, summary, model, base_url)

            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            print(f"  {i}: {sample_id} -> {label}")

    print(f"  Saved to {out_path}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument(
        "--output-dir",
        default="Summarization/Evaluation_Results",
        help="Directory to save per-model output CSVs",
    )
    p.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL. Falls back to OLLAMA_BASE_URL env var, then http://localhost:11434",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip already-labeled rows")
    p.add_argument("--force-binary", action="store_true", help="Force yes/no output")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run only specific models by name, e.g. --models qwen2.5:32b-instruct gemma2:27b",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    if load_dotenv is not None:
        load_dotenv(root / ".env")

    base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    print(f"Ollama at: {base_url}\n")

    inp = Path(args.input)
    with inp.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    end = args.end if args.end is not None else len(rows)
    out_dir = root / args.output_dir

    models_to_run = MODELS
    if args.models:
        models_to_run = [(m, f) for m, f in MODELS if m in args.models]

    for model, filename in models_to_run:
        out_path = out_dir / filename
        print(f"{'='*60}")
        print(f"Model : {model}")
        print(f"Output: {out_path}")
        print(f"{'='*60}")
        run_model(
            rows=rows,
            fieldnames=fieldnames,
            model=model,
            out_path=out_path,
            base_url=base_url,
            start=args.start,
            end=end,
            resume=args.resume,
            force_binary=args.force_binary,
        )

    print("All models done.")


if __name__ == "__main__":
    main()

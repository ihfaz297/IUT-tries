#!/usr/bin/env python3
"""Progress monitor for the CoT hallucination evaluation runs.

Reports, per model and task, how many rows have a valid label vs the total,
for both the Ollama models (evaluate_cot_ollama.py) and TigerLLM
(evaluate_cot_tigerllm.py). Also shows whether each run process is alive, what
it is currently working on (parsed from the logs), and GPU usage.

Usage:
    python scripts/monitor_cot_progress.py            # one-shot snapshot
    python scripts/monitor_cot_progress.py --watch 30 # refresh every 30s
    python scripts/monitor_cot_progress.py --pending   # only show unfinished
"""

import os
import sys
import time
import argparse
import subprocess

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (display name, slug, group)  group: "running" | "done" | "tigerllm"
MODELS = [
    ("qwen2.5:32b-instruct", "qwen2_5_32b",     "running"),
    ("gemma2:27b",           "gemma2_27b",      "running"),
    ("deepseek-r1:14b",      "deepseek_r1_14b", "running"),
    ("TigerLLM-9B",          "tigerllm_9b",     "tigerllm"),
    ("llama3.1:8b",          "llama3_1_8b",     "done"),
    ("mistral-nemo",         "mistral_nemo",    "done"),
]

# task -> (output path template, input file used to compute the total)
TASKS = {
    "qa_hallu":     ("QA/Results/qa_cot_hallu_{slug}.csv",
                     "Hallucination Generated Answers/qa_4000.csv"),
    "qa_gt":        ("QA/Results/qa_cot_gt_{slug}.csv",
                     "QA/qa_gt_1000.csv"),
    "summ_hallu":   ("Summarization/Evaluation_Results/summ_3000_cot_{slug}.csv",
                     "Hallucination Generated Answers/summarization_3000_corrected.csv"),
    "summ_gt":      ("Summarization/Evaluation_Results/summ_1000_cot_{slug}.csv",
                     "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv"),
    "reason_hallu": ("Reasoning/Results/reasoning_cot_{slug}.csv",
                     "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv"),
    "reason_gt":    ("Reasoning/Results/reasoning_gt_cot_{slug}.csv",
                     "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv"),
}

VALID = {"yes", "no"}
_total_cache: dict = {}


def total_for(input_rel: str) -> int:
    """Number of source rows for a task (cached)."""
    if input_rel not in _total_cache:
        try:
            _total_cache[input_rel] = len(pd.read_csv(os.path.join(BASE, input_rel)))
        except Exception:
            _total_cache[input_rel] = 0
    return _total_cache[input_rel]


def labeled_count(output_rel: str) -> int:
    """Rows in the output file that already carry a valid yes/no label."""
    path = os.path.join(BASE, output_rel)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    try:
        col = pd.read_csv(path, usecols=["is_hallucinated"])["is_hallucinated"]
    except Exception:
        return 0
    return int(col.astype(str).str.strip().str.lower().isin(VALID).sum())


def bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    frac = min(done / total, 1.0)
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def proc_alive(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def tail_current(log_rel: str) -> str:
    """Last 'Model: ... | Task: ...' header seen in a log, if any."""
    path = os.path.join(BASE, log_rel)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()[-400:]
    except Exception:
        return ""
    for line in reversed(lines):
        if line.startswith("Model:"):
            return line.strip()
    return ""


def gpu_status() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        used, tot, util = [x.strip() for x in out.split(",")]
        return f"{used} / {tot} MiB  |  util {util}%"
    except Exception:
        return "n/a"


def snapshot(only_pending: bool = False) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    ollama_up = proc_alive("evaluate_cot_ollama.py")
    tiger_up = proc_alive("evaluate_cot_tigerllm.py")
    waiter_up = proc_alive("run_cot_phase2_tigerllm.sh")

    print("=" * 78)
    print(f"CoT eval progress — {ts}")
    print("-" * 78)
    print(f"Ollama run : {'RUNNING' if ollama_up else 'stopped'}"
          f"   ({tail_current('logs/cot_ollama_run.log') or 'no header yet'})")
    if tiger_up:
        tiger_state = "RUNNING"
    elif waiter_up:
        tiger_state = "waiting for Phase 1"
    else:
        tiger_state = "stopped"
    print(f"TigerLLM   : {tiger_state}"
          f"   ({tail_current('logs/cot_tigerllm_run.log') or 'not started'})")
    print(f"GPU        : {gpu_status()}")
    print("=" * 78)

    grand_done = grand_total = 0
    for name, slug, group in MODELS:
        rows = []
        m_done = m_total = 0
        for task, (out_tpl, in_rel) in TASKS.items():
            total = total_for(in_rel)
            done = labeled_count(out_tpl.format(slug=slug))
            m_done += done
            m_total += total
            rows.append((task, done, total))
        grand_done += m_done
        grand_total += m_total

        complete = m_total > 0 and m_done >= m_total
        if only_pending and complete:
            continue

        pct = (100 * m_done / m_total) if m_total else 0
        tag = {"running": "▶", "tigerllm": "▶", "done": "✓"}.get(group, " ")
        print(f"\n{tag} {name}  [{slug}]   {m_done}/{m_total}  ({pct:.1f}%)")
        for task, done, total in rows:
            pct_t = (100 * done / total) if total else 0
            print(f"    {task:<13} {bar(done, total)} {done:>5}/{total:<5} {pct_t:5.1f}%")

    gp = (100 * grand_done / grand_total) if grand_total else 0
    print("\n" + "-" * 78)
    print(f"TOTAL: {grand_done}/{grand_total} labels  ({gp:.1f}%)")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", type=int, default=0,
                   help="refresh every N seconds (0 = one-shot)")
    p.add_argument("--pending", action="store_true",
                   help="only show models that are not yet complete")
    args = p.parse_args()

    if args.watch <= 0:
        snapshot(args.pending)
        return
    try:
        while True:
            os.system("clear")
            snapshot(args.pending)
            print(f"\n(refreshing every {args.watch}s — Ctrl-C to stop)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

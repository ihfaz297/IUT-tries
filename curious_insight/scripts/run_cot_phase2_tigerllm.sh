#!/usr/bin/env bash
# Phase 2: waits for Phase 1 to *cleanly complete* (logs/phase1.done sentinel),
# then runs the TigerLLM CoT eval. Keying off the sentinel — not process
# presence — avoids starting TigerLLM during a transient Phase 1 restart, which
# would fight for the GPU. Resumes from its own CSVs if interrupted.
set -u
cd "/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval"
PY="$HOME/anaconda3/envs/attention/bin/python"
LOG="logs/cot_tigerllm_run.log"
mkdir -p logs

echo "[phase2] $(date) waiting for Phase 1 completion (logs/phase1.done)..." | tee -a "$LOG"
while [ ! -f logs/phase1.done ]; do
    sleep 60
done
echo "[phase2] $(date) Phase 1 complete. Starting TigerLLM CoT." | tee -a "$LOG"

"$PY" -u scripts/evaluate_cot_tigerllm.py --task all >> "$LOG" 2>&1
echo "[phase2] $(date) TigerLLM CoT finished (rc=$?)." | tee -a "$LOG"

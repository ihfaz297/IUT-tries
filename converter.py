"""
Converter: Merges joggota_core.py + submission_pipeline.py into a single
self-contained Kaggle notebook. No external .py imports needed.

Usage:
  python converter.py

Output: submission_kaggle.ipynb
"""
import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Read both source files (normalize line endings for Windows compat)
with open("joggota_core.py", "r", encoding="utf-8") as f:
    joggota_code = f.read().replace("\r\n", "\n")

with open("submission_pipeline.py", "r", encoding="utf-8") as f:
    pipeline_code = f.read().replace("\r\n", "\n")

# Inline the joggota_core import — replace the import line with a comment
# and prepend the joggota_core source to the pipeline code
pipeline_code = pipeline_code.replace(
    "from joggota_core import extract_joggota_features\n",
    "# [joggota_core.py inlined below]\n",
)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# অলীকবচন — Joggota + Small LLM Judge Pipeline\n",
                "**Self-contained Kaggle notebook** for Datathon 2.0 Phase 1 & 2.\n\n",
                "### Architecture\n",
                "- **Phase 1**: mDeBERTa-v3 NLI + LaBSE embeddings + Joggota deterministic rules\n",
                "- **Phase 2**: Qwen2.5-1.5B-Instruct LLM Judge (no-context rows only)\n",
                "- **Phase 3**: XGBoost fusion (14 features → binary prediction)\n\n",
                "### Kaggle Setup Checklist:\n",
                "1. GPU: **T4 x2** or **P100** (16 GB VRAM minimum)\n",
                "2. Upload `dataset samples.json` and `test set.csv`\n",
                "3. For LLM Judge: upload Qwen2.5-1.5B-Instruct as a Kaggle Dataset → `/kaggle/input/qwen2.5-1.5b-instruct`\n",
                "   - OR download during Phase 1 via `download_models.py` before uploading\n",
                "4. For the cross-lingual check: upload NLLB-200-distilled-600M as a Kaggle Dataset → `/kaggle/input/nllb-200-distilled-600m`\n",
                "5. For Phase 2: toggle **Internet OFF** before running (all models must be offline)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q accelerate sentence-transformers xgboost\n"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 1: Joggota Engine (Deterministic Rules + Mini-RAG)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in joggota_code.split("\n")],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 2: Submission Pipeline (NLI + LLM Judge + XGBoost)\n\n",
                "See `implementation_plan.md` for the full strategy document.\n\n",
                "### Feature set:\n",
                "- `nli_ctx_entail`, `nli_ctx_contra` — mDeBERTa-v3 entailment/contradiction (bn)\n",
                "- `nli_en_entail`, `nli_en_contra`, `cross_lingual_disagreement` — same NLI check re-run on\n",
                "  the NLLB-translated English text; disagreement vs the bn verdict is the\n",
                "  \"correct in English, wrong in Bengali\" signal the organizers flagged as strongest\n",
                "- `sim_premise_response`, `xlingual_consistency` — LaBSE cosine similarity\n",
                "- `token_overlap_ctx_resp` — lexical Jaccard overlap\n",
                "- `has_context` — binary context flag\n",
                "- `word_entropy`, `char_entropy` — response randomness\n",
                "- `novel_char_ratio` — extrinsic hallucination signal\n",
                "- `length_ratio` — response vs prompt length ratio\n",
                "- `deterministic_joggota` — rule-based verdict (idioms, math, spelling)\n",
                "- `cultural_default_flag` — C1 band cultural default detection\n",
                "- `context_containment` — response bigrams verbatim in context\n",
                "- `resp_is_refusal`, `resp_code_switch_ratio`, `resp_repetition_score`, `resp_is_question` — response quality heuristics\n",
                "- `llm_judge_score` — Qwen2.5-1.5B logit-based faithfulness score (no-context rows)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in pipeline_code.split("\n")],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 3: Run Inference"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_and_predict()\n"
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("submission_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✅ Created submission_kaggle.ipynb (self-contained)")

# Quick sanity: verify the notebook JSON is valid
with open("submission_kaggle.ipynb", "r", encoding="utf-8") as f:
    json.load(f)
print("✅ Notebook JSON is valid")
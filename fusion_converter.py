"""
Converter: merges joggota_core.py + submission_pipeline.py + fusion_pipeline.py +
fusion_evaluate.py into one self-contained Kaggle notebook.

Care point: submission_pipeline.py ends with `if __name__ == "__main__": train_and_predict()`.
Inside a notebook cell __name__ IS "__main__", so inlining it verbatim would wrongly
auto-run the OLD standalone pipeline the moment that cell executes. That block is
stripped here -- we only want the Translator/NLIScorer/Embedder classes it defines.

Usage: python fusion_converter.py
Output: fusion_kaggle.ipynb
"""
import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("joggota_core.py", "r", encoding="utf-8") as f:
    joggota_code = f.read().replace("\r\n", "\n")

with open("submission_pipeline.py", "r", encoding="utf-8") as f:
    pipeline_code = f.read().replace("\r\n", "\n")

with open("fusion_pipeline.py", "r", encoding="utf-8") as f:
    fusion_code = f.read().replace("\r\n", "\n")

with open("fusion_evaluate.py", "r", encoding="utf-8") as f:
    evaluate_code = f.read().replace("\r\n", "\n")

# joggota_core: nothing to strip, it's pure definitions
# submission_pipeline: strip the joggota import (already inlined above) AND the
# __main__ guard (would wrongly auto-run the old pipeline inside a notebook cell)
pipeline_code = pipeline_code.replace(
    "from joggota_core import extract_joggota_features\n",
    "# [joggota_core.py inlined above]\n",
)
GUARD_RE = re.compile(r'if __name__ == "__main__":\n\s*train_and_predict\(\)\s*\Z')
new_pipeline_code, n_subs = GUARD_RE.subn(
    '# [__main__ guard stripped -- this notebook drives execution itself below,\n'
    '#  train_and_predict() is intentionally never called here]\n',
    pipeline_code,
)
assert n_subs == 1, (
    f"expected to strip exactly one __main__ guard, stripped {n_subs} -- "
    "check submission_pipeline.py's exact __main__ block text"
)
pipeline_code = new_pipeline_code
# belt-and-suspenders: no bare top-level call to train_and_predict() should survive
assert not re.search(r'^train_and_predict\(\)', pipeline_code, re.MULTILINE), \
    "a call to train_and_predict() survived the strip"

# fusion_pipeline: strip its imports of things already inlined above
fusion_code = fusion_code.replace(
    "from joggota_core import extract_joggota_features\n", "# [inlined above]\n",
)
fusion_code = fusion_code.replace(
    "from submission_pipeline import Translator, TRANSLATOR_CHECKPOINTS\n", "# [inlined above]\n",
)

def _banner_start_before(text, marker):
    """Index of the start of the '# ----' banner line immediately preceding `marker`,
    so a section's banner comment stays attached to the section it introduces."""
    idx = text.index(marker)
    return text.rindex("# ----------------------------------------------------------------------", 0, idx)


# THREE-way split, not two. Real bug found via a live Kaggle run: Part 2
# (submission_pipeline.py, inlined) does `from transformers import ...` at its
# top. The bootstrap (which upgrades transformers so gemma-4's architecture is
# recognized) was living inside the same cell as data-discovery, which ran
# AFTER Part 2 -- so by the time the upgrade ran, transformers was already
# imported and cached in sys.modules; the upgrade succeeded on disk but the
# already-loaded old module kept being used. Fix: bootstrap must be its own
# cell, positioned before ANYTHING imports transformers.
bootstrap_end = _banner_start_before(fusion_code, "# 2. DATA DISCOVERY")
bootstrap_code = fusion_code[:bootstrap_end]

lanes_start = _banner_start_before(fusion_code, "# 3. LANE 1a: SummaC-ZS windowed NLI")
fusion_data_code = fusion_code[bootstrap_end:lanes_start]
fusion_lanes_code = fusion_code[lanes_start:]

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# অলীকবচন Fusion: our cross-lingual pipeline + teammate's gemma-4/SummaC/math-solver pipeline\n",
                "\n",
                "Combines two independently-built approaches and picks the best-validated combination via\n",
                "proper StratifiedKFold OOF cross-validation -- no in-sample metrics, no threshold search\n",
                "double-dipped against the same data it was tuned on.\n\n",
                "**Needs**: GPU (2xT4 or P100), the `mdeberta-v3-xnli-multilingual` Kaggle dataset,\n",
                "and the `google/gemma-4/transformers/gemma-4-12b-it` Kaggle Model attached.\n",
                "Internet ON is fine for Phase 1 (our NLLB translator falls back to a HF download if no\n",
                "local checkpoint is attached); for Phase 2 compliance attach an NLLB dataset too.\n",
            ],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["!pip install -q accelerate sentence-transformers xgboost bitsandbytes kagglehub\n"],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": [
                "## Part 0.5: transformers bootstrap for gemma-4 -- MUST run before anything else\n",
                "imports `transformers` (Part 2 below does). If this runs after something has already\n",
                "`import transformers`'d, the upgrade succeeds on disk but the already-cached old module\n",
                "keeps being used for the rest of the session -- confirmed as a real failure mode, not\n",
                "theoretical (see CLAUDE.md's gemma-4/transformers gotcha).",
            ],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in bootstrap_code.split("\n")],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": ["## Part 1: Joggota Engine (deterministic rules + Form Engine, bug-fixed)"],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in joggota_code.split("\n")],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": ["## Part 2: our model classes (Translator/NLIScorer/Embedder/SmallLLMJudge definitions only --\n",
                        "the old `train_and_predict()` pipeline is intentionally never invoked in this notebook)"],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in pipeline_code.split("\n")],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": ["## Part 2.6: load competition data (needed before the smoke test below)"],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in fusion_data_code.split("\n")],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": [
                "## Part 2.7: gemma-4 smoke test (run this BEFORE Part 3)\n\n",
                "Loads gemma-4 and scores just 3 rows. Part 3 runs the NLI lane, math solver, "
                "AND gemma-4 in one cell -- if gemma-4 fails there, you've already waited through "
                "the full NLI lane on 2,815 rows for nothing. This catches a load/version failure "
                "in under a minute instead.",
            ],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [
                "# ---------- smoke test: does gemma-4 load and answer sanely on a tiny sample? ----------\n",
                "_smoke = train.head(3).reset_index(drop=True)\n",
                "(_pA, _pB), _ = run_gemma_judge(_smoke, _smoke)\n",
                "print('pA (bn judge, P(faithful)):', _pA)\n",
                "print('pB (en judge, P(faithful)):', _pB)\n",
                "print('true labels:                ', _smoke['label'].values)\n",
                "print()\n",
                "print('If pA/pB look like plausible probabilities (not all 0.5, not NaN, not errored) '\n",
                "      'and roughly track the true labels on this tiny sample, gemma-4 is working. '\n",
                "      'Safe to run Part 3 now.')\n",
            ],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": ["## Part 3: fusion feature extraction (both lanes, writes fusion_{train,test}_features.pkl)"],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in fusion_lanes_code.split("\n")],
        },
        {
            "cell_type": "markdown", "metadata": {},
            "source": ["## Part 4: honest OOF evaluation across feature-set x model combinations, then submission"],
        },
        {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in evaluate_code.split("\n")],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("fusion_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Created fusion_kaggle.ipynb")

with open("fusion_kaggle.ipynb", "r", encoding="utf-8") as f:
    json.load(f)
print("Notebook JSON is valid")

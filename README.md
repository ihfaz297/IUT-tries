# অলীকবচন — Submission Package

Final submission for Datathon 2.0 (Bengali LLM Hallucination Detection). Self-contained for
both phases; full development history lives on the `main` branch.

## Contents

| Path | What it is |
|---|---|
| `final_submission.ipynb` | The pipeline: gemma-4-31B bilingual judge + SummaC windowed NLI + 5-source ground-truth matcher + hidden-state probe + honest StratifiedKFold OOF evaluation (hallucinated-class F1, `pos_label=0`) |
| `wheelhouse/` | Linux/py3.11 wheels for every runtime `pip install` — includes `transformers 5.14.1` (first PyPI release that recognizes `gemma4_unified`). Upload as a Kaggle Dataset for offline Phase 2 |
| `matcher_caches/` | The 5 ground-truth QA sources (105,262 pairs): NCTB-QA, TyDi QA-bn, IndicQA-bn, BanglaRQA, bangla-mmlu. Upload as a Kaggle Dataset for offline Phase 2 |

## Phase 1 (internet ON)

Just run the notebook. Everything auto-downloads; nothing needs attaching.

## Phase 2 (offline)

The notebook's first code cell probes connectivity and flips itself into offline mode
(`HF_HUB_OFFLINE=1` + a download kill-switch) automatically. Attach as Kaggle inputs:

1. A dataset made from `wheelhouse/` — satisfies both pip cells, zero network.
2. A dataset made from `matcher_caches/` — satisfies all 5 matcher loaders (checked via
   `/kaggle/input/**` glob before any download is ever attempted).
3. gemma-4 weights (Kaggle Model mount, or a dataset copy whose dir name matches
   `*gemma*4*31b*` / `*gemma*4*12b*`).
4. `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` model files (dir name
   containing `mdeberta`).
5. `facebook/nllb-200-distilled-600M` model files (dir name containing `nllb`).

Full details in the checklist cell at the top of the notebook.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**অলীকবচন** — Bengali LLM Hallucination Detection Challenge (Datathon 2.0, IUT / IPD / Brain Lab).

Binary classification: given a Bengali prompt and a candidate response (plus an optional context passage), predict whether the response is **faithful (label=1)** or **hallucinated (label=0)**. Evaluation metric is **macro F1** (binary F1 on the hallucinated class).

Two-phase competition:
- **Phase 1**: Submit prediction CSVs to Kaggle. Public + private leaderboard scoring.
- **Phase 2**: Top 30 teams submit a runnable solution package. Runs **offline in a Kaggle kernel — open-weight models only, no paid APIs**. Build Phase 2-compliant from day one.

## Current Pipeline Architecture

The pipeline is implemented in `submission_pipeline.py` and runs in **3 sequential phases** designed to never exceed a Kaggle T4 GPU's 16GB VRAM:

### Phase 1 — NLI + Embeddings (all rows)
- **mDeBERTa-v3** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`): Produces entailment/contradiction probabilities between context (or prompt) and response.
- **LaBSE** (`sentence-transformers/LaBSE`): Cosine similarity between context/prompt embedding and response embedding.
- **Joggota Engine** (`joggota_core.py`): Deterministic rule-based features (entropy, length ratio, novel char ratio, task classification, idiom detection, corpus grounding).
- After features are extracted, **both models are deleted and GPU memory is flushed** (`gc.collect()` + `torch.cuda.empty_cache()`).

### Phase 2 — Feature Signal (no LLM judge)
- **NO TigerLLM**: Attempted 4-bit quantization on Kaggle T4 — OOM errors. The 9B model needs ~8GB at 4-bit and T4 has 16GB but with tokenizer overhead + VRAM fragmentation, it breaks. This is a dead end on current hardware.
- **NO BanglaBERT**: Attempted as a fallback — fine-tuned on 2.5K BenHalluEval QA pairs + 299 competition samples. CV F1 showed 0.823 but actual leaderboard score dropped to 0.602. The model learned a cheap shortcut (detecting non-Bengali tokens in BenHalluEval's gibberish hallucinated samples) rather than real faithfulness discrimination. Made things worse, not better.
- **Current fallback**: Uses `sim_premise_response` (LaBSE cosine similarity) as the proxy faithfulness signal in place of the failed LLM judge.

### Phase 3 — XGBoost Fusion
- **XGBoost** (`max_depth=3`, `n_estimators=300`): Shallow ensemble to prevent overfitting on 299-sample training set.
- Feature set (12 features total):
  - `nli_ctx_entail`, `nli_ctx_contra` — NLI scores
  - `sim_premise_response` — LaBSE cosine similarity
  - `token_overlap_ctx_resp` — Lexical overlap
  - `has_context` — Binary flag
  - `word_entropy`, `char_entropy` — Response randomness
  - `novel_char_ratio` — Extrinsic hallucination signal
  - `length_ratio` — Response vs prompt length
  - `deterministic_joggota` — Rule-based verdict
  - `corpus_match_score` — Offline corpus retrieval grounding
  - `banglabert_faithful_prob` — Currently filled with `sim_premise_response` (proxy fallback)

## File Map

| File | Description |
|---|---|
| `submission_pipeline.py` | **Main inference pipeline** — 3-phase architecture |
| `joggota_core.py` | Deterministic rules engine + Form Engine + Mini-RAG retriever |
| `build_corpus.py` | Wikipedia crawler — builds `offline_corpus.json` |
| `offline_corpus.json` | ~242 paragraphs from Bengali Wikipedia (Constitution, history, literature, etc.) |
| `converter.py` | Converts `submission_pipeline.py` → `submission_kaggle.ipynb` |
| `submission_kaggle.ipynb` | Auto-generated Kaggle-ready notebook |
| `train_banglabert.py` | ❌ Failed BanglaBERT fine-tuning script (do not use — makes scores worse) |
| `augment_from_benhallueval.py` | ❌ Failed data augmentation — extracts noisy training pairs from BenHalluEval |
| `benhallueval_training.json` | ❌ Noisy QA pairs — 5K hallucinated, 2.5K faithful (caused the 0.823→0.602 drop) |
| `dataset samples.json` | 299 labeled examples — the only supervised signal |
| `test set.csv` | 2,516 rows, no labels — submit predictions for these |
| `sample submission.csv` | Format: `id,label` |
| `question_in_hand.txt` | Competition overview |
| `some_notes.txt` | Organizer notes — winning approach hints |
| `some_catches.txt` | Phase 2 preview — source domains, corpus question, methodology refs |
| `LEGAL_DOCS.txt` | Full competition rules |
| `freelance.txt` | Example of cultural stereotype hallucination (Humayun Ahmed vs Taslima Nasrin) |
| `check_noise.py` | Diagnostic: spot-checks BenHalluEval training data quality |
| `check_submission.py` | Diagnostic: compares submission distributions across runs |
| `diagnose.py` | Diagnostic: analyzes CV F1 vs actual score gap |
| `run_log.csv` | Previous run log (CV F1 0.646, TigerLLM fallback) |
| `run_log (1).csv` | Failed run log (CV F1 0.823, BanglaBERT — actual score 0.602) |

## Joggota Engine (`joggota_core.py`)

### Form Engine Features
- `word_entropy` / `char_entropy` — verbose hallucinations have higher entropy
- `novel_char_ratio` — characters in response not in context (extrinsic hallucination)
- `length_ratio` — hallucinated responses are ~69% longer
- `resp_len` — raw response length

### Task Classifier
Routes prompts into: `idiom`, `vocabulary`, `spelling`, `math`, `translation`, `grammar`, `factual`.

### Deterministic Rules
- **Spelling**: Response >5 words = hallucination (should be a single corrected word)
- **Math**: No digits in response = hallucination
- **Idioms**: Hardcoded dictionary of 9 common বাগধারা with literal vs figurative keyword checks. If LLM gives literal meaning instead of figurative → instant `0.0`.

### Mini-RAG (Corpus Retriever)
- Loads `offline_corpus.json` at import time (singleton `_corpus_retriever`).
- Pure TF-IDF retrieval — no heavy dependencies.
- For each row: retrieves best-matching corpus paragraph for the prompt, then checks word overlap between the response and retrieved evidence.
- Produces `corpus_match_score` (0–1, higher = better grounded).

## Leaderboard Status

- **Current Phase 1 score**: 0.602 F1 (tanked from 0.636 after BanglaBERT experiment)
- **Previous best score**: 0.636 F1 (Phase 1 features + joggota, no LLM judge)
- **Primary bottleneck**: No-context rows have weak signal. LLM judge was the intended fix but can't fit on available hardware.
- **Test set distribution**: 54% with context, 46% without. Context-aware features are our best lever.

## What Actually Happened

### TigerLLM-9B OOM
Kaggle T4 (16GB) could not fit TigerLLM-9B even at 4-bit quantization. The model loads at ~5.5GB but with tokenizer buffers, CUDA context, and VRAM fragmentation, it consistently OOMed. This killed the LLM judge approach.

### BanglaBERT Failure (CV 0.823 → Actual 0.602)
Attempted as a GPU-free fallback. Fine-tuned locally on CPU using 7.5K pairs extracted from BenHalluEval:
- 5,018 hallucinated (label=0): mostly gibberish, non-Bengali tokens, obvious nonsense
- 2,509 faithful (label=1): 90% clean but 10% mislabeled

**Why it failed**: The BenHalluEval hallucinated samples were TOO easy. They contained English tokens, mixed scripts, and obvious non-answers. BanglaBERT learned to detect "non-Bengali token" as its primary signal — not actual faithfulness. On the diverse competition test set (summarization, reasoning, code-mixed), this shortcut failed catastrophically. CV F1 of 0.823 was a mirage because the 299 training samples came from the same easy-QA distribution.

### Why the Original Notebook's BanglaBERT Worked
The `banglabert-train-m.ipynb` (competition authors) used:
- **81K training pairs** from HuggingFace (30× more data)
- **Qwen 32B judge** to generate hard pseudo-labeled negatives
- **2-seed bagging** (two independent runs, averaged)
- **NLI + SQuAD + IndicQA + BanglaRQA** as additional training sources

We had none of this. Training on only 2.5K easy BenHalluEval pairs taught the model a cheap surface pattern, not real discrimination.

## Next Steps: Recovery Plan

### Phase 1: Revert BanglaBERT Damage ✅ (already diagnosed)
Current `submission_pipeline.py` already uses `sim_premise_response` fallback when BanglaBERT checkpoint is missing. Confirm no stale checkpoint exists.

### Phase 2: Better Context-Aware Features (highest ROI)
The current `nli_ctx_entail` does `context.fillna(prompt)` — this dilutes the signal for the 54% of rows that DO have context. Fix:
- **Separate NLI paths**: For context rows, run NLI(response, context) to check factual grounding. For no-context rows, keep the existing prompt-based approach.
- **TF-IDF context-response overlap**: Direct word overlap between response and context paragraph — more precise than LaBSE for fact-checking against a source.
- **Context containment**: Does the response text appear verbatim (or near-verbatim) in the context? If yes → high confidence faithful.

### Phase 3: Response Quality Heuristics
- `resp_is_refusal`: contains phrases like "আমি জানি না", "দুঃখিত", "ক্ষমা করবেন"
- `resp_has_code_switch`: ratio of non-Bengali words to total words
- `resp_repetition_score`: n-gram repetition within the response
- `resp_is_question`: response ends with ? (deflection pattern)

### Phase 4: Better XGBoost Configuration
Current `max_depth=3, n_estimators=300` is very conservative. Try:
- `max_depth=5`, `n_estimators=500` with early stopping on 20% validation
- Per-band threshold calibration (context vs no-context, task type)

### Phase 5: Simple Data Augmentation (no model needed)
For the 299 training samples:
- Synonym substitution using a Bengali synonym list
- Slight word-order shuffling (creates varied surface forms)
- This doubles the training set to ~600 without needing a judge model

## Key Constraints (from `LEGAL_DOCS.txt` & `some_catches.txt`)

- **Phase 2 GPU**: Single P100 or 2×T4, under 9 hours total runtime, under 50GB disk
- **No internet at inference** — everything must be offline
- **Open-weight models only** — no paid APIs
- **~5,000 test rows in Phase 2** (2× Phase 1) — inference efficiency matters
- **Phase 2 source domains** (broader than Phase 1):
  - Common Crawl Bengali text
  - Bengali Wikipedia / Banglapedia
  - Government websites (bangladesh.gov.bd, Citizen Charter)
  - ebanglalibrary.com
  - Major Bengali newspapers (last 5 years)
  - bdlaws.minlaw.gov.bd (legislation)
  - NCTB textbooks
- **Methodology papers**: BenHalluEval + LREC 2026 idiom paper — Phase 2 test data contains NO samples from these datasets

## Models Used (all open-weight, Phase 2 compliant)

| Model | Size | Role | Status |
|---|---|---|---|
| mDeBERTa-v3-base-mnli-xnli | ~280M | NLI entailment/contradiction | ✅ Working |
| LaBSE | ~470M | Sentence embeddings + cosine sim | ✅ Working |
| TigerLLM-9B-it | 9B | LLM-as-a-Judge | ❌ OOM on T4 |
| BanglaBERT-large | 350M | Cross-encoder classifier | ❌ Made scores worse (0.602) |
| XGBoost | — | Meta-classifier / feature fusion | ✅ Working |
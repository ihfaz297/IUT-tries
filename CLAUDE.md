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

### Phase 1.5 — VRAM Flush
Critical step. Without this, loading the 9B model causes OOM.

### Phase 2 — TigerLLM Judge (no-context rows only)
- **TigerLLM-9B-it** (`md-nishat-008/TigerLLM-9B-it`): Loaded in **4-bit quantization** (`BitsAndBytesConfig`).
- Only processes rows where `has_context == 0`.
- Dynamically categorizes each prompt into `code_mixed`, `vocabulary`, `math`, or `general_knowledge`.
- Uses category-specific Bengali system prompts with anti-stereotype guardrails.
- Extracts logit probability of generating `'1'` (Faithful) vs `'0'` (Hallucinated).
- After scoring, model is deleted and GPU flushed again.

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
  - `tigerllm_faithful_prob` — LLM judge probability

## File Map

| File | Description |
|---|---|
| `submission_pipeline.py` | **Main inference pipeline** — 3-phase architecture |
| `joggota_core.py` | Deterministic rules engine + Form Engine + Mini-RAG retriever |
| `build_corpus.py` | Wikipedia crawler — builds `offline_corpus.json` |
| `offline_corpus.json` | ~242 paragraphs from Bengali Wikipedia (Constitution, history, literature, etc.) |
| `converter.py` | Converts `submission_pipeline.py` → `submission_kaggle.ipynb` |
| `submission_kaggle.ipynb` | Auto-generated Kaggle-ready notebook |
| `dataset samples.json` | 299 labeled examples — the only supervised signal |
| `test set.csv` | 2,516 rows, no labels — submit predictions for these |
| `sample submission.csv` | Format: `id,label` |
| `question_in_hand.txt` | Competition overview |
| `some_notes.txt` | Organizer notes — winning approach hints |
| `some_catches.txt` | Phase 2 preview — source domains, corpus question, methodology refs |
| `LEGAL_DOCS.txt` | Full competition rules |
| `freelance.txt` | Example of cultural stereotype hallucination (Humayun Ahmed vs Taslima Nasrin) |

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

## Offline Corpus (`offline_corpus.json`)

Built by `build_corpus.py` from Bengali Wikipedia API. Currently covers:
- **Constitution & Law**: বাংলাদেশের সংবিধান, ডিজিটাল নিরাপত্তা আইন, সুপ্রিম কোর্ট, জাতীয় সংসদ
- **Liberation War & History**: স্বাধীনতা যুদ্ধ, মুক্তিযুদ্ধ, ভাষা আন্দোলন
- **Literature**: রবীন্দ্রনাথ, নজরুল, হুমায়ূন আহমেদ, জীবনানন্দ, মধুসূদন, বঙ্কিম, বিদ্যাসাগর, শরৎচন্দ্র, সুকান্ত, তসলিমা নাসরিন
- **Geography & Civics**: বাংলাদেশের অর্থনীতি, কক্সবাজার, চট্টগ্রাম, ঢাকা, পদ্মা সেতু, সুন্দরবন
- **Science**: পদার্থবিজ্ঞান, রসায়ন, জীববিজ্ঞান, গণিত, জগদীশ চন্দ্র বসু, সত্যেন্দ্রনাথ বসু

> **Note**: Some pages hit Wikipedia's rate limit (429). Re-running `build_corpus.py` with the built-in 1.5s delay fills more pages each time. Target: ~70+ pages, ~500+ paragraphs.

## TigerLLM Judge Design

### Category-Specific Prompts (Bengali)
Each prompt category gets a tailored system prompt to maximize detection accuracy:
- **code_mixed**: Handles Bangla-English混合 text without penalizing language mixing
- **vocabulary**: Checks for correct শব্দ/বাগধারার অর্থ
- **math**: Verifies exact numerical correctness
- **general_knowledge**: Strict fact-checker with **anti-stereotype guardrail** — explicitly warns TigerLLM not to be influenced by author/book name stereotypes (e.g., confusing Humayun Ahmed's writing with Taslima Nasrin's)

### Logit Extraction (not free-generation)
We do NOT generate text from TigerLLM. Instead:
1. Build the prompt with `apply_chat_template`.
2. Run a single forward pass.
3. Extract logits for token IDs of `'0'` and `'1'`.
4. Apply softmax to get `P(faithful)` as a continuous feature for XGBoost.

This is much faster than generating text and prevents the model from rambling.

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

## Current Status & Known Gaps

### ✅ Implemented
- Full 3-phase pipeline with VRAM management
- TigerLLM-9B 4-bit judge with category-specific Bengali prompts
- Joggota deterministic rules (spelling, math, idioms)
- Offline corpus retriever (TF-IDF mini-RAG from Bengali Wikipedia)
- Auto-converter to Kaggle notebook format
- GitHub push workflow

### ⚠️ Gaps / TODO
- **Corpus coverage**: Wikipedia rate-limiting means ~30 pages still missing. Need to re-run `build_corpus.py` or split into smaller batches.
- **Cross-lingual consistency**: Not yet implemented. Translate Bengali → English, verify with English model, use disagreement as feature.
- **bdlaws.minlaw.gov.bd**: Legislative text is NOT in the corpus yet (it's not on Wikipedia). Need a dedicated scraper or manual entry.
- **Cached frontier-model outputs**: Organizers promised GPT-4o/Claude/Gemini responses on the sample split but haven't released them yet.
- **Per-band threshold calibration**: XGBoost currently uses a single 0.5 threshold. Should tune per task type.
- **CV validation**: Need to run 5-fold CV on the 299-sample training set to measure actual F1 improvement from TigerLLM + corpus features.

## Leaderboard Status

- **Current Phase 1 score**: 0.636 F1 (136th place)
- **Primary cause**: Pipeline had zero signal for no-context rows. TigerLLM integration + corpus grounding should significantly improve this.

## Models Used (all open-weight, Phase 2 compliant)

| Model | Size | Role | Quantization |
|---|---|---|---|
| mDeBERTa-v3-base-mnli-xnli | ~280M | NLI entailment/contradiction | None (FP32) |
| LaBSE | ~470M | Sentence embeddings + cosine sim | None (FP32) |
| TigerLLM-9B-it | 9B | LLM-as-a-Judge for no-context rows | 4-bit NF4 |
| XGBoost | — | Meta-classifier / feature fusion | N/A |
